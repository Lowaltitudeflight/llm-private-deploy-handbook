# 02 环境勘察与准入检查

> **目标**：在动手装任何东西之前，用可执行的命令把客户环境的真实情况摸清楚，
> 并给出一个明确的 **Go / No-Go** 结论。
>
> 勘察不是"看一眼配置单"，而是**登录机器、跑命令、留证据**。

---

## 2.1 为什么必须现场实测

硬件清单和实际机器不一致是**常态**，不是意外。常见偏差：

| 偏差类型 | 真实案例 |
| --- | --- |
| 型号缩水 | 报价 A100 80G，现场 A100 40G |
| 数量不足 | 说好 8 卡，实际 6 卡在机箱内、另 2 卡在别的服务器上 |
| 显存被占 | 机器上已有其他业务在跑，可用显存只剩一半 |
| 拓扑不同 | 8 卡不是全 NVLink，而是 4+4 两组、组间走 PCIe |
| 软件栈陈旧 | 驱动是 470.x，装不了新版本 PyTorch |
| 网络受限 | 完全不出网，且有端口白名单 |

**勘察阶段的每一分钟，都能在实施阶段省下一小时。**

---

## 2.2 硬件勘察

### 2.2.1 GPU —— 最关键的 5 分钟

```bash
# 1) 卡的数量、型号、显存
nvidia-smi -L

# 2) 结构化输出，便于记进勘察表
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,serial,uuid \
           --format=csv

# 3) PCIe 链路实际状态（关键！很多"性能不达标"是这里降速）
nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,\
pcie.link.width.max,pcie.link.width.current --format=csv

# 4) 卡间拓扑（判断有没有 NVLink、是不是全互联）
nvidia-smi topo -m

# 5) NVLink 状态（如果有）
nvidia-smi nvlink -s
```

**怎么读这些结果：**

| 观察项 | 健康值 | 异常信号 |
| --- | --- | --- |
| `pcie.link.gen.current` vs `.max` | 两者相等 | current < max → **链路降速**，需查 BIOS 设置 / 转接卡 |
| `pcie.link.width.current` vs `.max` | 两者相等 | 如 max=x16 但 current=x8 → 插槽带宽被分 |
| `topo -m` 中 8 卡同一行 | `NV8`/`NV12` 等 | 出现 `SYS` 或 `PHB` → 跨 NUMA / 只走 PCIe，TP 通信会慢 |
| `memory.total` | 与采购一致 | 少于清单 → 型号被换 |
| 已有进程占用 | 无其他进程 | 有别的业务 → 需协商显存归属 |

> ⚠️ **`nvidia-smi topo -m` 是最容易被忽略、又最影响性能的一步。**
> 如果 8 卡分成 4+4 两组、组内 NVLink、组间走 PCIe，
> 那么 TP=8 的通信会跨组走慢链路，吞吐可能腰斩。
> 这种情况建议方案改成 **TP=4 × DP=2**（见 [03 章 3.5](03-solution-design.md)）。

**查看是否有其他进程占用显存：**

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### 2.2.2 CPU / 内存 —— 影响数据预处理与 tokenize

```bash
lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node\(s\)|Thread"
numactl --hardware 2>/dev/null || echo "numactl 未安装"
free -h
```

**关注点**：

- **CPU 核数**：要够 tokenize/detokenize 与请求处理。建议 **每张 GPU 至少 4~8 个物理核**。
  核数太少时，GPU 会空闲等 CPU，表现为"利用率上不去"。
- **NUMA 节点**：多路 CPU 时，要确保 GPU 与其"直属"的 CPU/内存同 NUMA 节点，
  否则跨节点访存会拖慢。启动容器时可用 `--cpuset-cpus` / `--numa-node` 绑定。
- **内存**：经验值 **≥ 单卡显存 × 卡数 × 1.5**。加载模型时权重先落内存再上卡，
  内存不够会导致加载失败或疯狂 swap。

### 2.2.3 存储 —— 模型加载速度的决定因素

```bash
# 磁盘类型与剩余空间
lsblk -d -o NAME,SIZE,ROTA,MODEL     # ROTA=1 表示机械盘
df -h

# 实测读写带宽（安装 fio 后）
fio --name=seqread --rw=read --bs=1M --size=4G --filename=/data/fio.test --direct=1
```

**关注点**：

| 项 | 要求 | 说明 |
| --- | --- | --- |
| **模型盘空间** | ≥ 模型权重的 2.5 倍 | 要存原始权重、量化后权重、可能的多个版本 |
| **读写速度** | 顺序读 ≥ 1 GB/s（NVMe） | 144 GB 的权重走 SATA SSD（500 MB/s）要 ~5 分钟才能加载完 |
| **共享存储** | 多机部署时必须是共享盘或本地同步 | 否则每台机器各存一份，版本容易不一致 |

> ✅ **实战技巧**：把模型放在 **本地 NVMe**，不要放在网络存储（NFS/CEPH）上。
> 网络存储上加载 144 GB 权重可能从 1 分钟变成 20 分钟，
> 而且会因为超时导致加载失败。多机场景用 `rsync` 提前分发到各节点本地盘。

### 2.2.4 网络 —— 多机部署的生死线

```bash
# 网卡列表与速率
ip -br link show
for d in $(ls /sys/class/net | grep -v lo); do \
  echo "$d: $(cat /sys/class/net/$d/speed 2>/dev/null) Mb/s"; done

# InfiniBand（如果有）
ibstat 2>/dev/null | head -20
ibv_devinfo 2>/dev/null | grep -E "hca_id|state|rate"

# 实测带宽（两机对打）
# 服务端: iperf3 -s
# 客户端: iperf3 -c <server_ip> -t 30 -P 8
```

**判断标准**：

| 场景 | 要求 |
| --- | --- |
| 单机多卡 | 卡间靠 NVLink / PCIe，网络只影响外部访问 |
| 多机 TP | **必须** ≥ 100 Gb IB 或 RoCE，25 Gb 以太网跑不了 TP=16 |
| 多机 PP | 通信量小，25~100 Gb 可接受 |
| 纯 DP（多副本） | 只有请求分发，普通千兆/万兆即可 |

> ⚠️ 多机 TP 用普通以太网会**卡到怀疑人生**。25 Gb 网跑 TP=16 时，
> 通信时间会超过计算时间。这种情况请改用 **PP 或 DP**。

**还要确认**：

- 是否走 **RoCE**？如果是，检查 PFC/ECN 配置，丢包会让性能剧烈抖动。
- MTU：建议 **9000（Jumbo Frame）**，可显著降低协议开销。
- 防火墙与端口白名单：模型服务端口、监控端口（9090/3000）、节点间通信端口
  是否放通。

### 2.2.5 供电与散热

```bash
# GPU 功耗上限与当前功耗
nvidia-smi --query-gpu=index,power.limit,power.draw,power.max_limit --format=csv

# 温度
nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory \
           --format=csv
```

- **功耗**：8 卡 H100 满载约 5~7 kW，加上 CPU/整机可能到 10 kW。
  确认机柜配电与 PDU 是否够，**是否需要降功耗运行**（`nvidia-smi -pl <瓦数>`）。
- **温度**：待机应 < 50°C。若待机就 70°C+，说明机房散热有问题，
  满载必然触发降频（thermal throttling），压测数据会很难看。
- **风扇**：整机机型注意风扇转速策略，部分机型满载会非常吵，客户可能要求限速。

### 2.2.6 ECC 与健康状态 —— 上线前必须查

```bash
nvidia-smi -q -d ECC | grep -E "GPU|ECC Mode|Volatile|Pending|Uncorrectable"
nvidia-smi -q | grep -iE "Retired|Remapped|Xid"
```

- **ECC Uncorrectable > 0**：显存有不可纠正错误，**该卡不可用于生产**，
  应立即报修。
- **Retired Pages / Remapped Rows > 0**：显存已有坏页被隔离，属于老化迹象，
  记录在案并持续观察。
- 有条件时跑一次硬件自检：`dcgmi diag -r 2`（需 DCGM）。

---

## 2.3 软件栈勘察

### 2.3.1 基础信息采集

```bash
# OS 与内核
cat /etc/os-release | head -3
uname -r

# 驱动与 CUDA
nvidia-smi | head -4          # 右上角显示 Driver Version / CUDA Version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
nvcc --version 2>/dev/null || echo "未装 CUDA Toolkit（仅驱动也可运行容器）"

# 容器
docker --version
docker info 2>/dev/null | grep -E "Runtime|Cgroup|Storage"
nvidia-container-cli -V 2>/dev/null || echo "未装 nvidia-container-toolkit"

# Python 环境（若走裸机部署）
python3 --version
pip3 list 2>/dev/null | grep -iE "^torch|^vllm|^transformers"
```

### 2.3.2 版本对应关系（勘察时用来判断"能不能装"）

**驱动 ↔ 支持的 CUDA 版本**（驱动向下兼容，装新不装旧）：

| 驱动版本（Linux） | 支持的最高 CUDA |
| --- | --- |
| ≥ 520.61.05 | 11.8 |
| ≥ 525.60.13 | 12.0 |
| ≥ 530.30.02 | 12.1 |
| ≥ 535.54.03 | 12.2 |
| ≥ 550.54.14 | 12.4 |
| ≥ 555.42.02 | 12.5 |
| ≥ 560.28.03 | 12.6 |
| ≥ 570.26 | 12.8 |

> 注意区分两个"CUDA 版本"：
> - `nvidia-smi` 右上角的 CUDA Version = **驱动支持的最高版本**（能力上限）
> - `nvcc --version` = **本机 CUDA Toolkit 版本**（实际编译用的）> 
> 容器方案里，宿主机只需要驱动，容器内自带 CUDA Runtime，
> 两者**只要满足"驱动 ≥ 容器所需"即可**，不需要版本号完全相等。

**PyTorch ↔ CUDA ↔ vLLM**（近似对应，**以官方发布说明为准**）：

| torch | 可选 CUDA 构建 | vLLM 大版本 |
| --- | --- | --- |
| 2.4 / 2.5 | cu121 / cu124 | 0.6.x |
| 2.6 | cu118 / cu124 / cu126 | 0.8.x |
| 2.7 | cu118 / cu126 / cu128 | 0.9.x |
| 2.8 | cu126 / cu128 | 0.10.x |
| 2.9 | cu126 / cu128 / cu130 | 0.11.x |

> ⚠️ **勘察时的实操判断**：
> 查到驱动是 535.x → 最高支持 CUDA 12.2 → 只能选 **cu121** 的 torch 构建。
> 若想用需要 cu128 的新版 vLLM，**必须先升级驱动**，而这通常要走客户变更流程。
> **这一步的结论要尽早给出**，别等到实施当天才发现要升驱动。

### 2.3.3 常见"环境陷阱"

| 陷阱 | 现象 | 处置 |
| --- | --- | --- |
| 已装旧版驱动 | `nvidia-smi` 报 version mismatch | 卸载重装或走 DKMS |
| 已装 NVIDIA 开源驱动 | 部分特性异常 | 确认 `nvidia-smi -q` 的 Driver Type |
| 已有 CUDA 多版本共存 | 编译时选错 | 用 `update-alternatives` 管理 |
| conda 环境污染 | `import torch` 报 undefined symbol | 用独立 venv / 容器隔离 |
| `docker` 但没装 toolkit | 容器内看不到 GPU | 装 `nvidia-container-toolkit` 并重启 docker |
| cgroup v1/v2 不一致 | 容器资源限制失效 | 记录在案，避免依赖内存限制 |
| SELinux / AppArmor | 挂载卷被拒 | 确认策略或加 `--security-opt` |

---

## 2.4 出网策略 —— 决定你能不能装东西

这是**勘察时最容易被漏掉、实施时最致命**的一项。

```bash
# 1) 能否出外网
curl -sI --max-time 8 https://pypi.org | head -1
curl -sI --max-time 8 https://huggingface.co | head -1
curl -sI --max-time 8 https://docker.io | head -1

# 2) 有没有内网镜像源
cat /etc/pip.conf 2>/dev/null || cat ~/.pip/pip.conf 2>/dev/null
cat /etc/docker/daemon.json 2>/dev/null
env | grep -i proxy

# 3) DNS 是否可用
getent hosts pypi.org || echo "DNS 解析失败"
```

**四种典型环境与对应策略：**

| 环境 | 特征 | 策略 |
| --- | --- | --- |
| **A. 完全出网** | 一切正常 | 直接装，仍建议固化版本号 |
| **B. 仅内网源** | 有 Nexus/私有 Harbor | 配置 pip/docker 指向内网源；确认源里**有需要的镜像和包** |
| **C. 完全离线** | 无任何外网 | **必须准备离线包**（见下） |
| **D. 白名单** | 只有特定域名端口通 | 提前把域名报给客户审批，留足审批时间 |

### 离线包准备清单（环境 C 必备）

在**能出网的机器**上准备好，用移动介质带进去：

```bash
# ① 容器镜像 → 导出为 tar
docker pull vllm/vllm-openai:v0.11.0
docker save vllm/vllm-openai:v0.11.0 -o vllm-openai-v0.11.0.tar
# 现场导入：docker load -i vllm-openai-v0.11.0.tar

# ② Python 包 → 下载 whl
pip download -r requirements.txt -d ./offline_wheels \
    --platform manylinux2014_x86_64 --python-version 3.11 \
    --only-binary=:all:
# 现场安装：pip install --no-index --find-links=./offline_wheels -r requirements.txt

# ③ 模型权重（用 ModelScope 更稳，国内可达）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-72B-Instruct \
    --local_dir ./Qwen2.5-72B-Instruct
# 或 huggingface 生态：
# HF_ENDPOINT=https://hf-mirror.com huggingface-cli download <repo> \
#     --local-dir <dir> --resume-download

# ④ 驱动与 CUDA 安装包
# 从 NVIDIA 官网按 OS/架构下载 .run 或 .deb
```

> ⚠️ **离线场景的两个常见翻车点**：
> 1. **模型分片没收全**。`huggingface-cli` 中断后重跑要加 `--resume-download`，
>    并且**必须核对文件数量和总大小**（见 [04 章 4.3](04-deployment.md)）。
> 2. **忘了带 `nvidia-container-toolkit` 的安装包**。现场装不了容器运行时，
>    GPU 在容器里永远不可见。

---

## 2.5 信创 / 国产化环境勘察要点

如果客户是信创环境，**勘察项要额外增加**，且很多常规经验会失效。

### 2.5.1 国产加速卡

| 厂商 | 产品 | 软件栈 | 生态成熟度 | 推理框架 |
| --- | --- | --- | --- | --- |
| 华为 | 昇腾 910B/910C | CANN + MindSpore | 较成熟 | **MindIE**、vllm-ascend、torch_npu |
| 海光 | DCU（深算） | DTK（类 ROCm） | 中等 | vLLM 的 ROCm 分支可移植 |
| 寒武纪 | MLU370/590 | Neuware | 中等 | 官方推理工具链 |
| 摩尔线程 | MTT S 系列 | MUSA | 起步 | vLLM-MUSA 分支 |
| 天数智芯 | 天垓 | Iluvatar CoreX | 起步 | 官方适配 |
| 昆仑芯 | R/K 系列 | 昆仑软件栈 | 起步 | 官方适配 |

> ⚠️ **三件必须现场确认的事**：
> 1. **模型支持清单**。国产卡的推理框架往往只支持特定架构（如仅 Llama 系），
>    你选的 Qwen / GLM 不一定在列。**要在 P2 就拿到官方支持矩阵。**
> 2. **量化方案不同**。昇腾上用 W8A8 / W4A16 等与 NVIDIA 的 AWQ/GPTQ 不通用，
>    需要重新转换、重新验证精度。
> 3. **性能量级不同**。把 NVIDIA 上的吞吐数字直接当成国产卡的预期，会严重失准。
>    **必须拿到目标卡上的实测值或厂商案例数据。**

### 2.5.2 国产操作系统

```bash
# 确认发行版
cat /etc/os-release
# 麒麟: Kylin Linux Advanced Server
# 统信: UnionTech OS Server
# openEuler: openEuler

# 确认内核与 glibc（影响 wheel 兼容性）
uname -r
ldd --version | head -1
```

**常见问题**：

| 问题 | 表现 | 处置 |
| --- | --- | --- |
| glibc 版本低 | 无法安装 manylinux_2_28 的 wheel | 找 manylinux2014 构建，或源码编译 |
| 缺内核头文件 | 驱动 DKMS 编译失败 | 安装 `kernel-devel` / `kernel-headers` |
| 自研安全模块 | 某些系统调用被拦 | 找系统方要白名单策略 |
| 无 `systemd` | 服务守护方式不同 | 改用 openrc / 自定义 supervisor |

### 2.5.3 国产 CPU

鲲鹏（ARM64）、飞腾（ARM64）、龙芯（LoongArch）、兆芯/海光（x86_64）。

> ⚠️ **ARM64 是最大的坑**：很多 Docker 镜像只有 `amd64` 版本。
> 勘察时必须确认 `uname -m`，并**提前找到对应架构的镜像**（或自行构建）。
> 如果只有 amd64 镜像，需要走 `qemu` 模拟，性能会不可接受 —— 这基本等于
> 需要重新做一套 ARM 镜像，工作量要提前报给项目经理。

---

## 2.6 准入检查清单（Go / No-Go）

勘察结束前逐项确认。**只要有一项"必须满足"未通过，结论就是 No-Go。**

### 必须满足（Go 的门槛）

| # | 检查项 | 判定 | 备注 |
| :---: | --- | :---: | --- |
| 1 | GPU 型号与数量符合采购/方案 | ⬜ | 实跑 `nvidia-smi -L` |
| 2 | 单卡可用显存 ≥ 方案要求 | ⬜ | 排除其他业务占用 |
| 3 | PCIe 链路无降速（current = max） | ⬜ | — |
| 4 | 卡间拓扑满足并行策略要求 | ⬜ | 尤其 TP 是否跨慢链路 |
| 5 | ECC Uncorrectable = 0 | ⬜ | 有值则该卡不可用 |
| 6 | 驱动版本 ≥ 目标框架要求 | ⬜ | 不够需先升级 |
| 7 | 模型盘剩余空间 ≥ 权重 × 2.5 | ⬜ | 含量化版本 |
| 8 | 内存 ≥ 单卡显存 × 卡数 × 1.5 | ⬜ | — |
| 9 | 出网策略明确，离线包已备 | ⬜ | 环境 C 必备 |
| 10 | 服务端口 + 监控端口可放通 | ⬜ | 与客户网络组确认 |
| 11 | 具备 root 或等价权限 | ⬜ | 装驱动/容器运行时需要 |
| 12 | 硬件质保与报修渠道明确 | ⬜ | 出问题找谁 |

### 有条件满足（需明确处置方案）

| # | 检查项 | 风险 | 处置 |
| :---: | --- | --- | --- |
| 13 | CPU 核数偏少 | GPU 利用率上不去 | 降低并发预期 / 申请加核 |
| 14 | 模型盘为 HDD | 加载慢、超时 | 迁移到 NVMe |
| 15 | 网络带宽 < 100 Gb | 多机 TP 不可行 | 改 PP/DP |
| 16 | 机房温度偏高 | 满载降频 | 限功耗运行，性能目标下调 |
| 17 | glibc/架构不匹配 | 需要重做镜像 | 追加工作量与工期 |
| 18 | 无监控系统 | 无法观测 | 需要部署 Prometheus + Grafana |

### 结论

```
Go / No-Go 判定：__________

判定依据（未通过项）：
  ________________________________________________

风险与处置：
  ________________________________________________

勘察人：__________  日期：__________
客户确认：__________  日期：__________
```

---

## 2.7 环境勘察表（可直接填写）

> 完整可打印版本见 [`templates/`](../templates/) 与 [附录 B](../appendix/b-checklists.md)。

| 分类 | 项 | 实测值 | 是否达标 |
| --- | --- | --- | :---: |
| **GPU** | 型号 / 数量 | | |
| | 单卡显存 | | |
| | 驱动版本 | | |
| | CUDA 支持上限 | | |
| | PCIe Gen / Width（max → current） | | |
| | 卡间拓扑（`topo -m` 摘要） | | |
| | ECC Uncorrectable | | |
| | 已有占用进程 | | |
| **CPU** | 型号 / 物理核数 / 路数 | | |
| | NUMA 节点 | | |
| **内存** | 总量 | | |
| **存储** | 模型盘类型 / 剩余空间 | | |
| | 顺序读带宽 | | |
| **网络** | 网卡速率 / 是否 IB | | |
| | MTU | | |
| | 出网策略 | | |
| **软件** | OS / 内核 | | |
| | 容器运行时 / toolkit | | |
| | Python 环境 | | |
| **机房** | 机柜配电余量 | | |
| | 环境温度 | | |
| **非技术** | 联系人 / 联系方式 | | |
| | root 权限获取路径 | | |
| | 变更窗口（可停机时段） | | |
| | 硬件报修渠道 | | |

> ⚠️ **"变更窗口"这一栏千万别漏**。生产环境往往只能在夜间或周末做变更，
> 如果实施时才发现"白天不能重启"，整个工期会往后拖一周。

---

**下一章**：[03 部署方案设计](03-solution-design.md) —— 拿到勘察结果后，怎么把需求变成方案。
