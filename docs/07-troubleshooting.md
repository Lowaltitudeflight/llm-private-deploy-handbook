# 07 运维与故障排查

> **目标**：交付之后，客户自己的人能看懂、能巡检、能处理常见问题，
> 出了大问题也知道该收集什么、找谁。
>
> 本章包含日常巡检制度、监控指标与告警阈值、**6 棵故障树**、应急预案。

---

## 7.1 排查总则：先分层，再深入

**90% 的排查时间浪费在"跳层"上** —— 一上来就怀疑模型，其实是网线松了。

```
第 1 层  硬件      GPU 是否存在、是否掉卡、ECC、温度、功耗、供电
第 2 层  驱动/内核  nvidia-smi 能不能跑、dmesg 有没有报错
第 3 层  容器运行时 容器能不能看到 GPU、共享内存够不够
第 4 层  引擎/服务 进程在不在、日志说什么、参数是什么
第 5 层  网络/网关 端口通不通、代理配置、超时设置
第 6 层  客户端    客户端自己的 CPU/内存/代码

排查顺序：从第 1 层往上逐层确认，不要跳。
```

**每一次排查，先跑这一组命令（30 秒建立全局认知）：**

```bash
# ① GPU 还在吗？
nvidia-smi

# ② 内核层面有没有报错？
dmesg -T | tail -50
dmesg -T | grep -iE "nvrm|xid|nvidia|oom-kill" | tail -30

# ③ 服务进程/容器还在吗？
docker ps -a | head
systemctl status vllm 2>/dev/null | head -20

# ④ 服务日志最后说了什么？
docker logs --tail 100 <container> 2>&1 | tail -50

# ⑤ 端口通吗？
curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://localhost:8000/health
```

> ✅ **把这 5 条存成 `triage.sh` 放在运维手册里**，
> 客户运维出问题时第一件事就是跑它，然后把输出发给你。
> **这一招能省掉大量"来回问三次才拿到信息"的沟通成本。**

---

## 7.2 日常巡检

### 7.2.1 巡检频次与项目

| 频次 | 项目 |
| --- | --- |
| **每日** | GPU 状态与告警、服务是否在线、错误率、显存趋势、磁盘剩余 |
| **每周** | 长周期延迟趋势、慢请求 TopN、日志异常关键字、温度与功耗峰值 |
| **每月** | ECC 计数变化、驱动/固件版本、备份可用性、容量水位评估 |

### 7.2.2 每日巡检脚本（可直接用）

```bash
#!/usr/bin/env bash
# daily_check.sh —— 每日巡检，输出一份简报
echo "===== 巡检时间: $(date '+%F %T') ====="

echo "--- 1. GPU 状态 ---"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,\
memory.used,memory.total,power.draw,power.limit \
--format=csv

echo "--- 2. ECC 与健康（应为 0）---"
nvidia-smi -q -d ECC 2>/dev/null | grep -iE "UNCORRECTABLE|Volatile" | head -8
nvidia-smi -q 2>/dev/null | grep -iE "Retired|Remapped" | head -4

echo "--- 3. 内核报错（Xid / 掉卡）---"
dmesg -T 2>/dev/null | grep -iE "xid|nvrm.*(error|fail)" | tail -10 \
  || echo "  无（或需 root 权限）"

echo "--- 4. 服务存活与延迟 ---"
curl -s -o /dev/null -w "  /health http_code=%{http_code} 耗时=%{time_total}s\n" \
  --max-time 10 http://localhost:8000/health

echo "--- 5. 关键业务指标（需 vLLM /metrics）---"
curl -s --max-time 5 http://localhost:8000/metrics 2>/dev/null | \
  grep -E "^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc)" \
  || echo "  指标不可达"

echo "--- 6. 磁盘 ---"
df -h / /data 2>/dev/null | grep -v tmpfs

echo "--- 7. 温度是否触发降频 ---"
nvidia-smi -q -d PERFORMANCE 2>/dev/null | \
  grep -A2 "Clocks Throttle Reasons" | head -12

echo "===== 巡检结束 ====="
```

### 7.2.3 关键趋势要盯的四个数

| 指标 | 看什么 | 危险信号 |
| --- | --- | --- |
| **显存占用** | 空闲时是否逐渐变高 | 持续上升 → 泄漏 |
| **GPU 温度** | 峰值是否接近阈值 | 待机 > 60°C → 散热问题 |
| **错误率** | 是否长期 > 0 | 任何持续错误都要查 |
| **队列长度** | 是否长期有积压 | 长期 > 0 → 容量不足 |

---

## 7.3 监控指标体系

### 7.3.1 两套指标，缺一不可

| 来源 | 覆盖 | 采集方式 |
| --- | --- | --- |
| **vLLM `/metrics`** | 服务层：请求、延迟、队列、KV 缓存 | Prometheus 抓 `/metrics` |
| **DCGM Exporter** | 硬件层：GPU 利用率、显存、温度、功耗、Xid | `nvidia/dcgm-exporter` 容器 |

> ⚠️ **只有 vLLM 指标不够**：你无法判断"慢"是模型的问题还是卡的问题。
> **只有 DCGM 指标也不够**：你无法知道有多少请求在排队。
> 两套都要。

### 7.3.2 vLLM 关键指标

| 指标 | 含义 | 说明 |
| --- | --- | --- |
| `vllm:num_requests_running` | 正在处理的请求数 | 运行的并发度 |
| `vllm:num_requests_waiting` | **排队中的请求数** | **最重要的指标之一** |
| `vllm:gpu_cache_usage_perc` | KV 缓存使用率 | 0~1 |
| `vllm:prefix_cache_hit_rate` | 前缀缓存命中率 | 越高越省算力 |
| `vllm:time_to_first_token_seconds` | TTFT 直方图 | 算 P50/P95/P99 |
| `vllm:time_per_output_token_seconds` | TPOT 直方图 | — |
| `vllm:e2e_request_latency_seconds` | 端到端延迟直方图 | — |
| `vllm:request_success_total` | 成功请求计数 | 按 finish_reason 分标签 |
| `vllm:num_preemptions_total` | **抢占次数** | ⚠️ 大于 0 说明显存吃紧 |
| `vllm:prompt_tokens_total` / `generation_tokens_total` | token 计数 | 算吞吐与计费 |

> ⚠️ **两个最该盯的指标**：
> 1. **`num_requests_waiting`** —— 这是"用户正在等"的直接信号。
>    长期 > 0 说明容量不足。
> 2. **`num_preemptions_total`** —— 大于 0 说明 vLLM 为了让新请求进来，
>    被迫把正在处理的请求踢出去重算。**这是显存吃紧的早期信号，
>    也是延迟抖动的元凶**，但很容易被忽略。

### 7.3.3 DCGM 关键指标

| 指标 | 含义 |
| --- | --- |
| `DCGM_FI_DEV_GPU_UTIL` | GPU 利用率 |
| `DCGM_FI_DEV_FB_USED` / `FB_FREE` | 显存使用 / 剩余 |
| `DCGM_FI_DEV_GPU_TEMP` | 温度 |
| `DCGM_FI_DEV_POWER_USAGE` | 功耗 |
| `DCGM_FI_DEV_XID_ERRORS` | **Xid 错误计数（掉卡预警）** |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` | 可纠正/不可纠正 ECC 错误 |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | NVLink 带宽 |
| `DCGM_FI_DEV_SM_CLOCK` / `MEM_CLOCK` | 核心/显存频率（看降频） |

### 7.3.4 告警阈值建议

| 告警 | 阈值 | 级别 | 处置 |
| --- | --- | --- | --- |
| GPU 温度 | > 83°C 持续 5 min | P1 | 检查散热/限功耗 |
| 显存使用率（空闲时） | > 90% | P2 | 排查泄漏 |
| `num_requests_waiting` | > 10 持续 5 min | P2 | 评估扩容 |
| TTFT P99 | > 目标值 2 倍持续 10 min | P2 | 排查 |
| 错误率 | > 1% 持续 5 min | P1 | 立即排查 |
| `num_preemptions_total` | 增速 > 0 | P2 | 调小并发或调大 KV 池 |
| **Xid 错误** | **> 0** | **P0** | **立即处理（见 7.8）** |
| ECC Uncorrectable | > 0 | **P0** | 该卡隔离并报修 |
| 磁盘剩余 | < 15% | P2 | 清理日志 |
| 服务不可达 | 连续 3 次探测失败 | P1 | 检查进程 |

> ⚠️ **告警一定要有"恢复"通知**，否则运维无法判断是还在故障还是已恢复。
> 另外**告警要收敛**：不要在磁盘满了之后每分钟发一条，
> 那样运维会直接把告警静音。

---

## 7.4 故障树 A：服务起不来

```
服务无法启动
├─ A1 端口被占用
│   └─ ss -tlnp | grep 8000
│      → 换端口，或杀掉旧进程
│
├─ A2 模型路径错 / 权限不足
│   └─ ls -l <model_dir>; sudo -u <run_user> ls <model_dir>
│      → 修正路径；注意容器内路径与宿主机不同
│
├─ A3 容器看不到 GPU
│   └─ docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
│      → 报 "could not select device driver" = 未装/未配 nvidia-container-toolkit
│      → 检查 /etc/docker/daemon.json 的 runtimes 配置
│
├─ A4 共享内存不足
│   └─ df -h /dev/shm        (容器内)
│      → 加 --shm-size 16g 或 --ipc=host
│      → 典型现象：多卡初始化时崩、报 NCCL 错误
│
├─ A5 驱动 / CUDA 版本不匹配
│   └─ nvidia-smi 看驱动支持的最高 CUDA；对比镜像要求的 CUDA
│      → 报 "CUDA driver version is insufficient for CUDA runtime version"
│      → 升级驱动，或换用低 CUDA 版本的镜像
│
├─ A6 权重文件不完整
│   └─ python check_model.py <model_dir>   (见 04 章 4.3)
│      → 报 "Error while deserializing header" / "unexpected EOF"
│      → 重新下载缺失分片
│
├─ A7 张量并行数不合法
│   └─ 报 "The number of attention heads is not divisible by tensor parallel size"
│      → 调整 --tensor-parallel-size 为注意力头数的约数
│
├─ A8 chat template 缺失
│   └─ 报 "chat template is not defined" / "Cannot use chat endpoint"
│      → 加 --chat-template <file>；或改用 /v1/completions（不推荐）
│
├─ A9 显存不足（启动阶段）
│   └─ 见故障树 B
│
└─ A10 systemd 启动超时被杀
    └─ 日志显示反复重启、每次都在加载模型
       → 把 TimeoutStartSec 调到 ≥ 模型加载时间
```

> ⚠️ **A10 是一个非常隐蔽的坑**：现象是"服务一直起不来"，
> 日志看起来是"加载到一半就没了"。根因是 systemd 的
> `TimeoutStartSec`（默认 90s）在模型加载完成前就把进程杀了。
> **72B 模型加载常常要 3~5 分钟。**

---

## 7.5 故障树 B：CUDA Out of Memory

**先判断是"启动时 OOM"还是"运行中 OOM"，处置完全不同。**

```
CUDA out of memory
├─ B0 先确认：是不是别的进程占了显存？
│   └─ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
│      → 有别的进程 → 协商释放，或减少本服务的 gpu-memory-utilization
│
├─ B1 启动时就 OOM
│   ├─ 日志含 "Free memory on device (X GiB) is less than desired..."
│   │   → 权重本身就装不下
│   │     处置：① 降低量化精度（FP16→INT8→AWQ）② 增大 --tensor-parallel-size
│   │           ③ 减少使用的卡数（改用更少的卡？不，是增加卡）
│   ├─ 日志含 "The model's max seq len ... larger than the maximum number
│   │   of tokens that can be stored in KV cache"
│   │   → 上下文太长，KV 装不下
│   │     处置：降低 --max-model-len（最有效）
│   └─ 日志含 "No available memory for the cache blocks"
│       → KV 池为 0
│         处置：降低 --max-model-len 或 --gpu-memory-utilization
│
├─ B2 运行中 OOM（并发上来之后）
│   ├─ KV 池被吃满
│   │   → 降低 --max-num-seqs（最直接）
│   │   → 开 --kv-cache-dtype fp8（KV 显存减半）
│   │   → 降低 --max-model-len
│   ├─ 单个超长请求
│   │   → 限制请求的 max_tokens；在网关层限制输入长度
│   └─ 显存碎片
│       → 现象：nvidia-smi 显示还有几百 MB 空闲，但就是分配失败
│       → 处置：重启服务释放；调整 --gpu-memory-utilization（略微降低）
│
└─ B3 长时间运行后逐渐 OOM
    → 疑似泄漏。对比启动时与当前的 memory.used
    → 收集 nvidia-smi 时间序列 + vLLM 日志，作为问题单附件
```

### 快速处置决策表

| 症状 | 首选动作 | 次选 | 代价 |
| --- | --- | --- | --- |
| 启动就装不下 | 加 `--tensor-parallel-size` | 换更激进的量化 | 需要更多卡 / 精度损失 |
| 上下文太长 | 降 `--max-model-len` | 开 `--kv-cache-dtype fp8` | 缩短可用上下文 |
| 并发起不来 | 降 `--max-num-seqs` | 开 KV fp8 | 降低并发能力 |
| 碎片 | 重启服务 | 调低 `gpu-memory-utilization` | 短暂中断 |

> ✅ **一个通用原则**：**OOM 时优先降 `--max-model-len`，而不是降 `--max-num-seqs`。**
> 因为长度是**单次请求**的需求（必须满足，否则请求失败），
> 并发是**整体吞吐**的指标（降了只是变慢，不影响可用性）。

---

## 7.6 故障树 C：请求卡死 / 超时

```
请求超时 / 长时间无响应
├─ C1 服务排队积压
│   └─ curl -s localhost:8000/metrics | grep num_requests_waiting
│      → 持续 > 10 → 容量不足，见 7.7
│
├─ C2 大量请求被"挂起"
│   └─ 看 vllm:num_requests_running 是否等于 --max-num-seqs
│      → 等于说明已达并发的硬上限，后续请求全在排队
│
├─ C3 单个超长请求占住槽位
│   └─ 现象：其他请求全部很慢，但只有 1 个请求在跑很久
│      → 网关层限制单请求 max_tokens 和输入长度
│      → 开 --enable-chunked-prefill 让长请求不独占
│
├─ C4 客户端超时太短
│   └─ 客户端设置的 timeout < 实际生成时间
│      → 调大客户端超时（流式场景建议 300s）
│
├─ C5 网关超时
│   └─ Nginx proxy_read_timeout 默认 60s
│      → 调到 300s；且必须 proxy_buffering off（流式）
│
├─ C6 服务进程假死（无响应但进程在）
│   └─ py-spy dump --pid <pid>   # 看卡在哪一行
│      → 若能定位到死锁/死循环，收集栈信息报 bug
│      → 临时处置：重启服务
│
└─ C7 网络问题
    └─ 在服务端本机 curl 是否正常？正常说明问题在网络链路
       → ping / traceroute / 检查防火墙与 MTU
```

> ✅ **区分"服务慢"和"网络慢"的最快方法**：
> **在服务端本机 curl 一次**。
> - 本机快、客户端慢 → 网络/网关问题
> - 本机也慢 → 服务端问题
>
> 这一步能立刻把排查范围砍掉一半。

---

## 7.7 故障树 D：性能不达标 / 性能劣化

```
性能不达标
├─ D1 没有预热
│   └─ 首次请求含 kernel 编译 / CUDA Graph 捕获
│      → 压测前预热 ≥2 次；判断方法是"第二次明显快"
│
├─ D2 PCIe 链路降速
│   └─ nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,\
│                      pcie.link.width.current,pcie.link.width.max --format=csv
│      → current < max → 查 BIOS / 转接卡 / 插槽
│
├─ D3 卡间拓扑不佳（TP 跨慢链路）
│   └─ nvidia-smi topo -m
│      → 出现 SYS / PHB → 改用 TP=4×DP=2 而非 TP=8
│
├─ D4 CPU 成为瓶颈（tokenize / 请求处理）
│   └─ 压测时看 CPU 利用率；单个 python 进程接近 100% = 瓶颈
│      → 增加 CPU 核数；调大 --max-num-batched-tokens 摊薄开销
│
├─ D5 温度降频
│   └─ nvidia-smi -q -d PERFORMANCE | grep -A3 "Clocks Throttle Reasons"
│      → 有 "HW Thermal Slowdown: Active" → 查机房散热
│      → 看 SM_CLOCK 是否明显低于 Boost 频率
│
├─ D6 未开启优化开关
│   └─ 检查 --enable-prefix-caching / --enable-chunked-prefill
│      → 见 05 章 5.4 调优矩阵
│
├─ D7 生产环境开了 --enforce-eager
│   └─ 关掉 CUDA Graph 会明显降低吞吐
│      → 生产不要开（只在调试时用）
│
├─ D8 日志 I/O 拖累
│   └─ 未加 --disable-log-requests + 日志盘慢
│      → 加上开关；docker 配置 log-opts 限大小
│
├─ D9 客户端自身是瓶颈
│   └─ 压测时客户端 CPU 先到 100%
│      → 换独立压测机；用多进程/异步
│
├─ D10 有其他任务争抢资源
│   └─ nvidia-smi 看是否有其他进程；top 看 CPU
│      → 协商错峰，或做 cpuset 隔离
│
└─ D11 相对基线劣化了
    └─ 对比"版本基线记录"（DEL-02）：驱动、参数、模型是否变过
       → 用同一份压测脚本复测，做 A/B 对照
```

> ✅ **D11 是最该有流程的一项**。很多"最近变慢了"的问题，
> 最后发现是"上周有人把驱动升了"或"参数被改过"。
> **没有版本基线记录，就无法判断"变了没有"，只能靠猜。**

---

## 7.8 故障树 E：掉卡与 Xid

**这是最严重的一类故障，通常意味着硬件问题。**

### 7.8.1 现象与定位

```bash
# 现象 1：nvidia-smi 报错或卡消失
nvidia-smi
#  → "Unable to determine the device handle for GPU 3"
#  → "No devices were found"
#  → 只列出 7 张卡（原本 8 张）

# 现象 2：内核日志
dmesg -T | grep -i xid
sudo journalctl -k | grep -iE "xid|nvrm" | tail -50

# 现象 3：DCGM 告警
#  DCGM_FI_DEV_XID_ERRORS > 0
```

### 7.8.2 常见 Xid 含义与处置

| Xid | 含义 | 严重度 | 处置建议 |
| ---: | --- | --- | --- |
| 13 | Graphics Engine Exception | 中 | 记录；若频繁出现需排查驱动/负载 |
| 31 | GPU memory page fault | 中高 | 可能是应用越界访问；结合应用日志 |
| 43 | GPU stopped processing | 高 | 记录并观察；频繁出现需报修 |
| 48 | Double Bit ECC Error | **极高** | **该卡隔离、立即报修** |
| 62 | Internal micro-controller breakpoint | 高 | 常与供电/温度相关；检查供电与散热 |
| 63 / 64 | ECC 页退役 / 行重映射事件 | 中高 | 记录；关注趋势，频繁则报修 |
| 74 | NVLink 错误 | 高 | 检查 NVLink 连接；可能需重插/报修 |
| **79** | **GPU has fallen off the bus** | **极高** | **掉卡。需重启机器；频繁出现必须报修/换卡** |
| 92 / 94 | Single-bit ECC 错误率过高 | 中高 | 记录；趋势性增长需报修 |
| 95 | Uncontained error | **极高** | 需重启；报修 |
| 119 / 120 | GSP RPC 超时 / GSP 错误 | 高 | 常见于驱动问题；升级驱动或报修 |
| 140 | ECC unrecoverable error | **极高** | 该卡不可用，报修 |

> ⚠️ **Xid 编码含义以 NVIDIA 官方 Xid 文档为准**，上表用于**快速判断严重度**。
> 排查时**一定要把原始 `dmesg` 输出保存下来**作为报修依据。

### 7.8.3 掉卡的处置流程

```
① 立即隔离
   - 从服务中摘除该卡（改 --tensor-parallel-size 或 --gpus 列表）
   - 若无法摘除，降级为"减少卡数继续服务"或"停止服务"

② 采集证据（报修必需，缺一样都可能被推诿）
   - dmesg -T | grep -i xid 的完整输出
   - nvidia-smi -q 的完整输出
   - nvidia-smi -q -d ECC,PERFORMANCE,TEMPERATURE,POWER
   - 服务日志中故障前后的片段
   - 故障时间点、当时的负载情况

③ 尝试恢复
   - 优先：重启服务
   - 其次：重置 GPU（nvidia-smi --gpu-reset -i <idx>，需无进程占用）
   - 最后：重启机器

④ 报修
   - 提交证据给硬件厂商/集成商
   - 明确要求：更换该卡或提供诊断结论

⑤ 记录
   - 写入运维日志：时间、卡号、Xid、处置、结果
   - 若同一张卡重复出问题，要求更换
```

> ✅ **一条重要经验**：**Xid 79（掉卡）和 Xid 48/140（不可纠正 ECC）
> 基本可以判定是硬件问题**，不要浪费时间在应用层排查。
> **但一定要先采证据再重启** —— 重启会清掉 dmesg 的关键信息。

> ⚠️ **重启机器前，先 `dmesg -T > /var/log/xid_$(date +%F_%T).log`**。
> 这一条命令能保住你的报修依据。

---

## 7.9 故障树 F：输出质量异常

**服务"能返回"不等于"返回得对"。**

```
输出异常
├─ F1 重复、循环、停不下来
│   ├─ 采样参数问题 → 调 temperature / top_p / repetition_penalty
│   ├─ eos token 未正确识别 → 检查 tokenizer 与 generation_config
│   └─ 量化过度 → 用 FP16 复测，若正常则确认是量化导致
│
├─ F2 回答被截断
│   ├─ max_tokens 太小 → 调大
│   ├─ stop 参数误配 → 检查是否传了不该有的停止词
│   └─ 上下文超限被静默截断 → ⚠️ 最常见，见下
│
├─ F3 乱码 / 出现 <|im_start|> 之类的特殊 token
│   └─ chat template 不匹配
│      → 用与模型配套的 template（--chat-template）
│      → 检查是否有 tokenizer 版本混用
│
├─ F4 答非所问 / 漏读信息
│   └─ 极可能是输入被静默截断
│      → 对比：实际发送的 prompt token 数 vs --max-model-len
│      → 用 API 返回的 usage.prompt_tokens 核对
│      → 若超出，服务应报错而不是静默截断（回归到 06 章 F08 用例）
│
├─ F5 输出语言/格式不对
│   └─ 模型本身能力，或 chat template 问题
│      → 用官方推荐的调用方式复测
│
└─ F6 同一个问题每次答案差异巨大
    └─ temperature 高、top_p 大
       → 业务要求稳定时设 temperature=0（注意不能完全确定，
         但会显著稳定）
```

> ⚠️ **F4（静默截断）是生产事故的常见来源**。
> 用户上传了 50 页文档，服务只读了前 10 页，然后给出一个"看起来合理但错误"的答案。
> **这种问题比报错危险得多**，因为它不会被发现。
>
> **对策**：在文档里明确写清"最大输入长度"，并在服务端对超限请求
> **返回明确错误**，让调用方能感知。

---

## 7.10 应急预案

### 7.10.1 三级降级方案

| 级别 | 触发条件 | 动作 | 影响 |
| --- | --- | --- | --- |
| **L1 限流** | 队列持续积压 | 网关限流至可承受 QPS | 部分请求被拒 |
| **L2 缩配** | 显存吃紧/OOM 反复 | 降 `max_tokens`、降 `max-model-len`、限并发 | 回答变短，长文本不可用 |
| **L3 切换** | 主力模型不可用 | 切到备用小模型（如 7B） | 效果下降但服务可用 |

### 7.10.2 关键操作的标准流程

**紧急重启：**

```bash
# 1) 先记录当前状态（用于事后分析）
nvidia-smi > /var/log/emergency_$(date +%F_%T)_gpu.log
docker logs <container> > /var/log/emergency_$(date +%F_%T)_svc.log 2>&1

# 2) 重启
sudo systemctl restart vllm

# 3) 确认恢复（等模型加载完，可能要几分钟）
watch -n 5 'curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health'
```

**回滚（版本或参数）：**

```bash
# 参数回滚：用备份的启动参数覆盖
cp /opt/vllm/params.bak /opt/vllm/params.current
sudo systemctl restart vllm

# 镜像回滚
docker pull <registry>/vllm:<previous_tag>
# 修改启动脚本的镜像 tag 后重启
```

> ✅ **回滚能成功的前提是"有备份"**。
> 部署时必须把当时的启动参数、镜像 tag、模型版本存成一份
> `deploy-snapshot.txt`（见 [04 章 4.10](04-deployment.md)）。
> **没有快照就没有回滚。**

### 7.10.3 应急联系人卡（贴在机房/写进运维手册）

| 角色 | 姓名 | 电话 | 负责 |
| --- | --- | --- | --- |
| 交付方一线 | | | 服务层问题 |
| 交付方二线 | | | 模型/引擎问题 |
| 客户 IT | | | 网络/机房/权限 |
| 硬件厂商 | | | 掉卡/报修 |

---

## 7.11 日志采集与取证

### 7.11.1 该留什么

| 类型 | 内容 | 保留期 |
| --- | --- | --- |
| 服务日志 | vLLM 启动日志 + 错误日志 | ≥ 30 天 |
| 访问日志 | 请求量、延迟、状态码（**不含内容**） | ≥ 30 天 |
| 系统日志 | `dmesg` / `journalctl -k` | ≥ 30 天 |
| GPU 指标 | 时间序列（Prometheus） | ≥ 90 天 |
| 变更记录 | 谁在什么时候改了什么 | 永久 |

> ⚠️ **访问日志默认不要记录 prompt/回答内容**。
> 私有化交付的客户通常对数据敏感，
> 记录内容可能违反客户的数据安全要求。
> **要记内容必须事先获得客户书面同意，并做好脱敏与访问控制。**

### 7.11.2 日志轮转（必配）

```json
// /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "100m", "max-file": "5" }
}
```

```bash
# journald 限制
sudo sed -i 's/^#SystemMaxUse=.*/SystemMaxUse=2G/' /etc/systemd/journald.conf
sudo systemctl restart systemd-journald
```

> ⚠️ **不配日志轮转 = 埋一颗定时炸弹**。
> 真实的线上故障：某服务跑了三个月，vLLM 日志把根分区写满，
> 导致 docker 无法写入、服务全部崩溃。
> **这类故障完全可以通过一行配置避免。**

### 7.11.3 报障时必须提供的信息（模板）

```markdown
## 故障信息
- 故障时间：
- 现象描述：
- 影响范围：（全部请求 / 部分请求 / 仅特定接口）
- 是否可复现：（必现 / 偶发，频率）

## 环境信息
- 服务版本 / 镜像 tag：
- 模型与精度：
- 启动参数：（完整粘贴）
- 驱动版本：

## 已收集的证据
- [ ] nvidia-smi 完整输出
- [ ] dmesg | grep -i xid 输出
- [ ] 服务日志故障前后 200 行
- [ ] Prometheus 截图（故障时段）
- [ ] lspci -vv | grep -i nvidia 输出（怀疑硬件时）
- [ ] 一个最小的复现请求（curl 命令）

## 已尝试的处置
- 做了什么，结果如何
```

> ✅ **要求客户按这个模板提交报障**，能减少 70% 的来回沟通。
> 把它写进运维手册的附录。

---

**下一章**：[08 交付物与模板](08-templates.md) —— 所有要交的文档及其填写要点。
