# 04 实施部署

> **目标**：把方案变成一台**正在提供服务**的机器，并且留下可复现的记录。
>
> 核心原则：**每一条执行的命令都要能写进部署记录表，并且能被别人重新执行一遍。**

---

## 4.1 实施前置检查

开工前的 5 分钟检查，能避免一半的返工：

```bash
# 1) 确认和勘察阶段是同一台机器（型号/数量/驱动没变）
nvidia-smi -L
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u

# 2) 确认没有残留的旧服务在跑
docker ps -a
ps aux | grep -E "vllm|python.*8000" | grep -v grep

# 3) 确认端口空闲
ss -tlnp | grep -E ":8000|:9090|:3000"

# 4) 确认剩余空间足够
df -h /data /var/lib/docker 2>/dev/null

# 5) 确认变更窗口（有没有人在用这台机器）
who
```

> ⚠️ **第 5 项容易被忽略**。生产机器上可能有别人的测试在现场跑，
> 你重启了 docker 就把人家的实验打断了。开工前在群里喊一声。

---

## 4.2 基础环境搭建

### 4.2.1 NVIDIA 驱动

**优先用系统包管理器**（可被包管理器统一管理、便于后续升级）：

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y nvidia-driver-550-server
sudo reboot

# RHEL / CentOS / openEuler / Kylin
sudo yum install -y nvidia-driver-550
sudo reboot
```

**如果需要指定版本**（如客户要求固定 535），用官方 `.run`：

```bash
# 先屏蔽 nouveau，否则装不上
sudo tee /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
sudo dracut --force 2>/dev/null || sudo update-initramfs -u
sudo reboot

# 重启后确认 nouveau 没加载
lsmod | grep nouveau || echo "nouveau 已屏蔽 ✓"

# 安装驱动（-s 静默，--no-opengl-files 避免覆盖图形库）
sudo sh NVIDIA-Linux-x86_64-550.xxx.run -s --no-opengl-files --dkms
```

**验证：**

```bash
nvidia-smi
# 应输出驱动版本、CUDA 版本、GPU 列表
```

> ⚠️ **驱动升级是有风险的操作**，会影响机器上所有用 GPU 的业务。
> 务必在变更窗口内做，并提前记录原版本号以便回滚。

### 4.2.2 容器运行时与 NVIDIA Container Toolkit

交付强烈建议走**容器方案**：环境隔离、可复现、易回滚。

```bash
# 安装 Docker（以 Ubuntu 为例）
sudo apt install -y docker.io
sudo systemctl enable --now docker

# 安装 NVIDIA Container Toolkit
# 注意：这一步在离线环境需要提前下载 deb/rpm 包
sudo apt install -y nvidia-container-toolkit

# 配置 Docker 使用 nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# ✅ 验证：容器里能看到 GPU
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

> ⚠️ **常见失败**：`docker run --gpus all` 报
> `could not select device driver "" with capabilities: [[gpu]]`
> → `nvidia-container-toolkit` 没装或没配置运行时，回到上面那两步。

### 4.2.3 配置国内镜像加速（可选，视出网策略）

```bash
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "runtimes": {
    "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] }
  },
  "default-runtime": "nvidia",
  "data-root": "/data/docker",
  "log-driver": "json-file",
  "log-opts": { "max-size": "100m", "max-file": "3" }
}
EOF
sudo systemctl restart docker
```

> ✅ **`data-root` 改到大盘**：模型镜像动辄 20~30 GB，
> 默认的 `/var/lib/docker` 很容易爆掉根分区。
>
> ✅ **`log-opts` 限制日志大小**：vLLM 日志很啰嗦，不限制的话
> 几周就能写满磁盘（这也是一个真实的线上故障原因）。

---

## 4.3 模型获取与完整性校验

### 4.3.1 下载

**方式一：ModelScope（国内推荐，稳定可达）**

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-72B-Instruct \
    --local_dir /data/models/Qwen2.5-72B-Instruct
```

**方式二：HuggingFace CLI（配镜像）**

```bash
pip install -U "huggingface_hub[cli]"
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
  --local-dir /data/models/Qwen2.5-72B-Instruct \
  --resume-download
```

**方式三：离线导入**（无外网环境）

在有网机器上下载后，打包带走：

```bash
# 有网机器
tar -cf qwen72b.tar -C /data/models Qwen2.5-72B-Instruct
# 或更省空间
tar -cJf qwen72b.tar.xz -C /data/models Qwen2.5-72B-Instruct   # 压缩率有限，耗时长

# 现场
tar -xf qwen72b.tar -C /data/models
```

> ⚠️ **别用 `--resume-download` 之外的任何"断点续传"自定义方案**，
> 半截文件是最难排查的问题来源。

### 4.3.2 完整性校验（必做）

下载完**一定要校验**，这是防止"加载报诡异错误"的第一道防线。

```python
# check_model.py —— 校验模型目录完整性
import json, os, sys

model_dir = sys.argv[1] if len(sys.argv) > 1 else "."
ok = True

# ---- 1) config.json 关键字段 ----
cfg_path = os.path.join(model_dir, "config.json")
if not os.path.exists(cfg_path):
    print("❌ 缺少 config.json")
    sys.exit(1)

cfg = json.load(open(cfg_path, encoding="utf-8"))
print("=== config.json ===")
for k in ("model_type", "num_hidden_layers", "hidden_size",
          "num_attention_heads", "num_key_value_heads",
          "max_position_embeddings", "torch_dtype",
          "vocab_size", "quantization_config"):
    if k in cfg:
        v = cfg[k]
        if k == "quantization_config":
            v = v.get("quant_method", v)
        print(f"  {k:28s} = {v}")
    else:
        print(f"  {k:28s} = (不存在)")

# 这些字段直接决定显存估算，缺了要警惕
if "num_key_value_heads" not in cfg:
    print("  ⚠️ 缺少 num_key_value_heads，KV Cache 估算需按 num_attention_heads 计算")

# ---- 2) safetensors 分片校验 ----
shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
print(f"\n=== 权重文件（{len(shards)} 个分片）===")

actual_total = 0
for f in shards:
    sz = os.path.getsize(os.path.join(model_dir, f))
    actual_total += sz

print(f"  实际总大小 = {actual_total/1024**3:.2f} GiB")

# 与 index.json 的元数据比对（这是最可靠的一致性检查）
idx_path = os.path.join(model_dir, "model.safetensors.index.json")
if os.path.exists(idx_path):
    idx = json.load(open(idx_path, encoding="utf-8"))
    declared = idx.get("metadata", {}).get("total_size")
    listed = set(idx.get("weight_map", {}).values())
    print(f"  index 声明 = {declared/1024**3:.2f} GiB")
    if declared:
        diff = abs(actual_total - declared) / declared
        if diff < 1e-6:
            print("  ✅ 大小完全一致")
        else:
            print(f"  ❌ 大小不一致，偏差 {diff:.4%} —— 分片可能不完整！")
            ok = False
    missing = listed - set(shards)
    if missing:
        print(f"  ❌ index 中列出但目录缺失的分片: {sorted(missing)}")
        ok = False
    else:
        print(f"  ✅ index 列出的 {len(listed)} 个分片全部存在")
else:
    print("  (无 index.json，单文件模型)")

# ---- 3) 必需文件 ----
print("\n=== 必需文件 ===")
for f in ("config.json", "tokenizer_config.json"):
    exists = os.path.exists(os.path.join(model_dir, f))
    print(f"  {'✅' if exists else '❌'} {f}")
    ok = ok and exists

# generation_config.json / chat template 建议有
for f in ("generation_config.json", "tokenizer.json"):
    exists = os.path.exists(os.path.join(model_dir, f))
    print(f"  {'✅' if exists else '⚠️ '} {f}")

print("\n" + ("✅ 校验通过" if ok else "❌ 校验未通过，不要启动服务"))
sys.exit(0 if ok else 1)
```

```bash
python check_model.py /data/models/Qwen2.5-72B-Instruct
```

> ✅ **为什么用 `index.json` 的 `metadata.total_size` 比对**：
> 这是分片下载工具自己写下的"应该有多大"。
> 实际大小完全一致，说明**每一个分片都完整**；
> 有偏差就说明某个分片截断了，这时候 `safetensors` 加载会报
> `Error while deserializing header` 之类的诡异错误。

> ⚠️ **量化模型的额外检查**：AWQ/GPTQ 模型目录里应有 `quant_config` 相关的
> 元信息，且 **不能有残留的 FP16 分片**。两者混在目录里会导致加载混乱。
> 另外，量化模型**必须用与量化时相同的 tokenizer**，跨版本混用会导致输出乱码。

---

## 4.4 部署方式

### 4.4.1 容器部署（推荐）

```bash
docker run -d --name vllm-qwen72b \
  --runtime nvidia --gpus '"device=0,1,2,3,4,5,6,7"' \
  --shm-size 16g \
  -v /data/models:/models:ro \
  -p 8000:8000 \
  --ipc=host \
  --restart unless-stopped \
  vllm/vllm-openai:v0.11.0 \
    --model /models/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name qwen2.5-72b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --quantization awq \
    --disable-log-requests \
    --api-key "${API_KEY}"
```

> ⚠️ **三个容易漏的容器参数**：
> - `--shm-size 16g`（或 `--ipc=host`）：默认 64 MB 的共享内存会导致
>   多卡通信失败或直接崩溃。**这是最常见的容器部署翻车原因。**
> - `--gpus '"device=0,1"'`：容器内 GPU 编号会**重新从 0 开始**。
>   宿主机 `device=4,5` → 容器内看到的是 `0,1`。
> - `--restart unless-stopped`：不加的话机器重启后服务不会自启。

### 4.4.2 多机部署

```bash
# 节点 0（主节点）
vllm serve /models/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --host 0.0.0.0 --port 8000

# 节点 1（工作节点）
# 先起 ray 集群
ray start --address='<主节点IP>:6379'
# 主节点上用 --pipeline-parallel-size 2 即会自动跨机切分
```

> ⚠️ 多机部署要求**各节点模型路径完全一致**（同一绝对路径）。
> 建议用共享存储挂载，或提前 rsync 到各节点本地盘的相同路径。

### 4.4.3 裸机部署（无容器运行时时的退路）

```bash
# 用独立 venv，避免污染系统 Python
python3 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate

# 用官方推荐的 pip 源（如需）
pip install vllm==0.11.0

vllm serve /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

> ⚠️ 裸机部署的依赖冲突远比容器常见
> （torch / CUDA / triton / xformers 版本互相拉扯）。
> **能用容器就用容器。**

---

## 4.5 启动参数逐项详解

这是本章最实用的部分。下面按"影响什么"分组。

### 4.5.1 并行与显存类（决定"能不能跑起来"）

| 参数 | 作用 | 怎么定 | 常见错误 |
| --- | --- | --- | --- |
| `--tensor-parallel-size` | 张量并行度数 | 见 [03 章 3.5](03-solution-design.md) 决策树 | 设成不能整除注意力头数的值 → 启动报错 |
| `--pipeline-parallel-size` | 流水线并行段数 | 跨机时用，等于机器数 | 单机内用 PP 反而增加延迟 |
| `--gpu-memory-utilization` | 显存使用比例 | **0.90**（默认）；与别的业务共卡时降到 0.6~0.7 | 设 1.0 → 必然 OOM（没给激活留空间） |
| `--max-model-len` | 最大上下文长度 | 按业务需要，**越小 KV 池越大** | 直接沿用模型上限（如 128K）→ 并发能力被吃光 |
| `--max-num-seqs` | 最大并发序列数 | 由 KV 池算出（见 3.3.4） | 设太大 → 显存不够；太小 → 请求排队 |
| `--max-num-batched-tokens` | 单批最大 token 数 | 2440~8192，调大提升吞吐但增延迟 | 调得过大 → 长 prompt 独占一个 batch |
| `--swap-space` | CPU 显存交换空间(GiB) | 默认 4；显存紧张可加大 | 设太大且真用上 → 性能暴跌 |
| `--cpu-offload-gb` | 卸载到 CPU 的显存 | 显存实在不够时的最后手段 | **会严重拖慢**，只用于"能跑就行" |

### 4.5.2 性能开关类（决定"跑得快不快"）

| 参数 | 作用 | 收益 | 代价 |
| --- | --- | --- | --- |
| `--enable-prefix-caching` | 复用相同前缀的 KV | **多轮对话/固定 system prompt 场景收益巨大**（命中率高时 TTFT 降数倍） | 显存多占一点；命中率低时几乎无收益 |
| `--enable-chunked-prefill` | 把长 prompt 预填充切块 | **显著降低 TTFT 抖动**，避免长请求阻塞短请求 | 略增调度开销 |
| `--kv-cache-dtype fp8` | KV Cache 用 FP8 | **KV 显存减半 → 并发能力翻倍** | 需较新 GPU；轻微精度影响 |
| `--quantization` | 权重反量化 | 显存大降 | 见 [03 章 3.4](03-solution-design.md) |
| `--dtype bfloat16` | 计算精度 | 与 FP16 等价，数值更稳 | 老卡（V100）不支持 bf16 |
| `--enforce-eager` | 关闭 CUDA Graph | 启动快、调试方便 | **吞吐下降**，生产不要开 |
| `--block-size` | PagedAttention 块大小 | 16 通用 | 调整收益有限，不建议乱改 |
| `--scheduling-policy` | 调度策略 | `fcfs` 默认；`priority` 可做优先级 | 需客户端配合传优先级字段 |

> ✅ **收益最高的两个开关，优先打开**：
> 1. **`--enable-prefix-caching`** —— 多轮对话场景能省掉重复的预填充
> 2. **`--enable-chunked-prefill`** —— 长 prompt 不再阻塞短请求，P99 明显改善
>
> ⚠️ **注意版本差异**：在 vLLM **V1 引擎**（约 0.8+）中，
> prefix caching 默认**开启**、chunked prefill 也默认**开启**；
> 在 V0 引擎中两者默认**关闭**。
> 所以在 V1 上你看到的"没配也有这些行为"是正常的，不是 bug。

### 4.5.3 服务与安全类（决定"能不能交付"）

| 参数 | 作用 | 说明 |
| --- | --- | --- |
| `--host` / `--port` | 监听地址端口 | 生产建议 `0.0.0.0` + 网关转发 |
| `--served-model-name` | 对外暴露的模型名 | **建议设置**，避免暴露真实路径 |
| `--api-key` | 简单鉴权 | 至少设这个；更强的鉴权放网关做 |
| `--chat-template` | 对话模板 | 模型自带模板有问题时手动指定 |
| `--disable-log-requests` | 关闭逐请求日志 | **生产必须开**，否则日志爆炸且泄露内容 |
| `--max-log-len` | 日志截断长度 | 防止超长 prompt 刷屏 |
| `--trust-remote-code` | 允许执行模型目录代码 | ⚠️ **安全风险**，只对可信来源开启 |
| `--seed` | 随机种子 | 需要可复现的验收测试时设置 |

> ⚠️ **`--trust-remote-code` 是安全红线**。
> 它会执行模型目录里的 Python 代码。**只对你自己下载并校验过的模型开启**，
> 从不明来源拿模型时绝不开。客户的安全审计通常会查这一项。

### 4.5.4 参数配置的推荐起点

```bash
# 8 卡整机 · 72B AWQ · 8K 上下文 · 面向并发
vllm serve /models/Qwen2.5-72B-Instruct-AWQ \
  --served-model-name qwen2.5-72b \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --quantization awq \
  --dtype bfloat16 \
  --disable-log-requests \
  --api-key "${API_KEY}"
```

```bash
# 单卡 · 7B FP16 · 长上下文
vllm serve /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --disable-log-requests
```

---

## 4.6 其他引擎的参数对照

| 能力 | vLLM | SGLang | TGI | ollama |
| --- | --- | --- | --- | --- |
| 张量并行 | `--tensor-parallel-size` | `--tp-size` | `--num-shard` | 不支持 |
| 显存比例 | `--gpu-memory-utilization` | `--mem-fraction-static` | — | `OLLAMA_*` 环境变量 |
| 最大并发 | `--max-num-seqs` | `--max-running-requests` | `--max-concurrent-requests` | `OLLAMA_NUM_PARALLEL` |
| 最大上下文 | `--max-model-len` | `--context-length` | `--max-input-length` + `--max-total-tokens` | `num_ctx`（请求内传） |
| 前缀缓存 | `--enable-prefix-caching` | 默认开启（RadixAttention） | — | 有限支持 |
| 量化 | `--quantization awq/gptq/fp8` | `--quantization` | `--quantize` | 模型自带 GGUF 量化 |

---

## 4.7 健康检查与冒烟测试

### 4.7.1 三级健康检查

```bash
# L1 进程与端口
ss -tlnp | grep 8000
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health

# L2 模型元信息（能列出模型 = 加载完成）
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# L3 真实推理（最关键的一步）
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "qwen2.5-72b",
    "messages": [{"role":"user","content":"用一句话介绍你自己"}],
    "max_tokens": 64,
    "stream": false
  }' | python3 -m json.tool
```

### 4.7.2 冒烟测试清单（写进部署记录表）

| # | 测试项 | 预期 | 结果 |
| :---: | --- | --- | :---: |
| 1 | `/health` 返回 200 | 200 | ⬜ |
| 2 | `/v1/models` 列出目标模型 | 含 `served-model-name` | ⬜ |
| 3 | 非流式对话返回正常 | 有 `choices[0].message.content` | ⬜ |
| 4 | **流式**对话逐 token 返回 | 多个 `data:` 分片 + `[DONE]` | ⬜ |
| 5 | 无 API Key 请求被拒 | 401 | ⬜ |
| 6 | 错误 API Key 被拒 | 401 | ⬜ |
| 7 | 超长输入被正确截断/报错 | 400 且信息清晰 | ⬜ |
| 8 | 中文输出正常（无乱码） | 正常中文 | ⬜ |
| 9 | 多轮对话上下文生效 | 记得上一轮内容 | ⬜ |
| 10 | 并发 2 路同时请求成功 | 都返回结果 | ⬜ |
| 11 | 重启服务后恢复正常 | 恢复 | ⬜ |
| 12 | 空 messages 请求不崩 | 400 | ⬜ |

**流式接口测试**（必须单独测，很多问题只在流式下暴露）：

```bash
curl -N -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "qwen2.5-72b",
    "messages": [{"role":"user","content":"从1数到10"}],
    "max_tokens": 64,
    "stream": true
  }' | head -20
```

> ⚠️ **`curl` 不加 `-N` 会缓冲整个响应**，看起来像"没有流式"。
> 这是排查"流式不生效"时最常见的误判。

> ⚠️ **`grep` 排除文本型 content 时的小坑**：
> 推理模型（如 DeepSeek-R1 系列）会先输出大量思考内容，
> 首个 **content** 分片可能出现在很靠后的位置。
> 如果用它测 TTFT，会把思考时间也算进去（见 [05 章 5.1](05-benchmark-tuning.md)）。

---

## 4.8 反向代理与对外暴露

```nginx
# /etc/nginx/conf.d/llm.conf
upstream llm_backend {
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name llm.example.internal;

    ssl_certificate     /etc/nginx/ssl/llm.crt;
    ssl_certificate_key /etc/nginx/ssl/llm.key;

    # ⚠️ 大模型响应很慢，超时必须放大
    proxy_read_timeout    300s;
    proxy_send_timeout    300s;
    proxy_connect_timeout 10s;

    # ⚠️ 流式响应必须关闭缓冲
    proxy_buffering off;
    proxy_cache off;

    location /v1/ {
        # 限流：每 IP 每秒 10 个请求，突发 20
        limit_req zone=llm_limit burst=20 nodelay;

        proxy_pass http://llm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 需要
        proxy_set_header X-Accel-Buffering no;
    }
}
```

```nginx
# 在 http 块中定义限流区
limit_req_zone $binary_remote_addr zone=llm_limit:10m rate=10r/s;
```

> ⚠️ **两个必设项**：
> 1. `proxy_buffering off` —— 不关的话流式响应会被 Nginx 攒成一个包再发，
>    用户体验上"流式"变成"等 30 秒一次性出结果"。
> 2. `proxy_read_timeout 300s` —— 默认 60s，长回答会被中途掐断。

---

## 4.9 进程守护与开机自启

```ini
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM Inference Service
After=docker.service network-online.target
Requires=docker.service

[Service]
Restart=always
RestartSec=15
# ⚠️ 大模型加载慢，给足启动时间
TimeoutStartSec=900
ExecStartPre=-/usr/bin/docker rm -f vllm-qwen72b
ExecStart=/usr/bin/docker run --rm --name vllm-qwen72b \
  --runtime nvidia --gpus all --ipc=host --shm-size 16g \
  -v /data/models:/models:ro -p 8000:8000 \
  vllm/vllm-openai:v0.11.0 \
  --model /models/Qwen2.5-72B-Instruct-AWQ \
  --tensor-parallel-size 8 --max-model-len 8192 \
  --disable-log-requests
ExecStop=/usr/bin/docker stop vllm-qwen72b

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm
sudo systemctl status vllm
```

> ⚠️ **`TimeoutStartSec` 一定要给足**（≥ 模型加载时间）。
> 默认 90 秒，72B 模型加载可能要 3~5 分钟，
> systemd 会在启动过程中判定失败并杀掉进程，**形成"永远起不来"的死循环**。

---

## 4.10 部署记录表

部署过程中**边做边记**，不要事后回忆。完整模板见
[`templates/deployment-record.md`](../templates/deployment-record.md)。

必记内容：

- 执行时间、执行人
- 每条关键命令（**原样记录**，含参数）
- 每步的结果与异常
- 服务最终参数快照
- 遇到的问题与解决方式
- 变更前后的环境差异

> ✅ **一条实用建议**：部署完成后，把"服务最终参数"原样存成一份
> `deploy-snapshot.txt` 放在交付目录里。
> 半年后要排查问题时，这份快照就是"当时到底是什么样"的唯一答案。

---

**下一章**：[05 压测与性能调优](05-benchmark-tuning.md) —— 服务起来了，怎么证明它够快。
