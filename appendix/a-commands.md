# 附录 A · 命令速查

> 按场景分组，可直接复制。`<...>` 为需要替换的占位符。
> 标 ⚠️ 的命令有副作用，执行前确认。

---

## A1 现场勘察

```bash
# ---- GPU ----
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,serial --format=csv
nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,\
pcie.link.width.max,pcie.link.width.current --format=csv
nvidia-smi topo -m                      # 卡间拓扑（关键）
nvidia-smi nvlink -s
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# ---- 健康 ----
nvidia-smi -q -d ECC | grep -iE "GPU|UNCORRECTABLE|Volatile"
nvidia-smi -q | grep -iE "Retired|Remapped|Xid"
nvidia-smi -q -d TEMPERATURE,POWER,PERFORMANCE | head -60
dcgmi diag -r 2                          # 硬件自检（需 DCGM）

# ---- CPU / 内存 ----
lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA|Thread"
numactl --hardware 2>/dev/null
free -h

# ---- 存储 ----
lsblk -d -o NAME,SIZE,ROTA,MODEL
df -h
fio --name=seqread --rw=read --bs=1M --size=4G \
    --filename=/data/fio.test --direct=1     # 顺序读带宽

# ---- 网络 ----
ip -br link show
for d in $(ls /sys/class/net | grep -v lo); do \
  echo "$d: $(cat /sys/class/net/$d/speed 2>/dev/null) Mb/s"; done
ibstat 2>/dev/null | head -20            # InfiniBand
ip -c a | grep -E "mtu|inet "
# 带宽实测：服务端 iperf3 -s ；客户端 iperf3 -c <ip> -t 30 -P 8

# ---- 软件 ----
cat /etc/os-release | head -3
uname -r
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u
nvcc --version
docker --version && docker info | grep -E "Runtime|Storage"
nvidia-container-cli -V
ldd --version | head -1

# ---- 出网 ----
curl -sI --max-time 8 https://pypi.org | head -1
curl -sI --max-time 8 https://huggingface.co | head -1
cat /etc/pip.conf ~/.pip/pip.conf 2>/dev/null
env | grep -i proxy
getent hosts pypi.org || echo "DNS 失败"
```

---

## A2 基础环境搭建

```bash
# ---- 屏蔽 nouveau（装驱动前）----
sudo tee /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
sudo dracut --force 2>/dev/null || sudo update-initramfs -u
# ⚠️ 需重启后确认：lsmod | grep nouveau

# ---- 容器运行时 ----
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# ✅ 验证容器可见 GPU
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# ---- Docker 日志与数据目录配置 ----
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "default-runtime": "nvidia",
  "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } },
  "data-root": "/data/docker",
  "log-driver": "json-file",
  "log-opts": { "max-size": "100m", "max-file": "5" }
}
EOF
sudo systemctl restart docker
```

---

## A3 模型获取与校验

```bash
# ---- ModelScope（国内推荐）----
pip install modelscope
modelscope download --model Qwen/Qwen2.5-72B-Instruct \
    --local_dir /data/models/Qwen2.5-72B-Instruct

# ---- HuggingFace CLI（走镜像）----
pip install -U "huggingface_hub[cli]"
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
  --local-dir /data/models/Qwen2.5-72B-Instruct --resume-download

# ---- 快速看模型关键配置（决定显存估算）----
python - <<'PY'
import json, sys, os
d = sys.argv[1] if len(sys.argv) > 1 else "."
cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
for k in ("model_type","num_hidden_layers","hidden_size","num_attention_heads",
          "num_key_value_heads","max_position_embeddings","torch_dtype",
          "vocab_size","quantization_config"):
    print(f"{k:26s} =", cfg.get(k, "(缺失)"))
# KV 单 token 字节数预估（FP16）
L  = cfg.get("num_hidden_layers")
KV = cfg.get("num_key_value_heads", cfg.get("num_attention_heads"))
HD = cfg.get("head_dim") or cfg.get("hidden_size",0)//max(cfg.get("num_attention_heads",1),1)
if L and KV and HD:
    b = 2 * L * KV * HD * 2
    print(f"\n单 token KV（FP16） = {b/1024:.0f} KiB")
    for ctx in (8192, 32768):
        print(f"  {ctx:>6} 上下文单序列 = {b*ctx/1024**3:.2f} GiB")
PY

# ---- 权重完整性校验（比对 index.json 声明大小）----
python - <<'PY'
import json, os
d = "."
shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
actual = sum(os.path.getsize(os.path.join(d,f)) for f in shards)
print(f"分片数={len(shards)}  实际总大小={actual/1024**3:.2f} GiB")
idx = "model.safetensors.index.json"
if os.path.exists(idx):
    m = json.load(open(idx, encoding="utf-8"))
    dec = m.get("metadata",{}).get("total_size")
    listed = set(m.get("weight_map",{}).values())
    print(f"index 声明={dec/1024**3:.2f} GiB  缺失分片={sorted(listed-set(shards)) or '无'}")
    print("✅ 一致" if dec and abs(actual-dec)/dec < 1e-6 else "❌ 不一致，分片可能不完整")
PY
```

---

## A4 服务启动

```bash
# ---- vLLM 容器（8 卡 · 72B AWQ）----
docker run -d --name vllm \
  --runtime nvidia --gpus '"device=0,1,2,3,4,5,6,7"' \
  --ipc=host --shm-size 16g \
  -v /data/models:/models:ro -p 8000:8000 \
  --restart unless-stopped \
  vllm/vllm-openai:v0.11.0 \
    --model /models/Qwen2.5-72B-Instruct-AWQ \
    --served-model-name qwen2.5-72b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --quantization awq --dtype bfloat16 \
    --disable-log-requests \
    --api-key "${API_KEY}"

# ---- 跟踪启动日志（模型加载要几分钟）----
docker logs -f vllm

# ---- 单卡 7B ----
vllm serve /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 32768 --max-num-seqs 16 \
  --enable-prefix-caching --kv-cache-dtype fp8 \
  --disable-log-requests
```

---

## A5 健康检查与冒烟测试

```bash
# L1 端口与健康
ss -tlnp | grep 8000
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" http://localhost:8000/health

# L2 模型列表
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# L3 非流式对话
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"qwen2.5-72b",
       "messages":[{"role":"user","content":"用一句话介绍你自己"}],
       "max_tokens":64,"stream":false}' | python3 -m json.tool

# L4 流式对话（-N 必须加，否则看不到流式效果）
curl -N -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"qwen2.5-72b",
       "messages":[{"role":"user","content":"从1数到10"}],
       "max_tokens":64,"stream":true}' | head -20

# 鉴权反例（应返回 401）
curl -s -o /dev/null -w "%{http_code}\n" \
  http://localhost:8000/v1/models -H "Authorization: Bearer wrong-key"
```

---

## A6 压测

```bash
# ---- vLLM 官方压测 ----
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8000 \
  --model qwen2.5-72b \
  --tokenizer /models/Qwen2.5-72B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 200 \
  --max-concurrency 16 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --seed 42

# ---- 并发梯度扫描（依次改 --max-concurrency）----
for c in 1 2 4 8 16 32; do
  echo "=== concurrency=$c ==="
  vllm bench serve --backend openai-chat --base-url http://localhost:8000 \
    --model qwen2.5-72b --endpoint /v1/chat/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts $((c*10)) --max-concurrency $c --request-rate inf --seed 42
done
```

---

## A7 监控与巡检

```bash
# ---- 一键巡检（建议存成 triage.sh）----
nvidia-smi
dmesg -T | grep -iE "nvrm|xid|nvidia" | tail -30
docker ps -a | head
docker logs --tail 100 <container> 2>&1 | tail -50
curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://localhost:8000/health

# ---- vLLM 关键指标 ----
curl -s http://localhost:8000/metrics | grep -E \
"^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|num_preemptions_total|prefix_cache_hit_rate)"

# ---- 实时显存/利用率 ----
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,\
power.draw --format=csv -l 5

# ---- 降频检查 ----
nvidia-smi -q -d PERFORMANCE | grep -A5 "Clocks Throttle Reasons"

# ---- ECC / 健康 ----
nvidia-smi -q -d ECC | grep -iE "UNCORRECTABLE|Volatile"
nvidia-smi -q | grep -iE "Retired|Remapped"

# ---- 掉卡定位 ----
dmesg -T | grep -i xid
sudo journalctl -k | grep -iE "xid|nvrm" | tail -50
lspci -vv | grep -iA3 "nvidia" | grep -i "LnkSta"

# ⚠️ 重启机器前务必先存证
dmesg -T > /var/log/xid_$(date +%F_%T).log
```

---

## A8 故障处置

```bash
# ---- 显存被谁占了 ----
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# 找到 pid 后确认归属再决定是否处理

# ⚠️ GPU 复位（需无进程占用，有风险）
nvidia-smi --gpu-reset -i <index>

# ---- 进程假死，看卡在哪 ----
py-spy dump --pid <pid>          # 需 pip install py-spy

# ---- 紧急重启（先存证！）----
nvidia-smi > /var/log/emergency_$(date +%F_%T)_gpu.log
docker logs <container> > /var/log/emergency_$(date +%F_%T)_svc.log 2>&1
sudo systemctl restart vllm
watch -n 5 'curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health'

# ---- 日志轮转 ----
sudo sed -i 's/^#SystemMaxUse=.*/SystemMaxUse=2G/' /etc/systemd/journald.conf
sudo systemctl restart systemd-journald
docker logs <container> --since 7d 2>&1 | \
  grep -icE "error|exception|traceback|out of memory"
```

---

## A9 磁盘与清理

```bash
# ⚠️ 以下命令会删除数据，执行前确认
docker system df                       # 看占用
docker image prune -a                  # 删未使用镜像
docker builder prune                   # 删构建缓存

# 清理旧日志（先看再删）
du -sh /data/docker/containers/* | sort -h | tail -5
```

---

## A10 常用端口

| 端口 | 用途 |
| ---: | --- |
| 8000 | vLLM 默认 API |
| 8080 | 常见备用 API 端口 |
| 9090 | Prometheus |
| 3000 | Grafana |
| 9400 | DCGM Exporter |
| 6379 | Ray head（多机部署） |
| 11434 | ollama |
| 443 / 80 | 网关 |

---

**返回**：[README](../README.md) · [附录 B 检查清单](b-checklists.md) · [附录 C 术语表](c-glossary.md)
