# 附录 C · 术语表

> 中英对照。**带 ⚠️ 的是容易混淆或容易被误用的项**，建议重点看。

---

## C1 性能指标

| 缩写 | 全称 | 中文 | 说明 |
| --- | --- | --- | --- |
| **TTFT** | Time To First Token | 首 token 延迟 | ⚠️ 口径差异极大，见下 |
| **TPOT** | Time Per Output Token | 单 token 生成时间 | ⚠️ 分母是 `输出token数 - 1` |
| **ITL** | Inter-Token Latency | token 间延迟 | TPOT 的另一种算法（相邻 token 间隔的均值） |
| **E2E** | End-to-End Latency | 端到端延迟 | 从请求发出到最后一个 token |
| **QPS** | Queries Per Second | 每秒请求数 | ⚠️ 与"并发"不是一回事 |
| **TPS** | Tokens Per Second | 每秒 token 数 | ⚠️ 需注明是输入还是输出 |
| **Goodput** | — | 有效吞吐 | SLO 内完成的请求数 / 时间 |
| **P50 / P95 / P99** | Percentile | 百分位延迟 | ⚠️ 只报均值会掩盖尾延迟问题 |
| **SLO** | Service Level Objective | 服务等级目标 | 内部目标 |
| **SLA** | Service Level Agreement | 服务等级协议 | **合同承诺**，比 SLO 强 |
| **RPS** | Requests Per Second | 每秒请求数 | 同 QPS |

### ⚠️ TTFT 的三个必须说清的点

| 点 | 选项 | 影响 |
| --- | --- | --- |
| 起点 | 发送请求 / 收到 200 / 预填充开始 | 相差可达数倍 |
| 是否含排队 | 含 / 不含 | **含排队才是用户体验** |
| 是否含网络 | 含 / 不含 | 跨机房时差异显著 |
| **终点** | 任意首帧 / 首个 `content` 字段 | ⚠️ 推理模型下差异巨大 |

> **推荐口径**：从客户端发送请求开始，到收到**第一个 `content` 字段**为止，
> 包含排队时间与网络时间。**必须在报告里写明。**

---

## C2 并行与显存

| 缩写 | 全称 | 中文 | 关键点 |
| --- | --- | --- | --- |
| **TP** | Tensor Parallelism | 张量并行 | 切每层权重；通信量大；建议 ≤ 8 且机内 |
| **PP** | Pipeline Parallelism | 流水线并行 | 按层切；通信量小；有气泡；适合跨机 |
| **DP** | Data Parallelism | 数据并行 | 复制整个模型；提升吞吐 |
| **EP** | Expert Parallelism | 专家并行 | MoE 专用 |
| **SP** | Sequence Parallelism | 序列并行 | 长序列场景 |
| **KV Cache** | Key-Value Cache | KV 缓存 | ⚠️ 最容易被漏算的显存项 |
| **GQA** | Grouped Query Attention | 分组查询注意力 | ⚠️ KV 头数 ≠ 注意力头数，差一个量级 |
| **MHA / MQA / GQA** | Multi-Head / Multi-Query / Grouped-Query Attention | 多头 / 多查询 / 分组查询注意力 | KV 显存：MHA > GQA > MQA |
| **PagedAttention** | — | 分页注意力 | vLLM 的核心技术，减少显存碎片 |
| **Prefix Caching** | — | 前缀缓存 | 复用相同前缀的 KV，多轮对话收益大 |
| **Chunked Prefill** | — | 分块预填充 | 降低长请求对短请求的阻塞 |
| **Preemption** | — | 抢占 | ⚠️ 发生抢占 = 显存吃紧的早期信号 |
| **`num_key_value_heads`** | — | KV 头数 | ⚠️ 算 KV 显存必须用这个，不是 `num_attention_heads` |

---

## C3 量化

| 名称 | 含义 | 显存 | 精度 |
| --- | --- | --- | --- |
| **FP32** | 32 位浮点 | 1× | 最高（推理用不上） |
| **FP16 / half** | 16 位浮点 | 1/2 | 高 |
| **BF16** | 16 位脑浮点 | 1/2 | 高，数值范围更大 |
| **FP8** | 8 位浮点 | 1/4 | 较高，需新卡 |
| **INT8 / W8A8** | 8 位整数（权重+激活） | 1/4 | 中高 |
| **AWQ** | Activation-aware Weight Quantization | ~1/3.6 | ⚠️ **推荐**，激活保 FP16 |
| **GPTQ** | — | ~1/3.6 | 中高，生态较老 |
| **GGUF** | llama.cpp 的模型格式 | 视量化等级 | Q4_K_M 约 1/3.6 |
| **W4A16** | 权重 4bit，激活 16bit | ~1/3.6 | AWQ/GPTQ 都属于这类 |
| **反量化开销** | — | — | ⚠️ 量化后**速度不一定更快** |

> ⚠️ **"量化省显存" 不代表 "量化更快"**。
> 4bit 权重需要反量化才能计算，某些场景下比 FP16 还慢。
> 量化的主要收益是**显存**（→ 更大模型或更多并发）。

---

## C4 引擎与格式

| 名称 | 说明 |
| --- | --- |
| **vLLM** | 高吞吐推理引擎，PagedAttention，生产首选 |
| **SGLang** | RadixAttention，前缀缓存强，结构化输出 |
| **TGI** | HuggingFace 官方，Rust 实现，稳定 |
| **TensorRT-LLM** | NVIDIA 官方，极致性能，编译复杂 |
| **lmdeploy** | 国产（上海 AI Lab），TurboMind 引擎 |
| **MindIE** | 华为昇腾官方推理框架 |
| **ollama** | 极简部署，⚠️ **并发能力弱，不适合生产** |
| **llama.cpp** | CPU/GPU 混合推理，低资源 |
| **OpenAI 兼容 API** | 即 `/v1/chat/completions` 等接口规范，便于切换后端 |
| **SSE** | Server-Sent Events，流式响应的传输方式 |
| **safetensors** | HF 的安全权重格式（比 pickle 安全） |
| **分片（Shard）** | 大模型权重被切成多个文件，需全部齐备 |
| **chat template** | 对话模板，把消息列表拼成模型期望的 prompt 格式 |

---

## C5 硬件与系统

| 缩写 | 全称 | 中文 | 说明 |
| --- | --- | --- | --- |
| **SM** | Streaming Multiprocessor | 流多处理器 | GPU 计算单元 |
| **VRAM** | Video RAM | 显存 | — |
| **ECC** | Error Correcting Code | 纠错码 | ⚠️ **Uncorrectable > 0 = 该卡不可用** |
| **DBE / SBE** | Double / Single Bit Error | 双位/单位错误 | DBE 严重 |
| **Xid** | — | NVIDIA 错误码 | ⚠️ 掉卡相关，见 [07 章 7.8](../docs/07-troubleshooting.md) |
| **NVLink** | — | 卡间高速互联 | 决定 TP 是否可行 |
| **NVSwitch** | — | NVLink 交换芯片 | 全互联拓扑 |
| **PCIe** | — | 外设互联总线 | ⚠️ `current < max` = 降速 |
| **IB** | InfiniBand | — | 高性能网络，多机 TP 需要 |
| **RoCE** | RDMA over Converged Ethernet | — | 以太网上的 RDMA |
| **NUMA** | Non-Uniform Memory Access | 非统一内存访问 | ⚠️ GPU 与 CPU 跨节点会变慢 |
| **MTU** | Maximum Transmission Unit | 最大传输单元 | Jumbo Frame = 9000 |
| **cgroup** | Control Group | 控制组 | 容器资源限制 |
| **shm** | Shared Memory | 共享内存 | ⚠️ 容器默认 64MB，多卡必需放大 |
| **DKMS** | Dynamic Kernel Module Support | 动态内核模块 | 驱动随内核自动重建 |

---

## C6 交付与流程

| 缩写 | 全称 | 中文 |
| --- | --- | --- |
| **POC** | Proof of Concept | 概念验证 |
| **Go/No-Go** | — | 准入判定 |
| **P0/P1/P2** | Priority | 故障等级 |
| **RCA** | Root Cause Analysis | 根因分析 |
| **MTTR** | Mean Time To Recovery | 平均恢复时间 |
| **MTBF** | Mean Time Between Failures | 平均无故障时间 |
| **UAT** | User Acceptance Test | 用户验收测试 |
| **信创** | — | 信息技术应用创新（国产化） |
| **MOU / SOW** | Memorandum of Understanding / Statement of Work | 谅解备忘录 / 工作说明书 |

---

## C7 ⚠️ 最容易被混用的五组概念

### ① 并发 ≠ 在线用户数

| 概念 | 含义 |
| --- | --- |
| 在线用户数 | 打开页面的人数 |
| **并发请求数** | 同时在处理的请求数 |
| **排队请求数** | 已提交但还没开始处理的 |

经验换算：**并发 ≈ 在线用户数 × 20% ~ 50%**（视交互节奏而定）。
**验收时必须把这个换算讲清楚**，否则客户说"100 个用户"你会以为要 100 并发。

### ② TTFT ≠ 首帧时间

推理模型会先输出思考内容。**首个 SSE 帧不等于首个 `content`。**
用错的后果是 TTFT 数字完全不可比。

### ③ 吞吐 ≠ 速度

| 说法 | 含义 |
| --- | --- |
| "服务速度快" | 单请求延迟低（TTFT/TPOT 小） |
| "服务吞吐高" | 单位时间能产出很多 token |

两者**可能反向**：加大 batch 提升吞吐，但会拉高单请求延迟。

### ④ 量化 ≠ 加速

量化主要省**显存**。速度可能变快、也可能变慢。
"我们做了量化所以更快"是一个常见的不严谨表述。

### ⑤ GPU 利用率 ≠ 显存占用率

| 指标 | 含义 | 健康值 |
| --- | --- | --- |
| GPU 利用率 | 计算单元忙不忙 | 因场景而异，30%~80% 常见 |
| 显存占用率 | 显存用了多少 | ⚠️ 长稳时应**稳定**，不是越低越好 |
| KV 缓存使用率 | KV 池用了多少 | 20%~70% |

**"显存占用率高"不代表"卡在忙"** —— 大模型部署中显存本来就该被权重占满。
**不要把显存占用率当 GPU 利用率读。**

---

**返回**：[README](../README.md) · [附录 A 命令速查](a-commands.md) · [附录 B 检查清单](b-checklists.md)
