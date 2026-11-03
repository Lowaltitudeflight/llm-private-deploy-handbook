#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显存估算：读模型 config.json，算出 KV Cache 大小与可支持并发路数。

用法：
    python estimate_kv.py --config /path/to/config.json --gpu-gib 80 --gpus 8 --ctx 32768
    python estimate_kv.py --config cfg.json --weight-dtype int4 --kv-dtype fp16 --params-b 72.7
    python estimate_kv.py --selftest

不依赖任何第三方库（不需要 transformers / torch），只做算术。
公式出处：docs/03-solution-design.md 1.2 / 1.3 节。

两个容易算错的地方，本脚本显式处理并打印推导过程：

1. **权重精度 ≠ KV 精度。** AWQ/GPTQ 是 W4A16：权重 0.55 B/param，KV 仍是
   FP16 的 2 B/element。早期版本用同一个 ``--dtype`` 同时决定两者，
   ``--dtype int4`` 会把 KV 算成 0.55 B，并发高估约 3.7 倍。
2. **不是所有模型都适用 GQA 公式。** DeepSeek 用 MLA（latent + RoPE），
   Mistral / Gemma 有滑窗；直接套 ``2×L×H_kv×D`` 会高估 25~60 倍。
   不认识的架构会直接报错，而不是静默套公式。
"""

import argparse
import json
import math
import sys

# 权重每参数字节数（含元数据余量）
WEIGHT_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.55}
# KV cache 只允许浮点精度：int4/int8 权重方案的 KV 仍是 fp16
KV_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0}

GIB = 1024 ** 3
# vLLM 的 block 大小；KV 池按 block 对齐，不能只算 ctx 个 token
DEFAULT_BLOCK = 16


# ----------------------------------------------------------------------
# 读 config
# ----------------------------------------------------------------------
def head_dim_of(cfg):
    """head_dim 的三种取法，优先级从高到低。"""
    if "head_dim" in cfg:
        return int(cfg["head_dim"])
    hidden = cfg.get("hidden_size") or cfg.get("n_embd")
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    if hidden and heads:
        return hidden // heads
    raise KeyError("config.json 里找不到 head_dim / hidden_size / num_attention_heads")


def kv_heads_of(cfg):
    """KV 头数：GQA/MQA 看 num_key_value_heads，MHA 时等于注意力头数。"""
    if "num_key_value_heads" in cfg:
        return int(cfg["num_key_value_heads"])
    return int(cfg.get("num_attention_heads") or cfg.get("n_head"))


def num_layers_of(cfg):
    for k in ("num_hidden_layers", "n_layer", "num_layers"):
        if k in cfg:
            return int(cfg[k])
    raise KeyError("config.json 里找不到层数字段")


def is_mla(cfg) -> bool:
    """DeepSeek 系的 MLA：KV 是压缩 latent + RoPE，不是每个头各存一份。"""
    return "kv_lora_rank" in cfg and (
        "qk_rope_head_dim" in cfg or "rope_head_dim" in cfg
    )


def layer_split(cfg):
    """返回 (full_attention_layers, sliding_layers, window)。

    - ``layer_types`` 存在时按类型统计（Gemma 2/3 这类交替结构）
    - 只有 ``sliding_window`` 时，全部层都按窗口截断（Mistral 7B）
    - 都没有时返回 (layers, 0, None)
    """
    layers = num_layers_of(cfg)
    window = cfg.get("sliding_window") or cfg.get("sliding_window_size")
    types = cfg.get("layer_types")
    if isinstance(types, list) and types:
        sliding = sum(1 for t in types if "sliding" in str(t).lower())
        full = layers - sliding
        return full, sliding, int(window) if window else None
    if window:
        return 0, layers, int(window)
    return layers, 0, None


# ----------------------------------------------------------------------
# KV 计算
# ----------------------------------------------------------------------
def analyze(cfg, kv_dtype="fp16", ctx=8192, block=DEFAULT_BLOCK):
    """返回几何信息与单序列 KV 占用（字节）。

    单序列占用按 block 对齐，并区分全注意力层与滑窗层。
    """
    if kv_dtype not in KV_BYTES:
        raise SystemExit(
            f"KV 精度只能是 {', '.join(sorted(KV_BYTES))}；"
            "int4/int8 权重方案的 KV 仍然是 fp16"
        )
    layers = num_layers_of(cfg)
    attn_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 0)
    nbytes = KV_BYTES[kv_dtype]
    aligned_ctx = math.ceil(ctx / block) * block

    if is_mla(cfg):
        rank = int(cfg["kv_lora_rank"])
        rope = int(cfg.get("qk_rope_head_dim") or cfg.get("rope_head_dim"))
        per_token = layers * (rank + rope) * nbytes
        mode = f"MLA（latent {rank} + RoPE {rope}）"
        per_seq = per_token * aligned_ctx
        kv_heads = None
        head_dim = None
    else:
        kv_heads = kv_heads_of(cfg)
        head_dim = head_dim_of(cfg)
        per_token = 2 * layers * kv_heads * head_dim * nbytes
        full, sliding, window = layer_split(cfg)
        if sliding and window:
            eff_tokens = min(aligned_ctx, window)
            per_seq = (
                2 * full * kv_heads * head_dim * nbytes * aligned_ctx
                + 2 * sliding * kv_heads * head_dim * nbytes * eff_tokens
            )
            mode = (
                f"GQA + 滑窗（{full} 层全注意力 @ {aligned_ctx} tokens，"
                f"{sliding} 层滑窗 @ {eff_tokens}）"
            )
        else:
            per_seq = per_token * aligned_ctx
            mode = "GQA / MHA"

    return {
        "layers": layers,
        "attn_heads": attn_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_dtype": kv_dtype,
        "kv_per_token": per_token,
        "kv_per_sequence": per_seq,
        "effective_ctx": aligned_ctx,
        "mode": mode,
        "gqa_ratio": (attn_heads / kv_heads) if kv_heads else 0,
        "kv_if_used_attn_heads": (
            2 * layers * attn_heads * head_dim * nbytes
            if kv_heads and head_dim
            else None
        ),
    }


def fmt_bytes(n):
    for unit, div in (("GiB", GIB), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if n >= div:
            return f"{n / div:.4g} {unit}"
    return f"{n:.0f} B"


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="按 config.json 估算 KV Cache 与可支持并发",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", help="模型 config.json 路径")
    ap.add_argument(
        "--weight-dtype", "--dtype", dest="weight_dtype", default="fp16",
        choices=sorted(WEIGHT_BYTES),
        help="权重精度（--dtype 是它的旧别名）",
    )
    ap.add_argument(
        "--kv-dtype", default="fp16", choices=sorted(KV_BYTES),
        help="KV 精度；与权重精度无关，AWQ/GPTQ 方案保持 fp16",
    )
    ap.add_argument("--params-b", type=float, help="真实参数量（B），如 72.7；不填则只算 KV")
    ap.add_argument(
        "--gpu-gib", "--gpu-gb", dest="gpu_gib", type=float, default=80.0,
        help="单卡显存 GiB（80GB 卡实际约 80 GiB）",
    )
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--util", type=float, default=0.90, help="gpu_memory_utilization")
    ap.add_argument(
        "--overhead-gib", "--overhead-gb", dest="overhead_gib", type=float,
        default=1.5,
        help="**每卡**激活/框架/通信开销 GiB，默认 1.5；手册建议 2~4",
    )
    ap.add_argument("--ctx", type=int, default=8192, help="单请求上下文长度（tokens）")
    ap.add_argument(
        "--block", type=int, default=DEFAULT_BLOCK, help="KV block 大小（vLLM 默认 16）"
    )
    ap.add_argument(
        "--safety", type=float, default=1.25,
        help="并发折减系数，默认 1.25（手册要求 1.2~1.3）；1.0 表示报理论值",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    # ---- 参数校验：宁可报错，也不要算出负数或除零 ----
    if not args.config:
        raise SystemExit("缺少 --config；只算 KV 时也要给模型 config.json")
    if args.ctx <= 0:
        raise SystemExit("--ctx 必须 > 0")
    if args.gpus <= 0:
        raise SystemExit("--gpus 必须 > 0")
    if not (0 < args.util < 1):
        raise SystemExit("--util 必须在 (0, 1) 之间")
    if args.gpu_gib <= 0:
        raise SystemExit("--gpu-gib 必须 > 0")
    if args.overhead_gib < 0:
        raise SystemExit("--overhead-gib 不能为负")
    if args.safety < 1:
        raise SystemExit("--safety 必须 >= 1（它是折减系数）")
    if args.params_b is not None and args.params_b <= 0:
        raise SystemExit("--params-b 必须 > 0")

    try:
        with open(args.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"读取 config 失败：{exc}")

    try:
        info = analyze(cfg, args.kv_dtype, args.ctx, args.block)
    except KeyError as exc:
        raise SystemExit(f"config 字段缺失：{exc}")

    print(f"架构            : {info['mode']}")
    print(f"层数            : {info['layers']}")
    if info["kv_heads"]:
        print(
            f"注意力头 / KV 头 : {info['attn_heads']} / {info['kv_heads']}"
            f"  （GQA 分组 ≈ {info['gqa_ratio']:.0f}:1）"
        )
        print(f"head_dim        : {info['head_dim']}")
    print(f"KV 精度         : {info['kv_dtype']}")
    print(f"单 token KV     : {fmt_bytes(info['kv_per_token'])}")
    if info["kv_if_used_attn_heads"] and info["gqa_ratio"] > 1.05:
        wrong = info["kv_if_used_attn_heads"]
        print(
            f"  ⚠️ 若误按注意力头数({info['attn_heads']})算 = {fmt_bytes(wrong)}"
            f"，是实际值的 {wrong / info['kv_per_token']:.1f} 倍"
        )
    print(
        f"单序列 {args.ctx} 上下文: {fmt_bytes(info['kv_per_sequence'])}"
        f"（按 block={args.block} 对齐到 {info['effective_ctx']} tokens）"
    )

    if args.params_b:
        weight = args.params_b * 1e9 * WEIGHT_BYTES[args.weight_dtype]
        total = args.gpu_gib * args.gpus * GIB
        usable = total * args.util
        overhead = args.overhead_gib * args.gpus * GIB
        pool = usable - weight - overhead

        print("\n--- 显存拆分（GiB，权重按十进制 GB 换算）---")
        print(
            f"权重            : {weight / GIB:.1f} GiB"
            f"  ({args.params_b}B × {WEIGHT_BYTES[args.weight_dtype]} B)"
        )
        print(
            f"可用显存        : {usable / GIB:.1f} GiB"
            f"  ({args.gpu_gib:g} × {args.gpus} × {args.util})"
        )
        print(
            f"减激活/框架开销  : {overhead / GIB:.1f} GiB"
            f"  ({args.overhead_gib:g} GiB/卡 × {args.gpus})"
        )
        print(f"KV Cache 池     : {pool / GIB:.1f} GiB")
        if pool <= 0:
            print("❌ KV Cache 池为负：权重就装不下，先降精度或加卡")
            return 1

        theoretical = pool / info["kv_per_sequence"]
        planned = int(theoretical / args.safety)
        print(f"单序列占用      : {fmt_bytes(info['kv_per_sequence'])}")
        print(f"理论并发        : {theoretical:.1f} 路")
        if args.safety > 1:
            print(
                f"≈ 可支持并发     : {planned} 路"
                f"  （理论 {theoretical:.1f} ÷ 安全系数 {args.safety:g}）"
            )
        else:
            print(f"≈ 可支持并发     : {planned} 路  （未折减）")
        if planned < 1:
            print("⚠️ 连 1 路都放不下：缩短上下文，或提高量化/加卡")
    return 0


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
def selftest():
    """复现手册 03 章 1.2/1.3 节的三组算例，并覆盖两个历史错误路径。"""
    cases = [
        # (名称, config, 期望单token, 期望理论并发, 期望折减后并发)
        (
            "Qwen2.5-7B  单卡 40GiB / 8K",
            {"num_hidden_layers": 28, "num_attention_heads": 28,
             "num_key_value_heads": 4, "head_dim": 128},
            56 * 1024, 46, 37,
        ),
        (
            "Qwen2.5-72B 8×80GiB / 32K",
            {"num_hidden_layers": 80, "num_attention_heads": 64,
             "num_key_value_heads": 8, "head_dim": 128},
            320 * 1024, 40, 32,
        ),
        (
            "Qwen2.5-72B AWQ 8×40GiB / 8K",
            {"num_hidden_layers": 80, "num_attention_heads": 64,
             "num_key_value_heads": 8, "head_dim": 128},
            320 * 1024, 90, 72,
        ),
    ]
    plan = [
        # (参数量B, 权重精度, 单卡GiB, 卡数, util, 每卡开销GiB, 上下文)
        (7.6, "fp16", 40, 1, 0.90, 1.5, 8192),
        (72.7, "fp16", 80, 8, 0.90, 4.0, 32768),
        (72.7, "int4", 40, 8, 0.90, 3.0, 8192),
    ]
    ok = True
    for (name, cfg, exp_kv, exp_theory, exp_plan), p in zip(cases, plan):
        params_b, wdtype, gpu_gib, gpus, util, overhead, ctx = p
        info = analyze(cfg, "fp16", ctx)
        weight = params_b * 1e9 * WEIGHT_BYTES[wdtype]
        pool = gpu_gib * gpus * GIB * util - weight - overhead * gpus * GIB
        theory = pool / info["kv_per_sequence"]
        planned = int(theory / 1.25)
        good = (
            info["kv_per_token"] == exp_kv
            and int(theory) == exp_theory
            and planned == exp_plan
        )
        ok &= good
        print(
            f"[{'OK ' if good else 'FAIL'}] {name}: 单token={fmt_bytes(info['kv_per_token'])}"
            f"(期望 {fmt_bytes(exp_kv)})  理论={theory:.1f}(期望 {exp_theory})"
            f"  折减后={planned}(期望 {exp_plan})"
        )

    # 1) W4A16：int4 权重不能把 KV 也降到 0.55 B
    cfg72 = {"num_hidden_layers": 80, "num_attention_heads": 64,
             "num_key_value_heads": 8, "head_dim": 128}
    kv_int4 = analyze(cfg72, "fp16", 8192)["kv_per_token"]
    good = kv_int4 == 320 * 1024
    ok &= good
    print(f"[{'OK ' if good else 'FAIL'}] W4A16 的 KV 仍按 fp16 算：{fmt_bytes(kv_int4)}")

    # 2) MLA：DeepSeek 不能套 GQA 公式
    mla = {
        "num_hidden_layers": 61, "num_attention_heads": 128,
        "hidden_size": 7168, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
    }
    info = analyze(mla, "fp16", 8192)
    good = info["kv_per_token"] == 61 * (512 + 64) * 2
    ok &= good
    print(
        f"[{'OK ' if good else 'FAIL'}] DeepSeek MLA：{fmt_bytes(info['kv_per_token'])}"
        f"（GQA 公式会算成 {fmt_bytes(2 * 61 * 128 * 56 * 2)}）"
    )

    # 3) 滑窗：Mistral 的超长上下文不应线性膨胀
    mistral = {"num_hidden_layers": 32, "num_attention_heads": 32,
               "num_key_value_heads": 8, "head_dim": 128, "sliding_window": 4096}
    s8k = analyze(mistral, "fp16", 8192)["kv_per_sequence"]
    s64k = analyze(mistral, "fp16", 65536)["kv_per_sequence"]
    good = s8k == s64k
    ok &= good
    print(f"[{'OK ' if good else 'FAIL'}] Mistral 滑窗：8K 与 64K 的 KV 相同（{fmt_bytes(s8k)}）")

    # 4) 非法 KV 精度必须报错，而不是静默用权重精度
    try:
        analyze(cfg72, "int4", 8192)
    except SystemExit:
        print("[OK ] --kv-dtype int4 被拒绝")
    else:
        ok = False
        print("[FAIL] --kv-dtype int4 竟然被接受了")

    print("\n自检", "全部通过" if ok else "存在不一致")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
