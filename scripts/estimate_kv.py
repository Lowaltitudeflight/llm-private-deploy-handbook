#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显存估算：读模型 config.json，算出 KV Cache 大小与可支持并发路数。

用法：
    python estimate_kv.py --config /path/to/config.json --gpu-gb 80 --gpus 8 --ctx 32768
    python estimate_kv.py --selftest

不依赖任何第三方库（不需要 transformers / torch），只做算术。
公式出处：docs/03-solution-design.md 1.2 / 1.3 节。
"""

import argparse
import json
import sys

BYTES_PER_PARAM = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.55}
GIB = 1024 ** 3


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


def analyze(cfg, dtype="fp16"):
    """返回单 token KV 字节数与可见的参数量信息。"""
    layers = num_layers_of(cfg)
    kv_heads = kv_heads_of(cfg)
    attn_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or kv_heads)
    head_dim = head_dim_of(cfg)
    nbytes = BYTES_PER_PARAM[dtype]
    # 单 token KV = 2(K 和 V) × 层数 × KV 头数 × head_dim × 每元素字节
    kv_per_token = 2 * layers * kv_heads * head_dim * nbytes
    return {
        "layers": layers,
        "attn_heads": attn_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "gqa_ratio": attn_heads / kv_heads if kv_heads else 0,
        "kv_per_token": kv_per_token,
        "kv_if_used_attn_heads": 2 * layers * attn_heads * head_dim * nbytes,
    }


def fmt_bytes(n):
    for unit, div in (("GiB", GIB), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if n >= div:
            return f"{n / div:.4g} {unit}"
    return f"{n} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="模型 config.json 路径")
    ap.add_argument("--dtype", default="fp16", choices=sorted(BYTES_PER_PARAM))
    ap.add_argument("--params-b", type=float, help="真实参数量（B），如 72.7；不填则只算 KV")
    ap.add_argument("--gpu-gb", type=float, default=80.0, help="单卡显存 GB")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--util", type=float, default=0.90, help="gpu_memory_utilization")
    ap.add_argument("--overhead-gb", type=float, default=None,
                    help="激活/框架/通信开销（GB，总量）；默认单卡 1.5 GB")
    ap.add_argument("--ctx", type=int, default=8192, help="单请求上下文长度（tokens）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    cfg = json.load(open(args.config, encoding="utf-8"))
    info = analyze(cfg, args.dtype)

    print(f"层数            : {info['layers']}")
    print(f"注意力头 / KV 头 : {info['attn_heads']} / {info['kv_heads']}"
          f"  （GQA 分组 ≈ {info['gqa_ratio']:.0f}:1）")
    print(f"head_dim        : {info['head_dim']}")
    print(f"单 token KV     : {fmt_bytes(info['kv_per_token'])}")
    if info["gqa_ratio"] > 1.05:
        wrong = info["kv_if_used_attn_heads"]
        print(f"  ⚠️ 若误按注意力头数({info['attn_heads']})算 = {fmt_bytes(wrong)}"
              f"，是实际值的 {wrong / info['kv_per_token']:.1f} 倍")
    print(f"单序列 {args.ctx} 上下文: {fmt_bytes(info['kv_per_token'] * args.ctx)}")

    if args.params_b:
        weight = args.params_b * BYTES_PER_PARAM[args.dtype]
        total = args.gpu_gb * args.gpus
        usable = total * args.util
        overhead = args.overhead_gb if args.overhead_gb is not None else 1.5 * args.gpus
        pool = usable - weight - overhead
        print("\n--- 显存拆分 ---")
        print(f"权重            : {weight:.1f} GB  ({args.params_b}B × {BYTES_PER_PARAM[args.dtype]})")
        print(f"可用显存        : {usable:.1f} GB  ({total:.0f} × {args.util})")
        print(f"减激活/框架开销  : {overhead:.1f} GB")
        print(f"KV Cache 池     : {pool:.1f} GB")
        if pool <= 0:
            print("❌ KV Cache 池为负：权重就装不下，先降精度或加卡")
            return 1
        per_seq = info["kv_per_token"] * args.ctx
        concurrency = pool * GIB / per_seq
        print(f"单序列占用      : {fmt_bytes(per_seq)}")
        # 容量规划一律向下取整：宁可报少，不能报多
        print(f"≈ 可支持并发     : {int(concurrency)} 路  (理论 {concurrency:.2f})")
        if concurrency < 1:
            print("⚠️ 连 1 路都放不下：缩短上下文，或提高量化/加卡")
    return 0


def selftest():
    """用 Qwen2.5 系列的公开 config 字段复现手册 03 章 1.2/1.3 节的数字。"""
    cases = [
        # (名称, layers, attn_heads, kv_heads, head_dim, 期望单token, 期望并发)
        ("Qwen2.5-7B  单卡 40G / 8K", 28, 28, 4, 128, 56 * 1024, 44),
        ("Qwen2.5-72B 8×80G / 32K", 80, 64, 8, 128, 320 * 1024, 39),
        ("Qwen2.5-72B AWQ 8×40G / 8K", 80, 64, 8, 128, 320 * 1024, 89),
    ]
    plan = [
        # (参数量B, 单卡GB, 卡数, util, 开销总量GB, 上下文)
        (7.6, 40, 1, 0.90, 1.5, 8192),
        (72.7, 80, 8, 0.90, 32.0, 32768),
        (72.7, 40, 8, 0.90, 24.0, 8192),
    ]
    ok = True
    for (name, layers, attn, kvh, hd, exp_kv, exp_cc), p in zip(cases, plan):
        cfg = {"num_hidden_layers": layers, "num_attention_heads": attn,
               "num_key_value_heads": kvh, "head_dim": hd}
        info = analyze(cfg)
        params_b, gpu_gb, gpus, util, overhead, ctx = p
        weight = params_b * (0.55 if "AWQ" in name else 2.0)
        pool = gpu_gb * gpus * util - weight - overhead
        cc = pool * GIB / (info["kv_per_token"] * ctx)
        good = info["kv_per_token"] == exp_kv and int(cc) == exp_cc
        ok &= good
        print(f"[{'OK ' if good else 'FAIL'}] {name}: 单token={fmt_bytes(info['kv_per_token'])}"
              f"(期望 {fmt_bytes(exp_kv)})  并发={cc:.1f}(期望 {exp_cc})")
    print("\n自检", "全部通过" if ok else "存在不一致")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
