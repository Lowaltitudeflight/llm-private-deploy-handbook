# LLM Private Deployment Delivery Handbook

> An end-to-end operations guide from environment survey to acceptance hand-over — written for delivery engineers, implementation teams, and anyone who has to write a private-deployment proposal.

[![Stars](https://img.shields.io/github/stars/Lowaltitudeflight/llm-private-deploy-handbook?style=flat&color=yellow)](https://github.com/Lowaltitudeflight/llm-private-deploy-handbook/stargazers)
[![Forks](https://img.shields.io/github/forks/Lowaltitudeflight/llm-private-deploy-handbook?style=flat&color=blue)](https://github.com/Lowaltitudeflight/llm-private-deploy-handbook/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/docs%20license-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs.md)
[![Chapters](https://img.shields.io/badge/chapters-8-brightgreen.svg)](docs/)
[![Templates](https://img.shields.io/badge/templates-3-orange.svg)](templates/)

**English** ｜ [简体中文](README.zh-CN.md)

| | |
| --- | --- |
| **Version** | v1.0 |
| **Last updated** | 2026-10-09 |
| **Engines covered** | vLLM / SGLang / TGI / TensorRT-LLM / lmdeploy / Ollama |
| **Scale covered** | single node & single GPU up to multi-node multi-GPU (8-GPU servers, dual-node 16-GPU) |
| **Format** | Markdown — publishable as a repository, exportable to Word |

> Note: the handbook body is currently written in Chinese. This README is the English entry point; each chapter is linked below so you can jump straight to the relevant file.

---

## What This Handbook Solves

The biggest difference between delivering a private deployment and calling an API is that **you are responsible for a system running inside someone else's data centre**. Whether the model loads, whether VRAM is sufficient, whether it survives concurrency, and who the customer calls when something breaks — none of these exist when you are calling an API, and all of them are yours to handle in a delivery.

Meanwhile, most material about large models online covers **how to run one** (`pip install` plus a single command). Very little covers **how to deliver one**: how to survey the environment, how to estimate VRAM, how to define acceptance criteria, which log to read when a card drops out, and which documents to hand over.

This handbook fills that gap.

---

## Choose Your Reading Path

| Your situation | Suggested path |
| --- | --- |
| **First solo delivery** | Read 01 → 07 in order, copying the forms in `templates/` into the project directory as you go |
| **You are on site and something looks wrong** | Jump straight to [02 Environment Survey](docs/02-environment-survey.md) and [07 Troubleshooting](docs/07-troubleshooting.md) |
| **You need to write a proposal or a quote** | [03 Solution Design](docs/03-solution-design.md) (VRAM estimation formulas and worked examples) + [06 Acceptance](docs/06-acceptance.md) |
| **The customer says "it's too slow"** | [05 Benchmarking & Tuning](docs/05-benchmark-tuning.md) — agree on the metric definitions before discussing optimisation |
| **You just want the forms** | Go straight to [`templates/`](templates/) |

---

## Map of the Handbook

```
                    ┌──────────────────────────────────────────┐
                    │  01 Overview · phase model · roles ·     │
                    │     deliverables                          │
                    └────────────────────┬─────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌──────────────────┐            ┌──────────────────┐
│ 02 Environment│─────────────▶│ 03 Solution      │───────────▶│ 04 Deployment    │
│    Survey     │   Go/No-Go   │    Design        │            │  driver / model  │
│ HW/SW/network │              │ sizing / choice /│            │  / parameters    │
│               │              │ parallelism      │            │                  │
└───────────────┘              └──────────────────┘            └────────┬─────────┘
                                                                        │
                                                              ┌─────────▼─────────┐
                                                              │ 05 Benchmarking   │
                                                              │    & Tuning       │
                                                              │ metrics / matrix  │
                                                              └─────────┬─────────┘
                                                                        │
                                                              ┌─────────▼─────────┐
                                                              │ 06 Acceptance     │
                                                              │ function / perf / │
                                                              │ stability         │
                                                              └─────────┬─────────┘
                                                                        │
                                                              ┌─────────▼─────────┐
                                                              │ 07 Operations &   │
                                                              │    Troubleshooting│
                                                              │ patrol / monitor /│
                                                              │ fault trees       │
                                                              └───────────────────┘
```

### Chapter Index

| Chapter | Contents | Key output |
| --- | --- | --- |
| [01 Delivery Overview](docs/01-overview.md) | 8-phase delivery model, role responsibility matrix, deliverable list, exit criteria | Project plan skeleton |
| [02 Environment Survey & Readiness Check](docs/02-environment-survey.md) | GPU/CPU/network/storage/software survey commands, version compatibility tables | Survey form + Go/No-Go decision |
| [03 Solution Design](docs/03-solution-design.md) | Model selection, **VRAM estimation formulas with worked examples**, quantisation, TP/PP decision tree, engine comparison | Deployment design document |
| [04 Deployment](docs/04-deployment.md) | Base environment, model integrity verification, **launch parameters explained one by one**, health checks | Deployment record |
| [05 Benchmarking & Performance Tuning](docs/05-benchmark-tuning.md) | **TTFT / TPOT definitions**, benchmark design, tuning matrix, curve interpretation, real case study | Benchmark report |
| [06 Acceptance](docs/06-acceptance.md) | Functional / performance / stability acceptance, how to write acceptance criteria, what to do when it fails | Acceptance test report + sign-off |
| [07 Operations & Troubleshooting](docs/07-troubleshooting.md) | Patrol routines, monitoring and alert thresholds, **6 fault trees**, contingency plans | Ops runbook + contingency plan |
| [08 Deliverables & Templates](docs/08-templates.md) | Purpose and completion guidance for every deliverable | Delivery document package |
| [Appendix A — Command Reference](appendix/a-commands.md) | Copy-pasteable commands grouped by scenario | — |
| [Appendix B — Checklists](appendix/b-checklists.md) | Four printable checklists: survey / deployment / go-live / patrol | — |
| [Appendix C — Glossary](appendix/c-glossary.md) | Bilingual term list with clarification of commonly confused items | — |

### Companion Article

- **"LLM Private Deployment: Three Fatal Traps in VRAM Estimation and Benchmarking"**
  → [docs/article-csdn-3-traps.md](docs/article-csdn-3-traps.md) ｜ [docs/article-juejin-3-traps.md](docs/article-juejin-3-traps.md)

  Extracts the VRAM estimation from [Chapter 03](docs/03-solution-design.md) and the benchmarking definitions from [Chapter 05](docs/05-benchmark-tuning.md) into a standalone walkthrough — useful for on-site reference or sharing externally. Covers the KV cache and GQA grouping-ratio trap, three concurrency worked examples, three TTFT definition traps, and how to read real-hardware benchmark curves (including the limits of what you may conclude).

### Companion Script

| Script | Purpose |
| --- | --- |
| [`scripts/estimate_kv.py`](scripts/estimate_kv.py) | Reads a model's `config.json` and computes per-token KV size, the VRAM breakdown, and **the maximum number of concurrent sequences**. No third-party dependencies; `--selftest` reproduces the three worked examples in Chapter 03. |

```bash
# Example: Qwen2.5-72B FP16 on 8 × A100 80G with 32K context
python scripts/estimate_kv.py --config <model_dir>/config.json \
    --params-b 72.7 --gpu-gb 80 --gpus 8 --overhead-gb 32 --ctx 32768
# → 320 KiB per token KV (computing via attention heads would be off by 8×);
#   KV cache pool 398.6 GB; ≈ 39 concurrent sequences
```

> Concurrency is always **rounded down**: when capacity planning, it is better to under-promise than to over-promise.

---

## Conventions

**Command format:** `$` denotes a normal user, `#` denotes root, `>>>` denotes a Python REPL.

**Callout blocks:**

- ✅ **Recommended** — safe to copy directly
- ⚠️ **Caution** — has preconditions or side effects; verify before proceeding
- ❌ **Anti-pattern** — someone has done this, and it went badly

**Numbers:** every performance figure in this handbook is annotated with its **source and runtime environment**. Figures marked `【measured】` come from reproducible run records; figures marked `【example】` are for demonstrating a calculation method only and **must not be quoted as your own delivery data**.

---

## Disclaimer & Scope

This handbook is a **summary of engineering practice**, not official vendor documentation. Parameter defaults and command syntax change between versions — **always defer to the official documentation for the version you are running**. The thresholds and rules of thumb given here come from public sources and personal practice; they are intended for quick estimation and fault localisation and **do not constitute a performance commitment**.

---

## Version History

| Version | Date | Change |
| --- | --- | --- |
| v1.0 | 2026-10-09 | First release: 8 chapters + 3 appendices + 3 templates |

---

## License

- **Documentation prose** (the narrative content of each `.md`): [CC BY 4.0](LICENSE-docs.md)
- **Code snippets** (shell / Python / YAML / Nginx config): [MIT](LICENSE)

Reposting and adaptation are welcome — just credit the source.

## Feedback

Parameter defaults and command syntax in this handbook change with upstream versions. If you find something that no longer matches your version, or you have a better approach, please open an issue. **The value of this handbook lies in being continuously corrected by reality.**
