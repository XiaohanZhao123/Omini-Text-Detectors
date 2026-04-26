# Cloud-agent roadmap — HAT-Bench + ablation experiments still pending

Authored 2026-04-26 from branch `damasha-sentence-annotation` (commit at branch tip).
Hand off this branch to a Cloud Code agent with multi-GPU access. Everything
below is paper-faithful by default; **do not** loosen any wrapper config or
swap to `-finetuned` variants without an explicit user OK.

## 0. Setup on the cloud host (once)

```bash
# UV env (CLAUDE.md storage discipline):
export UV_CACHE_DIR=/path/to/external/uv-cache
export UV_LINK_MODE=copy
uv sync

# .env must contain (NOT committed):
#   OPENAI_API_KEY=...
#   GEMINI_API_KEY=...
#   ANTHROPIC_API_KEY=...
#   HF_TOKEN=...        # for HuggingFace push
```

`pyproject.toml` already pins `setuptools<81` (fastNLP needs `pkg_resources`),
`anthropic>=0.97`, `google-genai`, `python-dotenv`. The seqxgpt local checkpoint
must be re-downloaded or fetched from `zcahjl3/seqxgpt-detector` (HF Hub).

## 1. HAT-Bench primary results — `default_setting/`

Goal: every detector × 15 cells (4 domains × 4 generators − reports/gpt-5.4)
under `HAT-Baselines/baseline_results/default_setting/<level>/<detector>/`.

### 1a. Detectors with all 15 cells local but only 11 on HF (push Qwen cells)

These ran during the original sweep but pre-Qwen. Local results exist at
`results/new_data_eval/doc_level/default_setting/<detector>_new4d_<dom>_qwen3-8b_*/`.
Need: re-validate the 4 Qwen cells (essays/abstracts/news/reports), then
upload to `default_setting/doc/<detector>/<dom>_qwen3-8b/`.

| Detector | HF cells | Local cells | Action |
|---|---|---|---|
| desklib | 11 | 15 | push 4 Qwen cells |
| detectllm | 11 | 15 | push 4 Qwen cells (verify orientation/score scale first — see §5) |
| e5-small | 11 | 15 | push 4 Qwen cells |
| fast-detectgpt | 11 | 15 | push 4 Qwen cells (verify score-orientation flip — see §5) |
| ood-llm-detect | 11 | 15 | push 4 Qwen cells |
| radar | 11 | 15 | push 4 Qwen cells |
| roberta-openai | 11 | 15 | push 4 Qwen cells |

### 1b. seqxgpt — paper-faithful, 0 of 15 on HF

Local checkpoint at `baseline/seqxgpt/data/seqxgpt_transformer.pt` (or
`zcahjl3/seqxgpt-detector` on HF Hub). Wrapper requires 4 LMs simultaneously
loaded: gpt2-xl (fp32, ~6 GB) + gpt-neo-2.7b/gpt-j-6b/llama-7b (8-bit).
**Total ~22 GB** — fits a single 24 GB GPU but tight; better to split
across ≥2 GPUs.

Launcher: `evaluate/aes/scripts/queue_seqxgpt_default_setting.sh` (waits
until `cuda:0 ≥14 GB free + partner_big ≥10 GB + partner_small ≥6 GB`,
then rewrites `feature_devices` in `omini_text/configs/seqxgpt.yaml` and
runs `evaluate/aes/run_detector_per_row.py --detector seqxgpt`).

Push target: `default_setting/sentence/seqxgpt/<dom>_<gen>/`.

### 1c. adaloc — 11/11 on HF? Verify

Memory said "adaloc 10/11 cells on default_setting" but local count shows 15.
Reconcile by listing HF folders under `default_setting/sentence/adaloc/`
and filling any (domain, generator) gap.

## 2. Ablations — LLM judges (almost done; finish + push)

Local data: `results/new_data_eval/sentence/per_row/llm_judge/<judge>/ablations/`

| Judge | A1 (12) | A2 (4) | A3 (4) | Status |
|---|---|---|---|---|
| GPT-5.4 (`gpt54-sent-conf-none`) | 12 | 4 | 4 | ✓ done |
| Gemini-2.5-Flash (`gemini-flash-sent-conf-minimal`) | 12 | 4 | 4 | ✓ done |
| Claude-Haiku-4.5 (`claude-haiku-sent-conf-minimal`) | 12 | 4 | 4 | ✓ done |
| Gemma-3n-E4B-it (`gemma-4-E4B-it`) | 12 | 2 | 0 | running on cuda:0; ~5 cells left |

After Gemma finishes:
- Push 80 cells to `HAT-Baselines/baseline_results/llm_judge/sentence/<judge>/ablations/<cell>/`
- Re-run `evaluate/aes/scripts/plot_judge_ablations.py` for the final 12 PNG + 12 MD.

## 3. Ablations — supervised + tuned detectors (NOT YET STARTED — need user OK)

Two big sweeps that mirror §2 but with the detector zoo instead of judges:

### 3a. default_setting × ablations (10 detectors × 20 CSVs = 200 cells)
Detectors: adaloc, damasha, desklib, detectllm, e5-small, fast-detectgpt,
gigacheck, ood-llm-detect, radar, roberta-openai.

Driver: `evaluate/aes/run_detector_per_row.py` already supports
`--detector <X>`. It uses the manifest, so it needs the same `--csvs` /
`--cell-name-template` flags that we added to `run_gpt_per_row.py`. **Code
change required**: port those two flags from `run_gpt_per_row.py:main()` to
`run_detector_per_row.py:main()`. Then a wrapper script analogous to
`run_judges_on_ablations.sh` can iterate detectors × CSVs.

Output target:
`results/new_data_eval/<level>/per_row/default_setting/<detector>/ablations/<cell>_<ts>/`

### 3b. tuned_on_new_data × ablations (11 detectors × 20 CSVs = 220 cells)
Detectors: adaloc, damasha, fast-detectgpt, genai-sentence, genai-sentence-v2,
gigacheck, gl-clic, gl-clic-simplified, gl-clic-v2, sendetex, seqxgpt.

Same driver edit as §3a. Each detector needs its `-finetuned.yaml` config
and a downloaded HF checkpoint. **Note**: 6 of these (gl-clic*, genai-sentence*,
sendetex) currently have no paper-faithful baseline — only the
fine-tuned variant. So strictly speaking only the tuned variant runs here.

## 4. Tuned-only methods missing a paper-faithful baseline (§5 Pareto blocker)

The following live in `tuned_on_new_data/` but never had a paper-recipe row
in `default_setting/`:

| Detector | Why missing | Effort to add |
|---|---|---|
| gl-clic | paper-incomplete (4 auxiliary heads not implemented in wrapper) | extend wrapper or document as "tuned-only" |
| gl-clic-simplified | tuned-only by design | n/a |
| gl-clic-v2 | tuned-only by design | n/a |
| genai-sentence | needs DeBERTa-v3-base + paper data + reproduction script | half-day reproduction |
| genai-sentence-v2 | tuned-only by design | n/a |
| sendetex | requires paid GPT-4o; out of scope per CLAUDE.md | skip |

## 5. Bug investigations (do **not** push HF until resolved)

- **fast-detectgpt** — score orientation may be flipped (Task #28).
  Symptom: AUROC < 0.5 on some cells. Inspect
  `results/new_data_eval/doc_level/default_setting/fast-detectgpt_new4d_*/predictions.jsonl`
  → look at `score` distribution per `version`; v0 (pure human) should
  get LOWER score than v8 if score = P(AI).
- **damasha** — always-AI label collapse on HAT-Bench (Task #29).
  Symptom: every prediction = AI regardless of input. Reproduce on
  `damasha_new4d_essays_qwen3-8b_*` → likely a calibration constant
  baked into the wrapper from the original paper's small training pool.
- **detectllm** — broken on HAT-Bench per memory; likely the same
  fp16/fp32 LRR scaling issue we fixed for the reproduction script but
  never back-ported to the per-row driver.

Each fix is local to its detector wrapper. Re-run the affected cells
on HAT-Bench + push.

## 6. Push to HF — uniform layout

For every cell folder, upload to:
```
HAT-Baselines/baseline_results/<category>/<level>/<detector>/<cell-id>/
  predictions.jsonl
  provenance.json
  run_config.json
  summary.json
```

`<category>` ∈ {`default_setting`, `tuned_on_new_data`, `llm_judge`}.
`<level>` ∈ {`doc`, `sentence`, `span`, `token`} (matches the detector's
output type).
`<cell-id>` for HAT-Bench cells = `<domain>_<generator>` (no timestamp);
for ablations = `<ablation-id>_<domain>[_<op>]_<generator>` (no timestamp).

Use `huggingface_hub.upload_file` per file (preserves resume); never
re-upload an existing identical file.

## 7. Helpful scripts already in this branch

| Path | Purpose |
|---|---|
| `evaluate/aes/run_gpt_per_row.py` | API-judge per-row driver (now `--csvs` + `--cell-name-template` + `--dataset-name`) |
| `evaluate/aes/run_detector_per_row.py` | Generic detector per-row driver (needs `--csvs` port from above) |
| `evaluate/aes/gemma_judge_runner.py` | Gemma local judge (now `--csvs`-aware) |
| `evaluate/aes/scripts/run_judges_on_ablations.sh` | API-judge ablation sweep launcher |
| `evaluate/aes/scripts/run_gemma_on_ablations.sh` | Gemma ablation sweep launcher |
| `evaluate/aes/scripts/queue_seqxgpt_default_setting.sh` | Wait-then-launch for §1b |
| `evaluate/aes/scripts/plot_judge_ablations.py` | 4 LLM judges × 3 ablations × 4 domains plots+tables |
| `evaluate/aes/scripts/plot_judge_accuracy_by_domain.py` | LLM judges on HAT-Bench (per (judge, generator) lines) |

## 8. What to do FIRST on the cloud host

1. `uv sync` and verify the 4 judge SDKs import (anthropic, google.genai,
   openai, python-dotenv).
2. Pull or rebuild local checkpoints for §1b (seqxgpt) and §3b (each
   tuned-on-new-data detector).
3. Re-run `evaluate/aes/scripts/plot_judge_ablations.py` against the
   local mirror to confirm the discovery patterns work on the new host.
4. Then start `default_setting` Qwen-cell pushes (§1a) — quick wins,
   no new compute.
5. Then §1b seqxgpt (1 GPU + 1-2 partner GPUs).
6. Then §3a + §3b in parallel batches across the 649 GPUs (one detector
   per GPU group).

## Cell-count totals

| Sweep | New cells |
|---|---|
| §1a (HF push only) | 28 |
| §1b seqxgpt | 15 |
| §1c adaloc gap | 0–4 |
| §2 Gemma finish | 5 |
| §3a default × ablations | 200 |
| §3b tuned × ablations | 220 |
| §5 bug-fix re-runs | up to 45 |
| **Total compute new** | **≤513 cells** |
