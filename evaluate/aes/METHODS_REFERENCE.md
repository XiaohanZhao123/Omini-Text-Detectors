# HAT-Bench detection methods — single-source reference

This is the one document a new reader should be able to open and fully understand:
**what each method is, where it came from, what we changed, what's broken, and how to interpret the result tables.** All other aggregate tables (`MACRO_F1_*.md`, `F1_PER_VERSION.md`, `SENTENCE_F1_PER_GENERATOR.md`, `STRANGE_RESULTS_DIAGNOSTIC.md`, `HATBENCH_AGGREGATE_*.md`) reference the names and categories defined here.

Naming is deliberate. If a name appears in this document, it appears identically in every other table.

---

## 1. About HAT-Bench (the dataset being evaluated)

HAT-Bench (April-15 release) is a corpus of human-AI co-authored essays with per-sentence and per-token AI annotations. It has:

- **4 domains**: `essays` (student writing), `abstracts` (academic), `news` (journalism), `reports` (technical)
- **3 text generators** (the LLMs that wrote the AI portions of the essays):
  - **GPT-5.4** = OpenAI `gpt-5.4-2026-03-05`
  - **GPT-5.4-nano** = OpenAI `gpt-5.4-nano-2026-03-17`
  - **Gemini-2.5-Flash** = Google `gemini-2.5-flash`
  - (`reports` domain has no GPT-5.4 cells; everywhere else is full 4×3.)
- **9 cumulative versions per essay** (v0–v8), each adding one editing operation. **Operation is fixed per version**:

| version | operation | typical AI ratio |
|---|---|---|
| v0 | none (pure human) | 0% |
| v1 | polish | 15% |
| v2 | paraphrase | 30% |
| v3 | style | 45% |
| v4 | compress | 50% |
| v5 | expand | 65% |
| v6 | style | 80% |
| v7 | paraphrase | 90% |
| v8 | polish | 100% |

Same operation appears at multiple AI ratios (polish at v1=15% and v8=100%, paraphrase at v2/v7, style at v3/v6) — this is what enables the "fix the operation, vary the AI ratio" comparison your colleague asked for.

**Total per-cell counts**: ~298 essays per (domain × generator × version) cell, ≈2 686 essays per (domain × generator) summed across versions.

---

## 2. Categories of method (this is the most important taxonomy in the document)

Every detector below falls into exactly one of these four categories. **Aggregate tables keep them in separate sections; never mix them.**

| Category | Plain-English meaning | What we did |
|---|---|---|
| **Reproduction** | Paper's published checkpoint AND paper-stated operating point. | Loaded the official model, applied the paper's threshold/protocol verbatim. Numbers reflect "the paper's method, applied to HAT-Bench." If HAT-Bench is OOD for that paper, the method may still score poorly — that's an honest transfer result, not a bug. |
| **Calibrated** | Paper's checkpoint BUT operating point (threshold / confidence cutoff / decision rule) chosen by us because the paper didn't specify one for this kind of data. | Same model weights as the paper; we picked a cutoff. Numbers reflect "the paper's model with our judgment call." Always footnoted with the cutoff value. |
| **Fine-tuned** | We re-trained the detector on HAT-Bench training data. | Lives under `HAT-Baselines/baseline_results/tuned_on_new_data/` on HF. Numbers reflect "what the architecture can achieve when trained on this distribution." Not comparable to reproduction or calibrated rows — different underlying model weights. |
| **Excluded** | Known to be broken, OOD-fundamentally, or missing required artifact. | Intentionally **does not appear in any aggregate**. Documented here so anyone wondering "why isn't X in the table?" has the answer. |

---

## 2.b Granularity support per method (verified against original paper + GitHub repo)

Each method's **paper primary** granularity determines which HF subfolder it lives
in (`default_setting/<granularity>/<method>/`). "All supported" lists the granularities
the original paper or repo demonstrates. Verified via sub-agent audits 2026-04-22.

| Method | Paper primary | All supported (paper/repo) | Notes |
|---|---|---|---|
| damasha | **token** | token, span (derived from token boundaries) | Paper headline is token-F1 ~0.98 on DAMASHA-MAS. |
| gigacheck | **span** | span (DETR detection model — headline), doc (separate Mistral-7B classification head) | GigaCheck paper ships TWO models; headline is the DETR interval detector. |
| adaloc | **sentence** | sentence only | Paper abstract: "machine-generated sentences within a document". |
| seqxgpt | **sentence** | word (BMES mechanism), sentence (paper headline), doc (aggregation) | Paper title: "Sentence-Level AI-Generated Text Detection". Our wrapper emits word-level; aggregation to sentence is downstream. |
| sendetex | **sentence** | sentence (aggregable to doc) | Paper title: "Sentence-Level AI-Generated Text Detection for Human-AI Hybrid Content". |
| genai-sentence | **sentence** | sentence, boundary (as a downstream evaluation metric) | Title: "Sentence-Level Segmentation of AI-Generated and Human Text". |
| gl-clic | **sentence** | sentence only | Title: "...for Sentence-Level AI-Generated Text Detection". |
| Judge: GPT-5.4, Judge: Gemini-3-Flash | sentence | sentence (+ doc trivially) | LLM-judges; no paper to constrain granularity. |
| detectllm | **doc** | doc only | LRR scalar per passage; paper evaluates on XSum/SQuAD/WritingPrompts. |
| fast-detectgpt | **doc** | doc only | Conditional probability curvature; paper uses ~150-word passages. |
| e5-small | **doc** | doc only | Sequence classification head; paper-equivalent protocol is one label per input. |
| desklib | **doc** | doc only | Same pattern as e5-small. |
| ood-llm-detect | **doc** | doc only | DeepSVDD on pooled SimCSE-RoBERTa embedding. |
| radar | **doc** | doc only | RoBERTa-large classifier, max 512 tokens. |
| roberta-openai | **doc** | doc only | OpenAI GPT-2 detector, passage-level. |

### Paper-original checkpoint availability (critical for any "paper-recipe" claim)

Audited 2026-04-23 via codex subagent against each method's upstream repo + arxiv paper. Four fine-tuned-only methods have **no publicly-released paper-original checkpoint** — authors released training code + data only. Running "paper-recipe inference" for these methods would require training from scratch on the paper's stated train split, not just loading weights.

| Method | Paper-original checkpoint | Our fine-tuned HF path |
|---|---|---|
| gl-clic | ❌ not released (IJCNLP 2025; repo `adirizq/gl-clic`) | `HAT-Baselines/.../tuned_on_new_data/sentence/gl-clic/` |
| genai-sentence | ❌ not released (arxiv 2509.17830; repo `saitejalekkala33/GenAI_Detect_Sentence_Level`) | `.../tuned_on_new_data/sentence/genai-sentence/` |
| sendetex | ❌ not released (EMNLP 2025; repo `TristoneJiang/SenDetEX`) | `.../tuned_on_new_data/sentence/sendetex/` |
| seqxgpt | ✅ paper-faithful **reproduction** at `zcahjl3/seqxgpt-detector` (own training, paper's recipe + paper's SniffeR-Reuters data). Headline metrics within Δ ≤ 0.1 pt of paper (sent-Acc 95.6 vs 95.7). Label set = `{gpt2, gptneo, gptj, llama, gpt3re, human}` — paper's own labels, **not** HAT-Bench labels — so this is a reproduction, not a tune on our data. Re-classified 2026-04-26: belongs in `default_setting/`. | `.../default_setting/sentence/seqxgpt/` |

**Consequence**: 3 methods (gl-clic, genai-sentence, sendetex) exist in `tuned_on_new_data/` only, with no `default_setting/` counterpart. Any "paper-recipe" comparison to their tuned variants is not possible without either (a) training from scratch on the paper's stated train set, or (b) the authors releasing weights. seqxgpt has both a paper-faithful default (our reproduction) and a HAT-Bench-tuned variant.

### Architecture-mismatch caveats (fine-tuned variants that deviate from the paper's granularity)

- **gigacheck-tuned** is placed under `tuned_on_new_data/doc/gigacheck/` (NOT `span/`) because its output collapsed to **doc**-level: every essay has ONE unique word-prob across all 8046 HAT-Bench essays — the doc score is broadcast to every word/sentence. The paper-recipe `gigacheck` in `default_setting/span/gigacheck/` preserves DETR intervals; the fine-tuned head was retrained as a doc-level classifier. **Not comparable to the paper-recipe variant as the same method** — compare them as two architectures that share a backbone.
- **seqxgpt wrapper** emits word-level BMES output; the paper's headline metric is sentence-level. Sentence aggregation is performed downstream of the wrapper. File layout places it under `sentence/` per the paper's primary.

## 3. Method index (one row per method)

Sorted by category, then by name. The "internal id" column matches the folder name on HuggingFace and the row label in result tables.

| Display name | Internal id | Category | Granularity | Paper checkpoint |
|---|---|---|---|---|
| e5-small | `e5-small` | Reproduction | doc | `MayZhou/e5-small-lora-ai-generated-detector` |
| ood-llm-detect | `ood-llm-detect` | Reproduction | doc | paper-released DSVDD checkpoint |
| damasha (token only) | `damasha` (token rows only) | Reproduction | token | HF `RoBERTa_ModernBERT_CRF.pth` |
| Judge: GPT-5.4 (raw) | `gpt54-sent-conf-none` (hard label) | Reproduction | sentence | OpenAI API `gpt-5.4`, reasoning_effort=none |
| Judge: Gemini-3-Flash-Preview (raw) | `gemini-flash-sent-conf-minimal` (hard label) | Reproduction | sentence | Google API `gemini-3-flash-preview`, thinking_level=minimal |
| desklib | `desklib` | Calibrated | doc | `desklib/ai-text-detector-v1.01` |
| gigacheck | `gigacheck` (doc + boundary rows) | Calibrated | doc, boundary | paper checkpoint |
| ~~Judge: GPT-5.4 (conf≥0.15)~~ | RETIRED | — | — | Cross-model calibration leak (0.15 was tuned for gpt54 then reused unchanged for gemini); removed 2026-04-24. |
| ~~Judge: Gemini-3-Flash-Preview (conf≥0.15)~~ | RETIRED | — | — | Same. |
| adaloc-tuned | `adaloc` (HF `tuned_on_new_data/`) | Fine-tuned | sentence | trained on HAT-Bench train split |
| damasha-tuned | `damasha` (HF tuned) | Fine-tuned | token | trained on HAT-Bench |
| fast-detectgpt-tuned | `fast-detectgpt` (HF tuned) | Fine-tuned | doc | calibrated head trained on HAT-Bench |
| genai-sentence-tuned | `genai-sentence` (HF tuned) | Fine-tuned | sentence | trained on HAT-Bench |
| genai-sentence-v2-tuned | `genai-sentence-v2` (HF tuned) | Fine-tuned | sentence | architecture variant 2 |
| gigacheck-tuned | `gigacheck` (HF tuned) | Fine-tuned | token, boundary | DETR head fine-tuned |
| gl-clic-tuned | `gl-clic` (HF tuned) | Fine-tuned | sentence | trained on HAT-Bench |
| gl-clic-simplified-tuned | `gl-clic-simplified` | Fine-tuned | sentence | architecture-pruned variant |
| gl-clic-v2-tuned | `gl-clic-v2` (HF tuned) | Fine-tuned | sentence | architecture variant 2 |
| sendetex-tuned | `sendetex` (HF tuned) | Fine-tuned | sentence | trained on HAT-Bench |
| seqxgpt-tuned | `seqxgpt` (HF tuned) | Fine-tuned | sentence | trained on HAT-Bench |
| ~~detectllm~~ | `detectllm` | **Excluded** | — | threshold-null bug — see §5 |
| ~~fast-detectgpt~~ | `fast-detectgpt` (default) | **Excluded** | — | score-orientation flip — see §5 |
| ~~damasha (doc)~~ | derived doc-level row | **Excluded** | — | always-AI bias on HAT-Bench |
| ~~radar~~ | `radar` | **Excluded** | — | trained on Vicuna-7B; HAT-Bench OOD |
| ~~roberta-openai~~ | `roberta-openai` | **Excluded** | — | trained on GPT-2; HAT-Bench OOD |
| ~~adaloc (default)~~ | `adaloc` (default_setting) | **Excluded** | — | YAML `checkpoint_path: null`; only fine-tuned variant has a real checkpoint |

---

## 4. Per-method full entries

Every entry below uses the same field set. If a field is N/A for a category (e.g. fine-tuned methods don't have a "paper-stated operating point"), it's marked N/A and explained in the "what's different from a paper reproduction" line.

### 4.1 — Reproduction methods

#### e5-small

- **Category**: Reproduction
- **Display name in tables**: `e5-small`
- **Granularities reported**: doc only
- **Paper**: Microsoft Hackathon 2024 / RAID benchmark submission
- **Origin code**: `baseline/e5_small/`
- **Checkpoint**: `MayZhou/e5-small-lora-ai-generated-detector` (HuggingFace)
- **Output**: P(AI) ∈ [0, 1] per document
- **Operating point**: threshold = **0.85** (paper-stated as "original training used 0.85 for optimal accuracy")
- **HAT-Bench coverage**: 11 cells (4 domains × 3 generators, minus reports/gpt-5.4)
- **Known caveats**: trained on RAID corpus; HAT-Bench is out-of-distribution for the original training set, so transfer may underperform RAID-reported numbers. This is honest, not a bug.
- **What to do if results look broken**: First check that the threshold is 0.85 (not the generic 0.5). If predictions collapse to one class, run `evaluate/aes/scripts/diagnose_strange_results.py` to print the prediction-distribution table for this method.

#### ood-llm-detect

- **Category**: Reproduction
- **Display name**: `ood-llm-detect`
- **Granularities**: doc only
- **Paper**: NeurIPS 2025
- **Origin code**: `baseline/ood-llm-detect/`
- **Checkpoint**: paper-released DeepSVDD checkpoint, mode=`raid`
- **Output**: distance score (lower = more like training distribution = more human)
- **Operating point**: 0.5 (standard DeepSVDD operating point used in the paper's RAID evaluation)
- **HAT-Bench coverage**: 11 cells
- **Known caveats**: trained on RAID; HAT-Bench OOD. Aggregate numbers are an honest transfer test, not a paper reproduction on the paper's own benchmark.
- **What to do if results look broken**: confirm checkpoint mode is `raid`, not `default`. If F1 is degenerate, check the score range: a flat score distribution (all values within 0.01 of each other) means the SVDD center wasn't loaded correctly.

#### damasha (token-only reproduction row)

- **Category**: Reproduction (**token granularity only**; doc and boundary rows derived from token are *not* reproduction — they're calibrated and excluded respectively)
- **Display name**: `damasha`
- **Granularities**: token (the doc-derived row is excluded; see §5 entry below)
- **Paper**: DAMASHA: Text-Source Localization in LLM-Generated Texts
- **Origin code**: `baseline/damasha/`
- **Checkpoint**: `RoBERTa_ModernBERT_CRF.pth` from the paper's HF release
- **Output**: per-token CRF-decoded label (0=human, 1=AI), via Viterbi decoding
- **Operating point**: CRF Viterbi (no threshold; argmax over CRF transition scores) — this is the paper's protocol exactly.
- **HAT-Bench coverage**: 11 cells under `results/new_data_eval/token/damasha/`
- **Known caveats**: paper benchmark is DAMASHA-MAS, not HAT-Bench; HAT-Bench is OOD. Token alignment uses damasha's RoBERTa+ModernBERT tokenizer — not directly comparable to gigacheck's char-interval tokenization.
- **What to do if results look broken**: confirm the CRF transition matrix loaded from the checkpoint. If every token is predicted human, the CRF state-zero bias is dominating; in that case the issue is the same one that breaks the doc-derived row (see §5).

#### Judge: GPT-5.4 (raw)

- **Category**: Reproduction (no operating point chosen by us — the LLM emits a 0/1 directly)
- **Display name**: `Judge: GPT-5.4 (reasoning=none)`
- **Internal id**: `gpt54-sent-conf-none` (hard-label variant, i.e. using `gpt.sentence_labels` directly)
- **Granularities**: sentence
- **Paper**: N/A — this is an LLM-as-judge baseline, not a published detection method.
- **Origin code**: `evaluate/aes/sentence_level_v0v8.py:OpenAISentenceDetector`
- **Model**: OpenAI `gpt-5.4`, `reasoning_effort=none`
- **Output**: per-sentence binary label + confidence (Pydantic-structured response)
- **Operating point**: hard label = the LLM's raw 0/1 emission. No threshold of ours.
- **HAT-Bench coverage**: 11 cells under `HAT-Baselines/baseline_results/llm_judge/gpt54-sent-conf-none_*`
- **Known caveats**: cross-LLM-family bias — both judges score Gemini-generated text easier than GPT-generated text by ~0.4 F1_AI. This is a real same-family signature; reported in the per-generator table.
- **What to do if results look broken**: check `errors.n_api_errors` in summary.json (transient 503s); check `errors.n_length_mismatch` (LLM emitted wrong number of sentence labels — currently silently clipped to `min(len(gt), len(pred))`).

#### Judge: Gemini-3-Flash-Preview (raw)

- **Category**: Reproduction (same logic as the GPT judge — no operating point chosen by us in the hard-label variant)
- **Display name**: `Judge: Gemini-3-Flash-Preview (thinking=minimal)`
- **Internal id**: `gemini-flash-sent-conf-minimal` (hard-label variant)
- **Granularities**: sentence
- **Paper**: N/A
- **Origin code**: `evaluate/aes/sentence_level_v0v8.py:GeminiSentenceDetector`
- **Model**: Google `gemini-3-flash-preview`, `thinking_level=minimal`
- **Output**: per-sentence binary label + confidence (Pydantic-structured response)
- **Operating point**: hard label = LLM's raw emission
- **HAT-Bench coverage**: 11 cells under `HAT-Baselines/baseline_results/llm_judge/gemini-flash-sent-conf-minimal_*`
- **Known caveats**: same family-bias note. The judge is **Gemini 3** even though one of the generators is **Gemini 2.5** — different model versions; do not conflate.
- **What to do if results look broken**: same as GPT judge.

### 4.2 — Calibrated methods

#### desklib

- **Category**: Calibrated
- **Display name**: `desklib`
- **Granularities**: doc only
- **Paper**: Desklib AI Detector v1.01
- **Origin code**: `baseline/desklib/`
- **Checkpoint**: `desklib/ai-text-detector-v1.01`
- **Output**: P(AI) ∈ [0, 1]
- **Operating point**: threshold = **0.5** (chosen by us as a balanced default; the paper does not specify a binary threshold for this kind of mixed-AI data)
- **HAT-Bench coverage**: 11 cells
- **Known caveats**: well-calibrated probabilistic output; the 0.5 cutoff produces a monotone-increasing prediction rate from 19% AI at v0 to 75% AI at v8.
- **What to do if results look broken**: try threshold sweep — `desklib --threshold 0.3` for higher recall, `--threshold 0.7` for higher precision. If even sweeping doesn't help, the LoRA adapter likely failed to load (check `omini_text/configs/desklib.yaml`).

#### gigacheck (calibrated doc/boundary rows)

- **Category**: Calibrated (the **token row of gigacheck is in `Reproduction`** — the calibrated rows are the *derived* doc-level label and boundary outputs)
- **Display name**: `gigacheck` (in doc/boundary tables); also `gigacheck` in token table — same checkpoint, different operating points
- **Granularities**: doc, boundary
- **Paper**: GigaCheck (arXiv 2410.23728)
- **Origin code**: `baseline/gigacheck/`
- **Checkpoint**: paper-released DETR boundary detector
- **Output**: character-level intervals with per-interval confidence
- **Operating point**: `conf_interval_thresh = 0.8` (chosen by us; not in paper). Doc-level label = 1 iff total AI-interval coverage ≥ 50% of document. Both choices are ours.
- **HAT-Bench coverage**: 11 doc cells + 11 token cells
- **Known caveats**: raw character intervals are paper-faithful (token row); the doc-level binary derivation is OUR convention, not the paper's evaluation protocol.
- **What to do if results look broken**: lower `conf_interval_thresh` (try 0.5) for higher AI recall; check `truncated_text_len` in the row (the model's 1024-token cap means long essays are clipped).

#### ~~Judge: GPT-5.4 (conf≥0.15)~~ — RETIRED 2026-04-24

- **Removed**: cross-model calibration leak. The 0.15 threshold was picked empirically for the gpt54 judge and silently applied to other judges (gemini-flash, was about to be applied to gemma-4); no per-judge calibration was ever done. Reporting numbers under a foreign judge's threshold misrepresents that judge's behavior.
- **What replaces it**: only the LLM's raw hard-label readout (`at_label_threshold_0.5`). Per-sentence confidences are still in `predictions.jsonl` if anyone wants to compute a properly per-judge calibrated threshold via a held-out sweep.
- **Affected published cells**: 30 LLM-judge `summary.json` files on HF were stripped of the `at_conf>=0.15` blocks 2026-04-24 (see `evaluate/aes/scripts/strip_conf_threshold_leak.py`).

#### ~~Judge: Gemini-3-Flash-Preview (conf≥0.15)~~ — RETIRED 2026-04-24

- Same as above.

### 4.3 — Fine-tuned methods (`tuned_on_new_data/`)

All entries below were **trained on HAT-Bench training split** by the fine-tuning trainer in `evaluate/aes/eval_finetuned_detectors.py`. Their numbers reflect what the *architecture* achieves when given HAT-Bench-shaped training data. They are NOT comparable to the reproduction/calibrated rows because the underlying model weights are different.

For all fine-tuned methods, the uniform fields are:

- **Category**: Fine-tuned
- **Trained on**: HAT-Bench train split (4 domains × 3 generators)
- **Origin code**: `evaluate/aes/eval_finetuned_detectors.py`
- **Checkpoint location**: HF `HAT-Baselines/baseline_results/tuned_on_new_data/<id>_new4d_*` cells
- **Operating point**: as decided during training (each tuned model carries its own threshold in the `summary.json` provenance block)
- **What to do if results look broken**: re-load the checkpoint from HF; check `provenance.json["training_data_for_this_eval"]` to confirm it points at `data_25_04_15/`.

Per-method shorthand:

| Display name | Source architecture | Granularity emitted |
|---|---|---|
| `adaloc-tuned` | adaloc head over RoBERTa-OpenAI features | sentence |
| `damasha-tuned` | RoBERTa+ModernBERT+CRF | token |
| `fast-detectgpt-tuned` | calibrated head on top of fast-detectgpt features | doc |
| `genai-sentence-tuned` | DeBERTa-v3-base + sentence head, hidden=512, layers=2 | sentence |
| `genai-sentence-v2-tuned` | DeBERTa-v3-base + sentence head, hidden=128, layers=1 | sentence |
| `gigacheck-tuned` | DETR boundary detector | token, boundary |
| `gl-clic-tuned` | GL-CLiC sentence head with 4 auxiliary heads | sentence |
| `gl-clic-simplified-tuned` | GL-CLiC sentence head, auxiliary heads removed | sentence |
| `gl-clic-v2-tuned` | architectural variant of GL-CLiC | sentence |
| `sendetex-tuned` | SenDeText | sentence |
| `seqxgpt-tuned` | SeqXGPT (4-LM context) | sentence |

### 4.4 — Excluded methods (with reason and required fix to un-exclude)

If you encounter these methods in old documentation or local result folders, here is why they are NOT in the published aggregates and what would make them reportable.

#### detectllm (doc / token / boundary)

- **Category**: Excluded
- **Reason**: YAML `threshold: null`; the `detectllm` "lrr" score is an unbounded log-ratio (typical range 1.5–2.5), not a probability. Applying any 0.5 cutoff yields label=0 for every document → F1_AI = 0 across v1–v8.
- **Required fix to include**: calibrate a per-domain threshold from the v0+v8 score distribution (e.g. pick the median LRR), or report only AUROC (the paper's original metric) and skip binary classification.
- **Tracked**: this is what tasks #16/#23 partially addressed for AUROC; the binary-classification path is still uncalibrated.

#### fast-detectgpt (default_setting)

- **Category**: Excluded
- **Reason**: Score orientation is **inverted** on HAT-Bench. Prediction rate falls from 20% AI at v0 to 6% AI at v8 — i.e., the more AI text in the essay, the LESS likely the detector flags it. The score sign is wrong somewhere in the pipeline; this is a real bug, not OOD.
- **Required fix to include**: open issue #28. Either flip the score sign or trace which side of `score < threshold` is being labeled AI.

#### damasha (doc-derived row)

- **Category**: Excluded (the **token** row of damasha is *included* under Reproduction; only the doc-level label derived from it is excluded)
- **Reason**: Always-AI bias — the doc-level label projects "any AI token" → doc=AI, but the token-CRF leaks AI predictions on pure-human v0 essays at a 64% rate. This is open issue #29.
- **Required fix to include**: investigate whether the leak is from CRF state-zero priors or from per-token logits saturating; threshold the per-token confidence before doc-level pooling.

#### radar

- **Category**: Excluded
- **Reason**: Trained exclusively on Vicuna-7B-generated text. HAT-Bench uses GPT-5.4/Gemini-2.5-Flash. Predicts ~10–18% AI everywhere — this is genuine domain shift, not a code bug. The detector is functioning as designed; the design just doesn't transfer.
- **Required fix to include**: nothing reasonable. Could re-train on HAT-Bench (which would make it a Fine-tuned method and move it to category 4.3).

#### roberta-openai

- **Category**: Excluded
- **Reason**: Trained on GPT-2 1.5B output only. Predicts ~2% AI on GPT-5.4/Gemini-2.5-Flash text. Same OOD story as radar.
- **Required fix to include**: same as radar — re-train, then categorize as Fine-tuned.

#### adaloc (default_setting)

- **Category**: Excluded (the `adaloc-tuned` variant under §4.3 is included)
- **Reason**: YAML `checkpoint_path: null` — the default config does not point at any usable checkpoint. The 10 cells on HF under `default_setting/adaloc_*` were produced before the checkpoint was lost / never existed; their provenance is unverified.
- **Required fix to include**: locate or re-derive the original checkpoint, OR drop the default_setting variant entirely (we recommend the latter; the fine-tuned variant supersedes it).

---

## 5. How to read the result tables

There are five aggregate tables on HF under `HAT-Baselines/baseline_results/aggregates/`. They differ only in slicing and metric choice; the underlying data is identical (re-derived from per-cell `predictions.jsonl`).

| Table | What it shows | Best for |
|---|---|---|
| `MACRO_F1_PER_VERSION.md` | One number per (method, version): macro-F1 = (F1_AI + F1_human)/2 | Headline at-a-glance ranking |
| `F1_PER_VERSION.md` | Per cell: `F1_human / F1_AI` separately | Diagnosing class imbalance / one-class collapse |
| `SENTENCE_F1_PER_GENERATOR.md` | Sentence judges only, per-generator breakdown | Same-family bias analysis |
| `STRANGE_RESULTS_DIAGNOSTIC.md` | (a) hard vs calibrated for sentence judges; (b) per-version % predicted AI for doc detectors | Debugging why a method's number looks wrong |
| `HATBENCH_AGGREGATE_<date>.csv` | Full cross product per (granularity, method, generator, domain, version, operation) — machine-readable only (CSV; the auto-generated MD pivot was retired 2026-04-22 because the per-method tables above cover the same data more cleanly) | Custom slicing in pandas / Excel |

### Cell conventions

- Cells use the F1 of one class. **Macro-F1 = (F1_AI + F1_human) / 2.** A single per-cell number lets you compare methods at a glance.
- `-` means the class is absent from the GT for that version. Doc-level v0 has no AI documents (every essay is pure human), so `F1_AI` is undefined → reported as `-`. Means in the `mean` column EXCLUDE undefined cells.
- The doc-level GT rule is **`y=1 iff version != v0`**. This collapses v1=15%-AI, v5=65%-AI, and v8=100%-AI all into a single "AI" class. An alternative GT rule `y=1 iff AI_token_ratio > 0.5` would change v1/v2 numbers materially; we did NOT use it. If you want to test sensitivity to this choice, the per-row `AI_token_ratio` field is in every `predictions.jsonl`.
- Token-level F1s are NOT directly comparable across detectors because each tokenizes differently (`damasha` uses RoBERTa+ModernBERT, `gigacheck` uses character-level intervals projected to tokens, `detectllm` uses its own LM tokenizer). Within one detector, per-version comparison is valid; between detectors at token level, treat with care.
- Boundary-level F1 here uses a **token-position approximation** ("is this token the start of a new segment?"), NOT IoU-based segment matching. v0 and v8 boundary cells are degenerate (a single segment, so no boundaries to detect — they trivially score ~0.99); ignore them when comparing.

### Pooling

Macro-F1 is **micro-pooled at the unit level across (domain × generator) cells, then macro-averaged across the two classes (AI / human).** This means a small essay's sentences are weighted the same as a large essay's sentences (each sentence is one unit). If you want cell-level macro-averaging instead, sum per-cell F1s and divide by the cell count — but the current micro-pooling is more appropriate for an unbalanced corpus.

### Per-version vs per-operation

- "Per version" = pooled across all operations active at that version. Since operation is fixed per version (§1), per-version IS per-operation for that version.
- The interesting cross-comparison is "same operation at different AI ratios": polish at v1 (15%) vs v8 (100%), paraphrase at v2 (30%) vs v7 (90%), style at v3 (45%) vs v6 (80%). The CSV in `HATBENCH_AGGREGATE_*.csv` has a per-(version, operation) row for every cell — load it in pandas and pivot.

---

## 6. File map (where everything lives)

### On HuggingFace (`HAT-Baselines/baseline_results/`)

```
default_setting/      # Reproduction + Calibrated cells (one folder per detector × domain × generator)
  desklib_new4d_essays_gpt-5.4_<ts>/
  desklib_new4d_essays_gpt-5.4-nano_<ts>/
  ...
  e5-small_new4d_*/
  ood-llm-detect_new4d_*/
  gigacheck_new4d_*/

llm_judge/            # Sentence-level LLM judges (Reproduction + Calibrated variants share the same folder)
  gpt54-sent-conf-none_new4d_*/
  gemini-flash-sent-conf-minimal_new4d_*/

tuned_on_new_data/    # Fine-tuned cells (one folder per fine-tuned variant)
  adaloc_*/  damasha_*/  fast-detectgpt_*/
  genai-sentence_*/  genai-sentence-v2_*/
  gigacheck_*/
  gl-clic_*/  gl-clic-simplified_*/  gl-clic-v2_*/
  sendetex_*/  seqxgpt_*/

aggregates/           # The roll-ups described in §5
  METHODS_REFERENCE.md          ← this file (start here)
  MACRO_F1_PER_VERSION.md
  F1_PER_VERSION.md
  SENTENCE_F1_PER_GENERATOR.md
  STRANGE_RESULTS_DIAGNOSTIC.md
  HATBENCH_AGGREGATE_<date>.csv  (raw machine-readable; no MD twin)
```

Each cell folder always contains exactly four files: `predictions.jsonl` (per-row), `provenance.json` (training_data + calibration_on_new_data flags), `run_config.json` (CLI / YAML snapshot), `summary.json` (computed metrics).

### Locally (in this repo)

```
omini_text/configs/<detector>.yaml          # default config; the operating-point source
omini_text/detectors/<detector>_detector.py # wrapper class
evaluate/aes/sentence_level_v0v8.py         # OpenAI + Gemini sentence-judge implementations
evaluate/aes/run_gpt_per_row.py             # the runner that writes llm_judge/ cells
evaluate/aes/run_detector_per_row.py        # the runner that writes default_setting/ cells
evaluate/aes/eval_finetuned_detectors.py    # the fine-tuning trainer for tuned_on_new_data/
evaluate/aes/scripts/_naming.py             # the SINGLE source of display names + categories
evaluate/aes/scripts/macro_f1_per_version.py
evaluate/aes/scripts/sentence_per_generator.py
evaluate/aes/scripts/diagnose_strange_results.py
evaluate/aes/scripts/aggregate_hatbench_results.py
evaluate/aes/scripts/augment_gemini_summaries.py
evaluate/aes/{MACRO_F1_PER_VERSION,F1_PER_VERSION,SENTENCE_F1_PER_GENERATOR,
              STRANGE_RESULTS_DIAGNOSTIC,METHODS_REFERENCE}.md
evaluate/aes/HATBENCH_AGGREGATE_<date>.csv  # raw cross-product (CSV only)
```

---

## 7. Reproducing the aggregates from scratch

If you want to regenerate every table from scratch (e.g. after adding a new detector), run:

```bash
# 1. The runners (already done; takes hours)
uv run python evaluate/aes/run_detector_per_row.py --detector <name> --models gpt-5.4 gpt-5.4-nano gemini-2.5-flash --fields essays abstracts news reports
uv run python evaluate/aes/run_gpt_per_row.py --method <judge-method> [...]

# 2. Re-derive metrics + cross-cell roll-up (no inference; fast)
uv run python evaluate/aes/scripts/aggregate_hatbench_results.py --rewrite-summaries

# 3. Per-granularity macro-F1 + per-class F1 tables
uv run python evaluate/aes/scripts/macro_f1_per_version.py

# 4. Sentence-judge per-generator tables
uv run python evaluate/aes/scripts/sentence_per_generator.py

# 5. Diagnostic tables (calibration + class collapse)
uv run python evaluate/aes/scripts/diagnose_strange_results.py

# 6. Push everything to HF
uv run python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='evaluate/aes',
    path_in_repo='aggregates',
    repo_id='HAT-Baselines/baseline_results',
    repo_type='dataset',
    allow_patterns=['*.md', '*.csv'],
    commit_message='regenerate aggregate tables',
)
"
```

To **add a new detector** to the aggregates: edit `evaluate/aes/scripts/_naming.py` to register its display name and category, then re-run steps 2–6 above.

---

## 8. Document version

- **First written**: 2026-04-22 by Claude on behalf of `zcahjl3@gmail.com`
- **Source of truth for naming**: `evaluate/aes/scripts/_naming.py` (`DETECTOR_LABEL`, `GENERATOR_LABEL`, `CATEGORY` dicts)
- **Source of truth for HAT-Bench data**: `draft/data_25_04_15/` (4 domains × 3 generators × 9 versions; April-15 deduplicated release)

If a method is added or a category changes, edit `_naming.py` first, then re-render this document.
