#!/usr/bin/env python3
"""Build aggregate CSVs over tuned_on_new_data/ predictions.

Inputs: a local mirror of HF tuned_on_new_data/ (predictions.jsonl per cell).

Outputs (under --out-dir):
  main_<level>.csv, ablation1_<level>.csv, ablation2_<level>.csv, ablation3_<level>.csv
    Per-row schema (one row = one (method, gen, domain, version, operation, ablation) slice):
      method, level, generator, domain, version, operation, ablation,
      n_units, n_expected, n_missing,
      tp, tn, fp, fn,
      accuracy, fpr, fnr, precision_ai, f1_ai, auroc, coverage_status
  tuned_on_new_data_method_summary.csv (per split × level × method, summed)
  tuned_on_new_data_coverage_report.csv (per method × split coverage)

Levels are method-natural; do not convert.
"""
from __future__ import annotations
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import ast
import pandas as pd
from sklearn.metrics import roc_auc_score

NATURAL_LEVEL = {
    'fast-detectgpt-default': 'document',
    'fast-detectgpt-tuned': 'document',
    'adaloc': 'sentence',
    'genai-sentence-v2': 'sentence',
    'gl-clic-simplified': 'sentence',
    # SeqXGPT paper: "SeqXGPT: Sentence-Level AI-Generated Text Detection".
    # Our infer_seqxgpt.py only saved word-level BMES outputs (no sentence post-
    # processing). Aggregator does the word→sentence majority vote at runtime
    # using sentence boundaries from the source CSV (see aggregate_seqxgpt_cell).
    'seqxgpt': 'sentence',
    'gigacheck': 'span',
    'damasha': 'token',
}

# Where each method's cells live in HF (we use doc/gigacheck because it has full 28 cells incl. ablation;
# span/gigacheck is a subset.)
# fast-detectgpt-default and fast-detectgpt-tuned share cells but differ in which
# label field to use (detection_doc_label_default vs _calibrated).
METHOD_HF_PATH = {
    'fast-detectgpt-default': 'doc/fast-detectgpt',
    'fast-detectgpt-tuned': 'doc/fast-detectgpt',
    'adaloc': 'sentence/adaloc',
    'genai-sentence-v2': 'sentence/genai-sentence-v2',
    'gl-clic-simplified': 'sentence/gl-clic-simplified',
    'seqxgpt': 'sentence/seqxgpt',
    'gigacheck': 'doc/gigacheck',
    'damasha': 'token/damasha',
}

DOMAINS = ('essay', 'abstract', 'news', 'report')
SPLITS = ('main', 'ablation1', 'ablation2', 'ablation3')


def parse_cell(cell: str):
    """Return (domain, split, ablation_label, generator_override).

    Examples:
      'essay'                                       → ('essay', 'main', '', None)
      'essay_qwen3-8b'                              → ('essay', 'main', '', 'qwen3-8b')
      'essay_ablation1_compress_gemini-2.5-flash'   → ('essay', 'ablation1', 'A1_compress', 'gemini-2.5-flash')
      'essay_ablation2_gemini-2.5-flash'            → ('essay', 'ablation2', 'A2', 'gemini-2.5-flash')
      'essay_ablation3_gemini-2.5-flash'            → ('essay', 'ablation3', 'A3', 'gemini-2.5-flash')
    """
    for d in DOMAINS:
        if cell == d:
            return d, 'main', '', None
        if cell == f'{d}_qwen3-8b':
            return d, 'main', '', 'qwen3-8b'
        m = re.match(rf'^{d}_ablation1_(compress|expand|paraphrase)_(.+)$', cell)
        if m:
            return d, 'ablation1', f'A1_{m.group(1)}', m.group(2)
        m = re.match(rf'^{d}_ablation2_(.+)$', cell)
        if m:
            return d, 'ablation2', 'A2', m.group(1)
        m = re.match(rf'^{d}_ablation3_(.+)$', cell)
        if m:
            return d, 'ablation3', 'A3', m.group(1)
    raise ValueError(f'cannot parse cell name: {cell}')


def get_doc_pred(method: str, rec: dict):
    """Doc-level label. fast-detectgpt has two threshold variants."""
    if method == 'fast-detectgpt-default' and 'detection_doc_label_default' in rec:
        return int(rec['detection_doc_label_default'])
    if method == 'fast-detectgpt-tuned' and 'detection_doc_label_calibrated' in rec:
        return int(rec['detection_doc_label_calibrated'])
    # Legacy fallback (single fast-detectgpt method, prefer calibrated)
    if method == 'fast-detectgpt' and 'detection_doc_label_calibrated' in rec:
        return int(rec['detection_doc_label_calibrated'])
    return int(rec['detection_doc_label'])


def get_doc_score(method: str, rec: dict):
    return rec.get('detection_doc_score', None)


def confusion_doc(method: str, rec: dict):
    g = int(rec.get('doc_label_gt', 0))
    p = get_doc_pred(method, rec)
    tp = int(g == 1 and p == 1); tn = int(g == 0 and p == 0)
    fp = int(g == 0 and p == 1); fn = int(g == 1 and p == 0)
    return tp, tn, fp, fn, 1, [g], [get_doc_score(method, rec)]


def confusion_list(pred, gt, scores=None):
    """Element-wise confusion. Returns (tp, tn, fp, fn, n, labels, scores)."""
    n = min(len(pred or []), len(gt or []))
    if n == 0:
        return 0, 0, 0, 0, 0, [], []
    tp = tn = fp = fn = 0
    labs = []; scs = []
    for i in range(n):
        g = int(gt[i]); p = int(pred[i])
        labs.append(g)
        if scores is not None and i < len(scores):
            scs.append(float(scores[i]))
        if g == 1 and p == 1: tp += 1
        elif g == 0 and p == 0: tn += 1
        elif g == 0 and p == 1: fp += 1
        else: fn += 1
    return tp, tn, fp, fn, n, labs, scs


def confusion_sentence(method: str, rec: dict):
    return confusion_list(rec.get('detection_sentence_labels', []),
                          rec.get('gt_sentence_labels', []),
                          rec.get('detection_sentence_scores', None))


def confusion_word(method: str, rec: dict):
    return confusion_list(rec.get('detection_word_labels', []),
                          rec.get('tok_labels', []),
                          rec.get('detection_word_probs', None))


CONFUSION_FN = {
    'document': confusion_doc,
    'sentence': confusion_sentence,
    'span': confusion_word,
    'token': confusion_word,
}


# --- SeqXGPT: word→sentence aggregation via CSV sentence boundaries ---
_CSV_CACHE = {}


def _safe_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            out = ast.literal_eval(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def _load_csv_lookup(repo_root: Path, csv_relpath: str):
    """Build lookup (essay_id, version, ai_model, operation) → (word_per_sent, sent_labels)."""
    if csv_relpath in _CSV_CACHE:
        return _CSV_CACHE[csv_relpath]
    csv_path = repo_root / csv_relpath
    if not csv_path.exists():
        # try alternate roots that the run_config might have used
        alt = Path('/datadrive/xiaohan/Omini-Text') / csv_relpath
        if alt.exists():
            csv_path = alt
        else:
            print(f'  [warn] CSV not found: {csv_relpath}')
            _CSV_CACHE[csv_relpath] = {}
            return {}
    df = pd.read_csv(csv_path, low_memory=False)
    table = {}
    for _, r in df.iterrows():
        essay_id = str(r.get('essay_id', ''))
        version = str(r.get('version', ''))
        ai_model_raw = r.get('ai_model', '')
        if pd.isna(ai_model_raw) or not ai_model_raw:
            ai_model_raw = r.get('model_used', '')
        ai_model = str(ai_model_raw or '')
        # Normalize "gemini/gemini-2.5-flash" → "gemini-2.5-flash"
        if '/' in ai_model:
            ai_model = ai_model.split('/')[-1]
        operation = str(r.get('operation', ''))
        sents = _safe_list(r.get('sentences'))
        slabs = _safe_list(r.get('sent_labels'))
        if not sents:
            continue
        word_per_sent = [len(str(s).split()) for s in sents]
        key = (essay_id, version, ai_model, operation)
        table[key] = (word_per_sent, [int(x) for x in slabs])
        # Also index by (essay_id, version, operation) for ablation lookups where ai_model is missing in record
        table.setdefault((essay_id, version, '', operation), (word_per_sent, [int(x) for x in slabs]))
    _CSV_CACHE[csv_relpath] = table
    return table


def confusion_seqxgpt_sentence(rec: dict, csv_lookup: dict, gen_override: Optional[str]):
    """Aggregate seqxgpt's word-level predictions to sentence level via majority vote.

    Returns (tp, tn, fp, fn, n_sents, gt_labs, sent_scores) at sentence level.
    """
    essay_id = str(rec.get('essay_id', ''))
    version = str(rec.get('version', ''))
    ai_model = gen_override if gen_override else str(rec.get('ai_model', ''))
    operation = str(rec.get('operation', ''))
    key = (essay_id, version, ai_model, operation)
    info = csv_lookup.get(key)
    if info is None:
        # Try without ai_model (some CSV columns might differ)
        for k in csv_lookup:
            if k[0] == essay_id and k[1] == version and k[3] == operation:
                info = csv_lookup[k]; break
    if info is None:
        return 0, 0, 0, 0, 0, [], []
    word_per_sent, sent_labels = info
    word_preds = rec.get('detection_word_labels', [])
    word_probs = rec.get('detection_word_probs', None)

    tp = tn = fp = fn = 0
    n = 0
    gts = []; scs = []
    word_idx = 0
    for i, n_words in enumerate(word_per_sent):
        if i >= len(sent_labels):
            break
        end = word_idx + n_words
        # majority vote on this sentence's word labels
        chunk = word_preds[word_idx:end] if end <= len(word_preds) else word_preds[word_idx:]
        if not chunk:
            word_idx = end
            continue
        ai_count = sum(1 for x in chunk if int(x) == 1)
        sent_pred = 1 if ai_count * 2 > len(chunk) else 0  # >50% = AI
        sent_gt = int(sent_labels[i])
        if word_probs is not None:
            chunk_probs = word_probs[word_idx:end] if end <= len(word_probs) else word_probs[word_idx:]
            if chunk_probs:
                sent_score = sum(chunk_probs) / len(chunk_probs)
                scs.append(float(sent_score))
            else:
                scs.append(None)
        else:
            scs.append(None)
        gts.append(sent_gt)
        n += 1
        if sent_gt == 1 and sent_pred == 1: tp += 1
        elif sent_gt == 0 and sent_pred == 0: tn += 1
        elif sent_gt == 0 and sent_pred == 1: fp += 1
        else: fn += 1
        word_idx = end
    return tp, tn, fp, fn, n, gts, scs


def metrics(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    fnr = fn / (fn + tp) if (fn + tp) > 0 else None
    pr = tp / (tp + fp) if (tp + fp) > 0 else None
    rc = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = 2 * pr * rc / (pr + rc) if pr and rc and (pr + rc) > 0 else None
    return acc, fpr, fnr, pr, f1


def safe_auroc(labels, scores):
    if not labels or not scores or len(labels) != len(scores):
        return None
    if len(set(labels)) < 2:
        return None
    # drop None scores
    paired = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not paired or len(set(l for _, l in paired)) < 2:
        return None
    s_arr = [x[0] for x in paired]
    l_arr = [x[1] for x in paired]
    try:
        return float(roc_auc_score(l_arr, s_arr))
    except Exception:
        return None


def aggregate_cell(method: str, level: str, cell_dir: Path, repo_root: Path = Path(__file__).resolve().parent.parent):
    """Process one cell. Returns list of row dicts grouped by (gen, ver, op)."""
    cell_name = cell_dir.name
    domain, split, ablation_label, gen_override = parse_cell(cell_name)
    cfn = CONFUSION_FN[level]

    # SeqXGPT needs sentence-aggregation via CSV lookup
    csv_lookup = None
    if method == 'seqxgpt':
        run_cfg_path = cell_dir / 'run_config.json'
        if run_cfg_path.exists():
            run_cfg = json.loads(run_cfg_path.read_text())
            csv_relpath = run_cfg.get('csv_path', '')
            if csv_relpath:
                csv_lookup = _load_csv_lookup(repo_root, csv_relpath)

    groups = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'n_units': 0,
                                  'labs': [], 'scores': []})
    with open(cell_dir / 'predictions.jsonl') as fd:
        for line in fd:
            rec = json.loads(line)
            # generator: ablation cells have empty ai_model → use cell-name override
            ai_model = rec.get('ai_model', '') or ''
            if gen_override is not None:
                gen = gen_override
            elif ai_model:
                gen = ai_model
            else:
                # ablation cell, derive from name
                gen = parse_cell(cell_name)[3] or ''
            ver = str(rec.get('version', ''))
            op = str(rec.get('operation', ''))
            key = (gen, ver, op)

            if method == 'seqxgpt' and csv_lookup is not None:
                tp, tn, fp, fn, n, labs, scs = confusion_seqxgpt_sentence(rec, csv_lookup, gen_override)
            else:
                tp, tn, fp, fn, n, labs, scs = cfn(method, rec)
            g = groups[key]
            g['tp'] += tp; g['tn'] += tn; g['fp'] += fp; g['fn'] += fn
            g['n_units'] += n
            g['labs'].extend(labs)
            g['scores'].extend(scs)

    rows = []
    for (gen, ver, op), g in groups.items():
        rows.append({
            'method': method, 'level': level,
            'generator': gen, 'domain': domain,
            'version': ver, 'operation': op,
            'ablation': ablation_label,
            'split': split,
            'tp': g['tp'], 'tn': g['tn'], 'fp': g['fp'], 'fn': g['fn'],
            'n_units': g['n_units'],
            '_labs': g['labs'], '_scores': g['scores'],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mirror-dir', default='/datadrive/xiaohan/Omini-Text/results/hf_mirror/tuned_on_new_data')
    p.add_argument('--out-dir', default='/datadrive/xiaohan/Omini-Text/results/aggregates/tuned_on_new_data')
    args = p.parse_args()

    mirror = Path(args.mirror_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Walk all cells across configured methods
    all_rows = []                          # per-slice rows
    cells_seen = defaultdict(set)          # (method, split) → {cell_name}
    for method, hf_path in METHOD_HF_PATH.items():
        level = NATURAL_LEVEL[method]
        method_dir = mirror / hf_path
        if not method_dir.exists():
            print(f'  WARN: {method_dir} not found')
            continue
        for cell_dir in sorted(method_dir.iterdir()):
            if not cell_dir.is_dir(): continue
            preds = cell_dir / 'predictions.jsonl'
            if not preds.exists():
                continue
            try:
                rows = aggregate_cell(method, level, cell_dir)
            except Exception as e:
                print(f'  ERR {method}/{cell_dir.name}: {e}')
                continue
            for r in rows:
                all_rows.append(r)
                cells_seen[(method, r['split'])].add(cell_dir.name)
        print(f'  [{method}] processed {sum(1 for r in all_rows if r["method"]==method)} slices')

    # 2. Compute n_expected per (level, gen, dom, ver, op, ablation): max n_units across methods at same level
    expected = {}
    for r in all_rows:
        k = (r['level'], r['generator'], r['domain'], r['version'], r['operation'], r['ablation'])
        expected[k] = max(expected.get(k, 0), r['n_units'])

    # 3. Annotate rows + compute metrics. Per user "少/缺" rule:
    #    - "少" (method has data, count mismatch slight) → still 'complete', use as-is
    #    - "缺" (method has 0 data for slice) → drop row (never emitted)
    all_rows = [r for r in all_rows if r['n_units'] > 0]
    for r in all_rows:
        k = (r['level'], r['generator'], r['domain'], r['version'], r['operation'], r['ablation'])
        n_exp = expected[k]
        r['n_expected'] = n_exp
        r['n_missing'] = max(0, n_exp - r['n_units'])
        # coverage_status is binary: 'complete' if any data; 'missing' rows are not emitted at all.
        r['coverage_status'] = 'complete'
        acc, fpr, fnr, pr, f1 = metrics(r['tp'], r['tn'], r['fp'], r['fn'])
        r['accuracy'] = acc; r['fpr'] = fpr; r['fnr'] = fnr
        r['precision_ai'] = pr; r['f1_ai'] = f1
        r['auroc'] = safe_auroc(r['_labs'], r['_scores'])
        del r['_labs']; del r['_scores']

    # 4. Write per-split per-level CSVs
    LEVELS = ('document', 'sentence', 'span', 'token')
    OUT_FIELDS = ['method', 'level', 'generator', 'domain', 'version', 'operation', 'ablation',
                  'n_units', 'n_expected', 'n_missing',
                  'tp', 'tn', 'fp', 'fn',
                  'accuracy', 'fpr', 'fnr', 'precision_ai', 'f1_ai', 'auroc',
                  'coverage_status']
    for split in SPLITS:
        for level in LEVELS:
            rows = [r for r in all_rows if r['split'] == split and r['level'] == level]
            if not rows:
                continue
            fname = out_dir / f'{split}_{level}.csv'
            with open(fname, 'w', newline='') as fd:
                w = csv.DictWriter(fd, fieldnames=OUT_FIELDS, extrasaction='ignore')
                w.writeheader()
                rows_sorted = sorted(rows, key=lambda r: (r['method'], r['generator'], r['domain'],
                                                          r['ablation'], r['version'], r['operation']))
                for r in rows_sorted:
                    w.writerow(r)
            print(f'  wrote {fname.name}: {len(rows)} rows')

    # 5. method_summary.csv (per split × level × method)
    summary = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'n_units': 0,
                                   'n_expected': 0, 'cells': set()})
    for r in all_rows:
        k = (r['split'], r['level'], r['method'])
        s = summary[k]
        s['tp'] += r['tp']; s['tn'] += r['tn']; s['fp'] += r['fp']; s['fn'] += r['fn']
        s['n_units'] += r['n_units']
        s['n_expected'] += r['n_expected']

    summary_path = out_dir / 'tuned_on_new_data_method_summary.csv'
    with open(summary_path, 'w', newline='') as fd:
        w = csv.writer(fd)
        w.writerow(['split', 'level', 'method', 'n_units', 'n_expected', 'n_missing',
                    'tp', 'tn', 'fp', 'fn',
                    'accuracy', 'fpr', 'fnr', 'precision_ai', 'f1_ai'])
        for (split, level, method), s in sorted(summary.items()):
            acc, fpr, fnr, pr, f1 = metrics(s['tp'], s['tn'], s['fp'], s['fn'])
            n_miss = max(0, s['n_expected'] - s['n_units'])
            w.writerow([split, level, method, s['n_units'], s['n_expected'], n_miss,
                        s['tp'], s['tn'], s['fp'], s['fn'],
                        acc, fpr, fnr, pr, f1])
    print(f'  wrote {summary_path.name}')

    # 6. coverage_report.csv (per method × split). Status is binary per user's "少/缺" rule:
    #    - 'complete': method has any data for this split (count mismatches are 少, ignore)
    #    - 'missing':  method has zero data for this split (this is 缺)
    coverage_rows = []
    for method in METHOD_HF_PATH:
        level = NATURAL_LEVEL[method]
        for split in SPLITS:
            present_cells = cells_seen.get((method, split), set())
            n_present = len(present_cells)
            n_units_present = sum(r['n_units'] for r in all_rows
                                  if r['method'] == method and r['split'] == split)
            status = 'complete' if n_units_present > 0 else 'missing'
            coverage_rows.append({
                'method': method, 'level': level, 'split': split,
                'n_cells_present': n_present, 'n_units_present': n_units_present,
                'status': status,
            })
    cov_path = out_dir / 'tuned_on_new_data_coverage_report.csv'
    with open(cov_path, 'w', newline='') as fd:
        w = csv.DictWriter(fd, fieldnames=['method', 'level', 'split',
                                            'n_cells_present', 'n_units_present', 'status'])
        w.writeheader()
        for r in sorted(coverage_rows, key=lambda x: (x['method'], x['split'])):
            w.writerow(r)
    print(f'  wrote {cov_path.name}')


if __name__ == '__main__':
    main()
