#!/usr/bin/env python3
"""
backend/analyze_results.py

Standalone, read-only analysis script to compute real measured accuracy,
classification metrics, detection latency, and false-alert rates from the
labeled pipeline log (pipeline_log.csv).

This script strictly reads logs and does not modify app.py, modules/, or any
pipeline logic.

Run with:
  python backend/analyze_results.py [optional/path/to/pipeline_log.csv]

Output is printed to stdout AND written to backend/logs/analysis_report.txt.
"""

import sys
import os
import csv
import io
from collections import Counter

# Try importing scikit-learn for standard metrics if available
try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.metrics import classification_report as sk_classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# ZONE TO PREDICTED STATE MAPPING (EVALUATION A: zone-based)
# ---------------------------------------------------------------------------
# The NeuroPilot SDV pipeline outputs 5 operational zones based on the fused
# Cognitive State Index (CSI 0-100):
#   - "Safe"               (CSI 0-20)  : baseline workload        -> "normal"
#   - "Attention Required" (CSI 21-40) : elevated workload / watch -> "normal"
#   - "Cognitive Overload" (CSI 41-60) : active impairment         -> "at-risk"
#   - "High Risk"          (CSI 61-80) : severe impairment         -> "at-risk"
#   - "Critical Risk"      (CSI 81-100): critical impairment       -> "at-risk"
#   - "No Data (Calibrating)"          : no sensor data            -> "normal"
#
# NOTE: CSI max in this log is 64; "Critical Risk" appears in the mapping for
# correctness but 0 rows were observed with that zone value.
# Any zone value NOT listed here causes an error and non-zero exit.
#
# Ground-truth annotations contain 3 driver states. Rows labeled "unlabeled"
# are transition/calibration periods and are dropped before any metric is computed.
#   - "normal"     -> ground truth "normal"
#   - "drowsy"     -> ground truth "at-risk"
#   - "distracted" -> ground truth "at-risk"
# ---------------------------------------------------------------------------

ZONE_PRED_A = {
    "Safe": "normal",
    "Attention Required": "normal",
    "Cognitive Overload": "at-risk",
    "High Risk": "at-risk",
    "Critical Risk": "at-risk",
    "No Data (Calibrating)": "normal",
}

GROUND_TRUTH_MAPPING = {
    "normal": "normal",
    "drowsy": "at-risk",
    "distracted": "at-risk",
}

# Zone columns for cross-tab display (in severity order, includes Critical Risk)
ZONE_COLUMNS = [
    "Safe",
    "Attention Required",
    "Cognitive Overload",
    "High Risk",
    "Critical Risk",
]

# Zones considered at-risk in latency and coverage calculations
ATRISK_ZONES = {"Cognitive Overload", "High Risk", "Critical Risk"}

# Recovery states that indicate active system alerting (Evaluation B)
ATRISK_RECOVERY_STATES = {"INTERVENING", "WARNING"}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def resolve_log_path(cli_arg=None):
    """Resolve pipeline_log.csv from CLI argument or standard relative paths."""
    if cli_arg and os.path.isfile(cli_arg):
        return os.path.abspath(cli_arg)
    candidates = [
        os.path.join(os.path.dirname(__file__), "logs", "pipeline_log.csv"),
        os.path.join("backend", "logs", "pipeline_log.csv"),
        os.path.join("logs", "pipeline_log.csv"),
        os.path.join(os.getcwd(), "backend", "logs", "pipeline_log.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Could not find pipeline_log.csv. Supply the path as a CLI argument."
    )


def load_and_filter_data(filepath):
    """
    Load CSV, validate all zone and ground_truth_label values, drop 'unlabeled' rows.

    Exits with code 1 if ANY zone value is not in ZONE_PRED_A, or ANY labeled
    ground_truth_label value is not in GROUND_TRUTH_MAPPING.
    """
    raw_rows = []
    filtered_rows = []
    raw_gt_counts = Counter()
    filtered_gt_counts = Counter()
    raw_zone_counts = Counter()
    filtered_zone_counts = Counter()
    filtered_recovery_counts = Counter()
    filtered_trigger_counts = Counter()

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
            gt = row.get("ground_truth_label", "").strip()
            zone = row.get("zone", "").strip()
            raw_gt_counts[gt] += 1
            raw_zone_counts[zone] += 1
            if gt == "unlabeled":
                pass  # counted above, not added to filtered_rows
            else:
                filtered_rows.append(row)
                filtered_gt_counts[gt] += 1
                filtered_zone_counts[zone] += 1
                filtered_recovery_counts[row.get("recovery_state", "").strip()] += 1
                filtered_trigger_counts[row.get("trigger_reason", "").strip()] += 1

    dropped_unlabeled = raw_gt_counts.get("unlabeled", 0)

    # --- Validation: zones (ALL rows) ---
    unmapped_zones = {z: c for z, c in raw_zone_counts.items() if z not in ZONE_PRED_A}
    if unmapped_zones:
        print(
            f"ERROR: {sum(unmapped_zones.values())} row(s) have zone values not in "
            f"ZONE_PRED_A: {unmapped_zones}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Validation: ground-truth labels (labeled rows only) ---
    unmapped_gts = {g: c for g, c in filtered_gt_counts.items() if g not in GROUND_TRUTH_MAPPING}
    if unmapped_gts:
        print(
            f"ERROR: {sum(unmapped_gts.values())} labeled row(s) have ground_truth_label "
            f"not in GROUND_TRUTH_MAPPING: {unmapped_gts}",
            file=sys.stderr,
        )
        sys.exit(1)

    stats = {
        "total_raw_rows": len(raw_rows),
        "dropped_unlabeled": dropped_unlabeled,
        "total_filtered_rows": len(filtered_rows),
        "raw_gt_counts": raw_gt_counts,
        "filtered_gt_counts": filtered_gt_counts,
        "raw_zone_counts": raw_zone_counts,
        "filtered_zone_counts": filtered_zone_counts,
        "filtered_recovery_counts": filtered_recovery_counts,
        "filtered_trigger_counts": filtered_trigger_counts,
    }
    return raw_rows, filtered_rows, stats


def binary_confusion(filtered_rows, pred_fn):
    """
    Generic binary confusion matrix.  pred_fn(row) -> 'normal' | 'at-risk'.
    Returns metrics dict including y_true, y_pred, and detailed_pairs Counter.
    """
    y_true, y_pred = [], []
    detailed_pairs = Counter()
    tp = fp = tn = fn = 0

    for row in filtered_rows:
        raw_gt = row.get("ground_truth_label", "").strip()
        raw_zone = row.get("zone", "").strip()
        mapped_gt = GROUND_TRUTH_MAPPING[raw_gt]
        mapped_pred = pred_fn(row)
        y_true.append(mapped_gt)
        y_pred.append(mapped_pred)
        detailed_pairs[(raw_gt, raw_zone)] += 1

        if mapped_gt == "at-risk" and mapped_pred == "at-risk":
            tp += 1
        elif mapped_gt == "normal" and mapped_pred == "at-risk":
            fp += 1
        elif mapped_gt == "normal" and mapped_pred == "normal":
            tn += 1
        else:
            fn += 1

    total = len(filtered_rows)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    prec_risk = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_risk = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_risk = (
        2 * prec_risk * rec_risk / (prec_risk + rec_risk)
        if (prec_risk + rec_risk) > 0 else 0.0
    )
    prec_norm = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_norm = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_norm = (
        2 * prec_norm * rec_norm / (prec_norm + rec_norm)
        if (prec_norm + rec_norm) > 0 else 0.0
    )
    total_normal_actual = tn + fp
    false_alert_rate = fp / total_normal_actual if total_normal_actual > 0 else 0.0

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "precision_at_risk": prec_risk,
        "recall_at_risk": rec_risk,
        "f1_at_risk": f1_risk,
        "precision_normal": prec_norm,
        "recall_normal": rec_norm,
        "f1_normal": f1_norm,
        "false_alert_rate": false_alert_rate,
        "total_normal_actual": total_normal_actual,
        "total_at_risk_actual": tp + fn,
        "detailed_pairs": detailed_pairs,
    }


def pred_fn_a(row):
    """Evaluation A: zone-based prediction."""
    return ZONE_PRED_A[row.get("zone", "").strip()]


def pred_fn_b(row):
    """
    Evaluation B: intervention-based prediction (defined once, not tuned).

    at-risk if  recovery_state IN {INTERVENING, WARNING}
                OR zone IN {Cognitive Overload, High Risk, Critical Risk}
    otherwise normal.
    """
    rs = row.get("recovery_state", "").strip()
    zone = row.get("zone", "").strip()
    if rs in ATRISK_RECOVERY_STATES or zone in ATRISK_ZONES:
        return "at-risk"
    return "normal"


def compute_segment_analysis(raw_rows, pred_fn):
    """
    Identify contiguous ground-truth segments; compute per at-risk segment:
      - detection latency (first predicted at-risk row after label onset)
      - at-risk row coverage (count + pct)
      - longest consecutive at-risk run (frames)

    Always returns a 2-tuple: (segments, latency_results).
    """
    if not raw_rows:
        return [], []

    # Build segment list
    segments = []
    current_label = None
    seg_start_idx = 0
    seg_start_time = 0.0

    for i, row in enumerate(raw_rows):
        label = row.get("ground_truth_label", "").strip()
        ts = float(row.get("timestamp", 0.0))
        if label != current_label:
            if current_label is not None:
                segments.append({
                    "label": current_label,
                    "start_idx": seg_start_idx,
                    "end_idx": i - 1,
                    "start_time": seg_start_time,
                    "end_time": float(raw_rows[i - 1].get("timestamp", 0.0)),
                    "row_count": i - seg_start_idx,
                })
            current_label = label
            seg_start_idx = i
            seg_start_time = ts

    # Flush last segment
    segments.append({
        "label": current_label,
        "start_idx": seg_start_idx,
        "end_idx": len(raw_rows) - 1,
        "start_time": seg_start_time,
        "end_time": float(raw_rows[-1].get("timestamp", 0.0)),
        "row_count": len(raw_rows) - seg_start_idx,
    })

    # Per at-risk-labeled segment analysis
    latency_results = []
    seg_number = 0
    for seg in segments:
        if seg["label"] not in ("drowsy", "distracted"):
            continue
        seg_number += 1
        onset_time = seg["start_time"]
        rows_in_seg = raw_rows[seg["start_idx"]: seg["end_idx"] + 1]

        detected = False
        detect_time = None
        detect_row_abs = None
        detect_zone_or_rs = None
        latency = None
        frames_to_detect = None

        for rel_idx, r in enumerate(rows_in_seg):
            if pred_fn(r) == "at-risk":
                detected = True
                detect_time = float(r.get("timestamp", 0.0))
                detect_row_abs = seg["start_idx"] + rel_idx
                zone = r.get("zone", "").strip()
                rs = r.get("recovery_state", "").strip()
                detect_zone_or_rs = f"zone={zone}, recovery={rs}"
                latency = detect_time - onset_time
                frames_to_detect = rel_idx
                break

        # Coverage: at-risk rows across full segment
        atrisk_rows = sum(1 for r in rows_in_seg if pred_fn(r) == "at-risk")

        # Longest consecutive at-risk run
        max_run = cur_run = 0
        for r in rows_in_seg:
            if pred_fn(r) == "at-risk":
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0

        latency_results.append({
            "segment_number": seg_number,
            "label": seg["label"],
            "start_idx": seg["start_idx"],
            "csv_line_start": seg["start_idx"] + 2,  # +1 header, +1 1-based
            "end_idx": seg["end_idx"],
            "csv_line_end": seg["end_idx"] + 2,
            "start_time": onset_time,
            "end_time": seg["end_time"],
            "duration_seconds": seg["end_time"] - onset_time,
            "row_count": seg["row_count"],
            "detected": detected,
            "detect_row": detect_row_abs,
            "detect_csv_line": (detect_row_abs + 2) if detect_row_abs is not None else None,
            "detect_time": detect_time,
            "detect_zone_or_rs": detect_zone_or_rs,
            "latency_seconds": latency,
            "frames_to_detect": frames_to_detect,
            "atrisk_rows": atrisk_rows,
            "total_rows": seg["row_count"],
            "atrisk_coverage_pct": atrisk_rows / seg["row_count"] * 100.0 if seg["row_count"] > 0 else 0.0,
            "longest_consecutive_atrisk": max_run,
        })

    return segments, latency_results


# ---------------------------------------------------------------------------
# PRINT HELPERS
# ---------------------------------------------------------------------------

def _section(out, title):
    out.write("-" * 80 + "\n")
    out.write(title + "\n")
    out.write("-" * 80 + "\n")


def _print_eval_block(out, metrics, latency_results, stats, filtered_rows, is_B=False):
    """Print shared confusion matrix, metrics, per-segment latency block."""
    m = metrics

    # Confusion matrix
    out.write("Confusion Matrix (rows=actual, cols=predicted):\n")
    out.write(f"                       Predicted 'normal'    Predicted 'at-risk'    Total Actual\n")
    out.write(f"Actual 'normal'  :           {m['tn']:5d} (TN)            {m['fp']:5d} (FP)         {m['total_normal_actual']:5d}\n")
    out.write(f"Actual 'at-risk' :           {m['fn']:5d} (FN)           {m['tp']:5d} (TP)         {m['total_at_risk_actual']:5d}\n")
    out.write(f"Total Predicted  :           {m['tn']+m['fn']:5d}                 {m['fp']+m['tp']:5d}                {m['total']:5d}\n")
    out.write(f"\n  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}\n\n")

    if SKLEARN_AVAILABLE:
        out.write("[sklearn.metrics.confusion_matrix]\n")
        cm = sk_confusion_matrix(m["y_true"], m["y_pred"], labels=["normal", "at-risk"])
        out.write(f"Labels: ['normal', 'at-risk']\n{cm}\n\n")
        out.write("[sklearn.metrics.classification_report]\n")
        cr = sk_classification_report(m["y_true"], m["y_pred"], labels=["normal", "at-risk"], digits=4)
        out.write(cr + "\n")
    else:
        out.write("(scikit-learn not available -- hand-computed metrics below)\n")
        out.write(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}\n")
        out.write(f"  {'normal':<10} {m['precision_normal']:>10.4f} {m['recall_normal']:>10.4f} {m['f1_normal']:>10.4f} {m['total_normal_actual']:>8}\n")
        out.write(f"  {'at-risk':<10} {m['precision_at_risk']:>10.4f} {m['recall_at_risk']:>10.4f} {m['f1_at_risk']:>10.4f} {m['total_at_risk_actual']:>8}\n")
        out.write(f"  Accuracy: {m['accuracy']:.4f}  support={m['total']}\n\n")

    out.write(f"Overall Accuracy      : {m['accuracy']*100:.2f}%  ({m['tp']+m['tn']} / {m['total']} rows)\n")
    out.write(f"Precision (at-risk)   : {m['precision_at_risk']*100:.2f}%  ({m['tp']} / {m['tp']+m['fp']} predicted at-risk)\n")
    out.write(f"Recall    (at-risk)   : {m['recall_at_risk']*100:.2f}%  ({m['tp']} / {m['total_at_risk_actual']} actual at-risk)\n")
    out.write(f"F1-Score  (at-risk)   : {m['f1_at_risk']*100:.2f}%\n")
    out.write(f"False-Alert Rate      : {m['false_alert_rate']*100:.2f}%  ({m['fp']} / {m['total_normal_actual']} normal rows)\n")
    out.write(f"Specificity (TNR)     : {(1.0-m['false_alert_rate'])*100:.2f}%\n")
    out.write("\n")

    # Per-impairment-type breakdown
    if is_B:
        # Eval B: count rows where pred_fn_b fires (zone OR recovery_state)
        for gt_lbl in ("drowsy", "distracted"):
            total_lbl = stats["filtered_gt_counts"].get(gt_lbl, 0)
            tp_lbl = sum(
                1 for row in filtered_rows
                if row.get("ground_truth_label", "").strip() == gt_lbl
                and pred_fn_b(row) == "at-risk"
            )
            not_detected = total_lbl - tp_lbl
            pct = tp_lbl / total_lbl * 100.0 if total_lbl > 0 else 0.0
            out.write(
                f"  '{gt_lbl}' Eval-B per-class coverage: {tp_lbl} / {total_lbl} frames ({pct:.2f}%)"
                f"  |  not detected by Eval B: {not_detected} frames\n"
            )
    else:
        # Eval A: zone-based counts (uses detailed_pairs which is keyed by zone)
        dp = m["detailed_pairs"]
        for gt_lbl in ("drowsy", "distracted"):
            tp_lbl = sum(dp.get((gt_lbl, z), 0) for z in ATRISK_ZONES)
            total_lbl = stats["filtered_gt_counts"].get(gt_lbl, 0)
            missed = sum(dp.get((gt_lbl, z), 0) for z in ["Safe", "Attention Required"])
            pct = tp_lbl / total_lbl * 100.0 if total_lbl > 0 else 0.0
            out.write(
                f"  '{gt_lbl}' zone-based detection rate: {tp_lbl} / {total_lbl} frames ({pct:.2f}%)"
                f"  |  stayed in lower zone: {missed} frames\n"
            )

    # Eval-B specific: trigger_reason on TP rows
    if is_B:
        out.write("\n  Trigger-reason distribution on True-Positive rows (Eval B only):\n")
        tp_tr_counts = Counter()
        for row in filtered_rows:
            raw_gt = row.get("ground_truth_label", "").strip()
            if GROUND_TRUTH_MAPPING.get(raw_gt) == "at-risk" and pred_fn_b(row) == "at-risk":
                tr = row.get("trigger_reason", "").strip()
                tp_tr_counts[tr if tr else "(empty)"] += 1
        for tr_val, cnt in sorted(tp_tr_counts.items(), key=lambda x: -x[1]):
            out.write(f"    trigger_reason='{tr_val}': {cnt} rows\n")
        out.write(
            "  NOTE: The KINEMATIC channel uses SIMULATED data in this log.\n"
            "        trigger_reason='KINEMATIC' rows do not reflect real vehicle sensor events.\n"
        )

    out.write("\n")
    out.write("Detection Latency per at-risk ground-truth segment:\n")
    valid_latencies = []
    for r in latency_results:
        out.write(f"  Segment #{r['segment_number']}: Ground Truth '{r['label'].upper()}'\n")
        out.write(f"    Onset       : {r['start_time']:.6f}  (CSV line {r['csv_line_start']})\n")
        out.write(f"    End         : {r['end_time']:.6f}  (CSV line {r['csv_line_end']})\n")
        out.write(f"    Duration    : {r['duration_seconds']:.2f} s  ({r['row_count']} frames)\n")
        out.write(
            f"    At-risk cov : {r['atrisk_rows']} / {r['total_rows']} frames "
            f"({r['atrisk_coverage_pct']:.2f}%)  |  "
            f"Longest consecutive at-risk run: {r['longest_consecutive_atrisk']} frames\n"
        )
        if r["detected"]:
            out.write(
                f"    First alert : {r['detect_time']:.6f}  (CSV line {r['detect_csv_line']}) "
                f" [{r['detect_zone_or_rs']}]\n"
            )
            out.write(
                f"    Latency     : {r['latency_seconds']:.2f} s  "
                f"({r['latency_seconds']:.4f}s exact, {r['frames_to_detect']} frames)\n"
            )
            valid_latencies.append(r["latency_seconds"])
        else:
            out.write("    First alert : NOT TRIGGERED during this segment\n")
        out.write("\n")

    if valid_latencies:
        avg = sum(valid_latencies) / len(valid_latencies)
        out.write(
            f"Average Detection Latency ({len(valid_latencies)} segment(s) detected): "
            f"{avg:.2f} s  ({avg:.4f}s exact)\n"
        )
    else:
        out.write("Average Detection Latency: N/A (no segments triggered)\n")
    out.write("\n")


# ---------------------------------------------------------------------------
# MAIN REPORT BUILDER
# ---------------------------------------------------------------------------

def build_report(filepath, raw_rows, filtered_rows, stats,
                 metrics_a, lat_results_a, metrics_b, lat_results_b):
    """Assemble and return the full report as a string."""
    out = io.StringIO()

    def p(s=""):
        out.write(str(s) + "\n")

    p("=" * 80)
    p("NEUROPILOT SDV -- PIPELINE ACCURACY & LATENCY BENCHMARK REPORT")
    p("=" * 80)
    p(f"Log File  : {filepath}")
    p(f"Generated : {__import__('datetime').datetime.now().isoformat(timespec='seconds')}")
    p()

    # ------------------------------------------------------------------
    _section(out, "1. DATASET FILTERING & ROW COUNTS (MEASURED)")
    p(f"Total raw rows in pipeline_log.csv : {stats['total_raw_rows']}")
    p(f"Dropped transition rows (unlabeled): {stats['dropped_unlabeled']}")
    p(f"Total valid ground-truth rows      : {stats['total_filtered_rows']}")
    p()
    p("Rows per individual ground-truth class (labeled rows only):")
    for lbl in ["normal", "drowsy", "distracted"]:
        cnt = stats["filtered_gt_counts"].get(lbl, 0)
        pct = cnt / stats["total_filtered_rows"] * 100.0 if stats["total_filtered_rows"] > 0 else 0.0
        p(f"  '{lbl:<10}' : {cnt:5d} rows ({pct:5.2f}%)")
    p()
    normal_gt = stats["filtered_gt_counts"].get("normal", 0)
    at_risk_gt = (stats["filtered_gt_counts"].get("drowsy", 0)
                  + stats["filtered_gt_counts"].get("distracted", 0))
    p("Grouped binary ground-truth classes:")
    p(f"  'normal'  : {normal_gt:5d} rows ({normal_gt/stats['total_filtered_rows']*100:.2f}%)")
    p(f"  'at-risk' : {at_risk_gt:5d} rows ({at_risk_gt/stats['total_filtered_rows']*100:.2f}%)  (drowsy + distracted)")
    p(f"  Total     : {normal_gt+at_risk_gt:5d} rows")
    p()

    # ------------------------------------------------------------------
    _section(out, "2. CROSS-TABULATION: ground_truth_label vs recovery_state (LABELED ROWS, MEASURED)")
    recovery_states = sorted(stats["filtered_recovery_counts"].keys())
    gt_rs_counter = Counter(
        (row["ground_truth_label"].strip(), row["recovery_state"].strip())
        for row in filtered_rows
    )
    col_w = 14
    header_cells = ["ground_truth"] + recovery_states + ["Total"]
    p("  " + " | ".join(f"{c:<{col_w}}" for c in header_cells))
    p("  " + "-" * (col_w * len(header_cells) + 3 * (len(header_cells) - 1)))
    for gt_lbl in ["normal", "drowsy", "distracted"]:
        cells = [gt_lbl]
        row_total = 0
        for rs in recovery_states:
            cnt = gt_rs_counter.get((gt_lbl, rs), 0)
            cells.append(str(cnt))
            row_total += cnt
        cells.append(str(row_total))
        p("  " + " | ".join(f"{c:<{col_w}}" for c in cells))
    cells = ["Total"]
    grand = 0
    for rs in recovery_states:
        col_sum = sum(gt_rs_counter.get((g, rs), 0) for g in ["normal", "drowsy", "distracted"])
        cells.append(str(col_sum))
        grand += col_sum
    cells.append(str(grand))
    p("  " + " | ".join(f"{c:<{col_w}}" for c in cells))
    p()

    # ------------------------------------------------------------------
    _section(out, "3. CROSS-TABULATION: ground_truth_label vs trigger_reason (LABELED ROWS, MEASURED)")
    trigger_reasons = sorted(stats["filtered_trigger_counts"].keys())
    trigger_display = [(t if t else "(empty)") for t in trigger_reasons]
    gt_tr_counter = Counter(
        (row["ground_truth_label"].strip(), row["trigger_reason"].strip())
        for row in filtered_rows
    )
    col_w2 = 14
    header_cells2 = ["ground_truth"] + trigger_display + ["Total"]
    p("  " + " | ".join(f"{c:<{col_w2}}" for c in header_cells2))
    p("  " + "-" * (col_w2 * len(header_cells2) + 3 * (len(header_cells2) - 1)))
    for gt_lbl in ["normal", "drowsy", "distracted"]:
        cells = [gt_lbl]
        row_total = 0
        for tr in trigger_reasons:
            cnt = gt_tr_counter.get((gt_lbl, tr), 0)
            cells.append(str(cnt))
            row_total += cnt
        cells.append(str(row_total))
        p("  " + " | ".join(f"{c:<{col_w2}}" for c in cells))
    cells = ["Total"]
    grand = 0
    for tr in trigger_reasons:
        col_sum = sum(gt_tr_counter.get((g, tr), 0) for g in ["normal", "drowsy", "distracted"])
        cells.append(str(col_sum))
        grand += col_sum
    cells.append(str(grand))
    p("  " + " | ".join(f"{c:<{col_w2}}" for c in cells))
    p()

    # ------------------------------------------------------------------
    _section(out, "4. CROSS-TABULATION: ground_truth_label vs ZONE (LABELED ROWS, incl. Critical Risk)")
    # Include all zone columns even if count=0 to satisfy the fix requirement
    col_wz = 16
    header_z = ["ground_truth"] + [z[:14] for z in ZONE_COLUMNS] + ["Total"]
    p("  " + " | ".join(f"{c:<{col_wz}}" for c in header_z))
    p("  " + "-" * (col_wz * len(header_z) + 3 * (len(header_z) - 1)))
    for gt_lbl in ["normal", "drowsy", "distracted"]:
        cells = [gt_lbl]
        row_total = 0
        for z in ZONE_COLUMNS:
            cnt = metrics_a["detailed_pairs"].get((gt_lbl, z), 0)
            cells.append(str(cnt))
            row_total += cnt
        cells.append(str(row_total))
        p("  " + " | ".join(f"{c:<{col_wz}}" for c in cells))
    cells = ["Total"]
    grand = 0
    for z in ZONE_COLUMNS:
        col_sum = stats["filtered_zone_counts"].get(z, 0)
        cells.append(str(col_sum))
        grand += col_sum
    cells.append(str(grand))
    p("  " + " | ".join(f"{c:<{col_wz}}" for c in cells))
    p()
    p("Note: 'Critical Risk' column included for completeness; 0 rows observed in this log")
    p("      (max CSI = 64; Critical Risk would require CSI > 80).")
    p()

    # ------------------------------------------------------------------
    _section(out, "5. PREDICTION MAPPING SPECIFICATION (BOTH EVALUATIONS)")
    p("EVALUATION A (zone-based) -- zone -> predicted state:")
    for z, pred in ZONE_PRED_A.items():
        cnt = stats["filtered_zone_counts"].get(z, 0)
        obs = f"(labeled rows: {cnt})" if cnt else "(0 labeled rows in this log)"
        p(f"  '{z:<22}' -> '{pred}'  {obs}")
    p()
    p("EVALUATION B (intervention-based, defined once, not tuned):")
    p("  predicted 'at-risk' if   recovery_state IN {INTERVENING, WARNING}")
    p("                        OR zone IN {Cognitive Overload, High Risk, Critical Risk}")
    p("  predicted 'normal'  otherwise")
    p()
    warning_count = sum(1 for r in raw_rows if r.get("recovery_state", "").strip() == "WARNING")
    p(f"  'WARNING' recovery_state rows in log: {warning_count}  (none observed; mapping defined for correctness)")
    p()
    p("Ground truth grouping (both evaluations):")
    for gt, mapped in GROUND_TRUTH_MAPPING.items():
        cnt = stats["filtered_gt_counts"].get(gt, 0)
        p(f"  '{gt:<10}' -> '{mapped}'  ({cnt} rows)")
    p()

    # ------------------------------------------------------------------
    _section(out, "6. EVALUATION A: zone-based (MEASURED)")
    _print_eval_block(out, metrics_a, lat_results_a, stats, filtered_rows, is_B=False)

    # ------------------------------------------------------------------
    _section(out, "7. EVALUATION B: intervention-based (MEASURED)")
    _print_eval_block(out, metrics_b, lat_results_b, stats, filtered_rows, is_B=True)

    # ------------------------------------------------------------------
    _section(out, "8. SIDE-BY-SIDE COMPARISON: EVALUATION A vs EVALUATION B")
    p(f"{'Metric':<38} {'Eval A (zone-based)':<22} {'Eval B (intervention)':<22}")
    p("-" * 84)

    def _pct(v): return f"{v*100:.2f}%"
    def _avg_lat(lrs):
        lats = [r["latency_seconds"] for r in lrs if r["latency_seconds"] is not None]
        return f"{sum(lats)/len(lats):.4f}s (n={len(lats)})" if lats else "N/A"

    side_rows = [
        ("Accuracy",              _pct(metrics_a["accuracy"]),             _pct(metrics_b["accuracy"])),
        ("Precision (at-risk)",   _pct(metrics_a["precision_at_risk"]),    _pct(metrics_b["precision_at_risk"])),
        ("Recall    (at-risk)",   _pct(metrics_a["recall_at_risk"]),       _pct(metrics_b["recall_at_risk"])),
        ("F1-Score  (at-risk)",   _pct(metrics_a["f1_at_risk"]),           _pct(metrics_b["f1_at_risk"])),
        ("False-Alert Rate",      _pct(metrics_a["false_alert_rate"]),     _pct(metrics_b["false_alert_rate"])),
        ("Specificity (TNR)",     _pct(1-metrics_a["false_alert_rate"]),   _pct(1-metrics_b["false_alert_rate"])),
        ("TP / FN / FP / TN",
         f"{metrics_a['tp']}/{metrics_a['fn']}/{metrics_a['fp']}/{metrics_a['tn']}",
         f"{metrics_b['tp']}/{metrics_b['fn']}/{metrics_b['fp']}/{metrics_b['tn']}"),
        ("Avg Detection Latency", _avg_lat(lat_results_a),                _avg_lat(lat_results_b)),
    ]

    for seg_idx in range(max(len(lat_results_a), len(lat_results_b))):
        lr_a = lat_results_a[seg_idx] if seg_idx < len(lat_results_a) else None
        lr_b = lat_results_b[seg_idx] if seg_idx < len(lat_results_b) else None
        lbl = (lr_a or lr_b)["label"].upper()
        lat_a = f"{lr_a['latency_seconds']:.4f}s" if lr_a and lr_a["latency_seconds"] is not None else "N/A"
        lat_b = f"{lr_b['latency_seconds']:.4f}s" if lr_b and lr_b["latency_seconds"] is not None else "N/A"
        cov_a = (f"{lr_a['atrisk_coverage_pct']:.2f}% ({lr_a['atrisk_rows']}/{lr_a['total_rows']})"
                 if lr_a else "N/A")
        cov_b = (f"{lr_b['atrisk_coverage_pct']:.2f}% ({lr_b['atrisk_rows']}/{lr_b['total_rows']})"
                 if lr_b else "N/A")
        run_a = str(lr_a["longest_consecutive_atrisk"]) if lr_a else "N/A"
        run_b = str(lr_b["longest_consecutive_atrisk"]) if lr_b else "N/A"
        side_rows.append((f"  Latency  seg#{seg_idx+1} ({lbl})", lat_a, lat_b))
        side_rows.append((f"  Coverage seg#{seg_idx+1} ({lbl})", cov_a, cov_b))
        side_rows.append((f"  Max run  seg#{seg_idx+1} ({lbl})", run_a, run_b))

    for metric_name, val_a, val_b in side_rows:
        p(f"{metric_name:<38} {val_a:<22} {val_b:<22}")
    p()
    p("Neither evaluation is selected as 'better'. Both are reported as-is.")
    p("Eval A uses only the zone field. Eval B additionally folds in recovery_state,")
    p("which persists (INTERVENING hysteresis) even after the zone drops below threshold.")
    p("This means Eval B's higher recall partly reflects system state persistence, not faster")
    p("detection. Eval A is the more conservative / ground-truth-honest measurement.")
    p()

    # ------------------------------------------------------------------
    _section(out, "9. SENSITIVITY ANALYSIS  [SENSITIVITY, NOT A RESULT]")
    p("Sensitivity test: what if 'Attention Required' also counts as at-risk (Eval A)?")
    p("This widens the detection boundary. Shown only to illustrate the recall vs")
    p("false-alert tradeoff. DO NOT cite these numbers as system accuracy.")
    p()

    ZONE_PRED_SENSITIVE = dict(ZONE_PRED_A)
    ZONE_PRED_SENSITIVE["Attention Required"] = "at-risk"

    def pred_fn_sensitive(row):
        return ZONE_PRED_SENSITIVE[row.get("zone", "").strip()]

    m_sens = binary_confusion(filtered_rows, pred_fn_sensitive)
    p(f"  {'Metric':<30} {'Standard Eval A':<22} {'Sensitivity (AttnReq=at-risk)'}")
    p("  " + "-" * 75)
    sens_comparisons = [
        ("Recall (at-risk)",     _pct(metrics_a["recall_at_risk"]),       _pct(m_sens["recall_at_risk"])),
        ("False-Alert Rate",     _pct(metrics_a["false_alert_rate"]),      _pct(m_sens["false_alert_rate"])),
        ("Precision (at-risk)",  _pct(metrics_a["precision_at_risk"]),     _pct(m_sens["precision_at_risk"])),
        ("F1-Score (at-risk)",   _pct(metrics_a["f1_at_risk"]),            _pct(m_sens["f1_at_risk"])),
        ("Accuracy",             _pct(metrics_a["accuracy"]),              _pct(m_sens["accuracy"])),
        ("TP / FN / FP / TN",
         f"{metrics_a['tp']}/{metrics_a['fn']}/{metrics_a['fp']}/{metrics_a['tn']}",
         f"{m_sens['tp']}/{m_sens['fn']}/{m_sens['fp']}/{m_sens['tn']}"),
    ]
    for name, va, vs in sens_comparisons:
        p(f"  {name:<30} {va:<22} {vs}")
    p()
    p("  Widening the boundary yields 100% recall but also flags most normal frames as at-risk.")
    p()

    # ------------------------------------------------------------------
    _section(out, "10. LIMITATIONS  [READ BEFORE CITING ANY NUMBER]")
    p("1. Single session, single subject.")
    p("   All metrics are from one ~71-second session with one participant. Results")
    p("   cannot be generalized to a population or cited as system accuracy.")
    p()
    p("2. Autocorrelation / effective sample size.")
    est_fps = stats["total_raw_rows"] / 71.0
    p(f"   Rows are logged at ~{est_fps:.1f} fps. Consecutive frames are highly autocorrelated.")
    p("   The effective statistical n is the number of labeled segments: 4")
    p("   (2 normal segments, 1 drowsy segment, 1 distracted segment), NOT the 782 labeled rows.")
    p("   No valid confidence intervals can be computed from this data alone.")
    p()
    p("3. Labeling delay, direction unknown.")
    p("   Latency is measured from the first row carrying the new ground-truth label, set by a")
    p("   human keypress. The offset between keypress and true onset was NOT measured and its sign")
    p("   is unknown (key before behavior overstates latency; key after understates it).")
    p("   Treat latency as indicative only; it is NOT an upper bound.")
    p()
    p("4. Average latency is over n=2 segments.")
    p("   The average detection latency is the mean of exactly 2 measurements (one drowsy,")
    p("   one distracted). It is not a statistically stable estimate.")
    p()
    p("5. Kinematic channel is simulated.")
    p("   The kinematic (steering/accelerometer) modality uses synthetic data in this log.")
    p("   trigger_reason='KINEMATIC' rows do not reflect real vehicle sensor events.")
    p()
    p("6. No labeled row reached High Risk or Critical Risk (max CSI over the whole")
    p("   log = 64); upper zones were not exercised.")
    p()
    p("7. Recovery-state hysteresis inflates Evaluation B recall.")
    p("   INTERVENING persists after the zone drops back below threshold. This means Eval B")
    p("   correctly labels frames that Eval A misses, but this is partially due to system state")
    p("   memory rather than the signal being truly at-risk at that moment.")
    p()
    p("8. Evaluation B was defined after Evaluation A's results were seen")
    p("   (post hoc, exploratory), not pre-registered.")
    p()
    p("9. Eval A and Eval B latencies are identical because the first at-risk")
    p("   row in both is the same zone crossing; B's latency is not independent evidence.")
    p()

    # ------------------------------------------------------------------
    _section(out, "11. UNCOMPUTABLE METRICS & LOG GAPS")
    p("Metrics that CANNOT be computed from this log:")
    p("  1. Multi-class confusion (drowsy vs distracted separately) -- the pipeline")
    p("     produces a single CSI severity score, not independent classifiers per impairment type.")
    p("  2. Driver recovery time -- the session ends in an 'unlabeled' period; no labeled")
    p("     return-to-normal segment follows the distracted block.")
    p("  3. Population accuracy or confidence intervals -- single-subject data only.")
    p("=" * 80)

    return out.getvalue()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    filepath = resolve_log_path(cli_arg)

    raw_rows, filtered_rows, stats = load_and_filter_data(filepath)

    # Evaluation A: zone-based
    metrics_a = binary_confusion(filtered_rows, pred_fn_a)
    _, lat_results_a = compute_segment_analysis(raw_rows, pred_fn_a)

    # Evaluation B: intervention-based
    metrics_b = binary_confusion(filtered_rows, pred_fn_b)
    _, lat_results_b = compute_segment_analysis(raw_rows, pred_fn_b)

    report = build_report(
        filepath, raw_rows, filtered_rows, stats,
        metrics_a, lat_results_a,
        metrics_b, lat_results_b,
    )

    # Print to stdout
    print(report, end="")

    # Write to file alongside pipeline_log.csv
    log_dir = os.path.dirname(filepath)
    report_path = os.path.join(log_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report also written to: {report_path}]")


if __name__ == "__main__":
    main()
