from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from evaluate_class7_rule import predict_class7
from evaluate_rules import GCodePoint, feature_table, parse_points


VARIANTS = ("reference", "balanced", "high_precision")


def is_class8_candidate(feature: dict[str, float], is_class7: bool, variant: str) -> bool:
    if not feature.get("valid") or is_class7 or feature["turn1"] > 105:
        return False

    if variant == "reference":
        return (
            feature["turn1"] >= 10
            and (feature["j2"] >= 30 or feature["step_ratio"] >= 8)
            and feature["d2"] >= 0.2
        )

    if variant == "balanced":
        slope_step = (
            feature["turn1"] >= 10
            and (feature["j2"] >= 40 or feature["step_ratio"] >= 4)
            and feature["d2"] >= 0.15
        )
        tiny_transition = (
            feature["min_step"] <= 0.01
            and feature["step_ratio"] >= 100
            and feature["max_step"] >= 5
            and feature["j2"] >= 40
            and feature["d2"] >= 0.15
        )
        return slope_step or tiny_transition

    if variant == "high_precision":
        return (
            feature["turn1"] >= 20
            and (feature["j2"] >= 40 or feature["step_ratio"] >= 4)
            and feature["d2"] >= 0.2
        )

    raise ValueError(f"Unknown variant: {variant}")


def predict_class8(points: list[GCodePoint], variant: str) -> list[bool]:
    features = feature_table(points)
    class7_predictions, _, _, _ = predict_class7(points)
    return [
        is_class8_candidate(feature, is_class7, variant)
        for feature, is_class7 in zip(features, class7_predictions)
    ]


def score(points: list[GCodePoint], predictions: list[bool]) -> tuple[int, int, int, int, float, float, float]:
    truth = [point.label == 8 for point in points]
    tp = sum(t and p for t, p in zip(truth, predictions))
    fp = sum((not t) and p for t, p in zip(truth, predictions))
    fn = sum(t and (not p) for t, p in zip(truth, predictions))
    tn = sum((not t) and (not p) for t, p in zip(truth, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def evaluate_variant(data_dir: Path, variant: str, show_errors: int) -> tuple[int, int, int, int, Counter[int]]:
    totals = [0, 0, 0, 0]
    false_positive_labels: Counter[int] = Counter()

    print(f"Variant: {variant}")
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() != ".mpf":
            continue

        points = parse_points(path)
        predictions = predict_class8(points, variant)
        tp, fp, fn, tn, precision, recall, f1 = score(points, predictions)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        totals[3] += tn

        print(
            f"{path.name}\ttrue8={tp + fn}\tpred8={tp + fp}"
            f"\ttp={tp}\tfp={fp}\tfn={fn}"
            f"\tprecision={precision:.4f}\trecall={recall:.4f}\tf1={f1:.4f}"
        )

        shown = 0
        for point, prediction in zip(points, predictions):
            is_true = point.label == 8
            if prediction and not is_true:
                false_positive_labels[point.label] += 1
            if show_errors <= 0 or is_true == prediction or shown >= show_errors:
                continue
            kind = "FP" if prediction else "FN"
            print(f"  {kind} {point.line_no}: label={point.label} {point.raw}")
            shown += 1

    tp, fp, fn, tn = totals
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / sum(totals) if sum(totals) else 0.0
    print(
        f"TOTAL\ttp={tp}\tfp={fp}\tfn={fn}"
        f"\tprecision={precision:.4f}\trecall={recall:.4f}\tf1={f1:.4f}\taccuracy={accuracy:.4f}"
    )
    print(f"False-positive labels: {dict(sorted(false_positive_labels.items()))}")
    return tp, fp, fn, tn, false_positive_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate class-8 slope-discontinuity boundary rules.")
    parser.add_argument("--data-dir", type=Path, default=Path("Gcode"))
    parser.add_argument("--variant", choices=VARIANTS, default="reference")
    parser.add_argument("--compare", action="store_true", help="Evaluate all rule variants.")
    parser.add_argument("--errors", type=int, default=0, help="Show up to N errors per file.")
    args = parser.parse_args()

    variants = VARIANTS if args.compare else (args.variant,)
    for index, variant in enumerate(variants):
        if index:
            print()
        evaluate_variant(args.data_dir, variant, args.errors)


if __name__ == "__main__":
    main()
