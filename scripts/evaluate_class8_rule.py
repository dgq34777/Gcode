from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from evaluate_class7_rule import predict_class7
from evaluate_rules import GCodePoint, feature_table, parse_points


VARIANTS = ("reference", "balanced", "high_precision", "scanline_dedup")


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


def group_by_scan_coordinate(items: list[tuple[float, int]], tolerance: float) -> list[list[tuple[float, int]]]:
    groups: list[list[tuple[float, int]]] = []
    current: list[tuple[float, int]] = []

    for item in sorted(items):
        if not current or abs(item[0] - current[-1][0]) <= tolerance:
            current.append(item)
            continue

        if len(current) > 1:
            groups.append(current)
        current = [item]

    if len(current) > 1:
        groups.append(current)
    return groups


def duplicate_event_score(feature: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        feature.get("step_ratio", 0.0),
        feature.get("max_step", 0.0),
        feature.get("turn1", 0.0),
        feature.get("d2", 0.0),
    )


def nearest_neighbor_transition_score(
    scan_coordinate: float,
    line_coordinates: list[list[float]],
    line_index: int,
    max_gap: int = 6,
) -> float | None:
    distances: list[float] = []
    for direction in (-1, 1):
        neighbor_index = line_index + direction
        skipped = 0
        while 0 <= neighbor_index < len(line_coordinates) and skipped <= max_gap:
            if line_coordinates[neighbor_index]:
                distances.append(
                    min(abs(scan_coordinate - coordinate) for coordinate in line_coordinates[neighbor_index])
                )
                break
            neighbor_index += direction
            skipped += 1

    if not distances:
        return None
    return sum(distances) / len(distances)


def deduplicate_scanline_events(
    points: list[GCodePoint],
    features: list[dict[str, float]],
    predictions: list[bool],
    class7_predictions: list[bool],
    scan_axis: int,
    duplicate_tolerance: float = 0.01,
    row_event_tolerance: float = 0.08,
    row_margin: float = 0.005,
    row_best_max: float = 0.08,
) -> list[bool]:
    output = predictions[:]
    class7_indices = [index for index, prediction in enumerate(class7_predictions) if prediction]

    for start, end in zip(class7_indices, class7_indices[1:]):
        candidates = [
            (float(points[index].xyz[scan_axis]), index)
            for index in range(start + 1, end)
            if output[index]
        ]
        for group in group_by_scan_coordinate(candidates, duplicate_tolerance):
            best_index = max(group, key=lambda item: duplicate_event_score(features[item[1]]))[1]
            for _, index in group:
                if index != best_index:
                    output[index] = False

    line_coordinates: list[list[float]] = []
    for start, end in zip(class7_indices, class7_indices[1:]):
        line_coordinates.append(
            [
                float(points[index].xyz[scan_axis])
                for index in range(start + 1, end)
                if output[index]
            ]
        )

    for line_index, (start, end) in enumerate(zip(class7_indices, class7_indices[1:])):
        candidates = [
            (float(points[index].xyz[scan_axis]), index)
            for index in range(start + 1, end)
            if output[index]
        ]
        for group in group_by_scan_coordinate(candidates, row_event_tolerance):
            scored: list[tuple[float, float, int]] = []
            for scan_coordinate, index in group:
                score = nearest_neighbor_transition_score(scan_coordinate, line_coordinates, line_index)
                if score is not None:
                    scored.append((score, scan_coordinate, index))

            if len(scored) != len(group) or len(scored) < 2:
                continue

            scored.sort()
            if scored[0][0] <= row_best_max and scored[1][0] - scored[0][0] >= row_margin:
                best_index = scored[0][2]
                for _, _, index in scored[1:]:
                    if index != best_index:
                        output[index] = False

    return output


def predict_class8(points: list[GCodePoint], variant: str) -> list[bool]:
    features = feature_table(points)
    class7_predictions, scan_axis, _, _ = predict_class7(points)
    if variant == "scanline_dedup":
        base_variant = "reference"
    else:
        base_variant = variant

    predictions = [
        is_class8_candidate(feature, is_class7, base_variant)
        for feature, is_class7 in zip(features, class7_predictions)
    ]
    if variant != "scanline_dedup":
        return predictions

    return deduplicate_scanline_events(points, features, predictions, class7_predictions, scan_axis)


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
