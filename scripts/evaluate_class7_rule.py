from __future__ import annotations

from collections import Counter
from pathlib import Path

from evaluate_rules import GCodePoint, parse_points


def infer_scan_axis_and_bounds(points: list[GCodePoint]) -> tuple[int, float, float]:
    best: tuple[float, int, float, float] | None = None
    for axis in (0, 1):
        values = [round(float(point.xyz[axis]), 3) for point in points]
        counts = Counter(values)
        unique_values = sorted(counts)
        value_range = unique_values[-1] - unique_values[0]
        if value_range == 0:
            continue

        top_values = counts.most_common(30)
        for left_index, (a_value, a_count) in enumerate(top_values):
            for b_value, b_count in top_values[left_index + 1 :]:
                separation = abs(a_value - b_value)
                score = (a_count + b_count) * separation / value_range
                if best is None or score > best[0]:
                    best = (score, axis, min(a_value, b_value), max(a_value, b_value))

    if best is None:
        raise ValueError("Could not infer scan boundaries")
    _, axis, low, high = best
    return axis, low, high


def boundary_side(point: GCodePoint, axis: int, low: float, high: float, tol: float) -> str | None:
    value = float(point.xyz[axis])
    if abs(value - low) <= tol:
        return "low"
    if abs(value - high) <= tol:
        return "high"
    return None


def leaves_boundary_from_side(point: GCodePoint, axis: int, low: float, high: float, side: str, tol: float) -> bool:
    value = float(point.xyz[axis])
    if side == "low":
        return value > low + tol
    return value < high - tol


def predict_class7(points: list[GCodePoint], tol: float = 0.02) -> tuple[list[bool], int, float, float]:
    axis, low, high = infer_scan_axis_and_bounds(points)
    predictions = [False for _ in points]

    for index, point in enumerate(points):
        side = boundary_side(point, axis, low, high, tol)
        if side is None:
            continue

        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(points):
                continue

            neighbor = points[neighbor_index]
            neighbor_side = boundary_side(neighbor, axis, low, high, tol)
            goes_inside_or_to_other_side = leaves_boundary_from_side(neighbor, axis, low, high, side, tol)

            if goes_inside_or_to_other_side or (neighbor_side is not None and neighbor_side != side):
                predictions[index] = True
                break

    return predictions, axis, low, high


def score(points: list[GCodePoint], predictions: list[bool]) -> tuple[int, int, int, int, float, float, float]:
    truth = [point.label == 7 for point in points]
    tp = sum(t and p for t, p in zip(truth, predictions))
    fp = sum((not t) and p for t, p in zip(truth, predictions))
    fn = sum(t and (not p) for t, p in zip(truth, predictions))
    tn = sum((not t) and (not p) for t, p in zip(truth, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def main() -> None:
    data_dir = Path("Gcode")
    totals = [0, 0, 0, 0]
    axis_names = ["X", "Y", "Z"]

    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() != ".mpf":
            continue

        points = parse_points(path)
        predictions, axis, low, high = predict_class7(points)
        tp, fp, fn, tn, precision, recall, f1 = score(points, predictions)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        totals[3] += tn

        print(
            f"{path.name}\taxis={axis_names[axis]} bounds=({low:g},{high:g})"
            f"\ttrue7={tp + fn}\tpred7={tp + fp}\ttp={tp}\tfp={fp}\tfn={fn}"
            f"\tprecision={precision:.4f}\trecall={recall:.4f}\tf1={f1:.4f}"
        )

        for point, prediction in zip(points, predictions):
            is_true = point.label == 7
            if is_true != prediction:
                kind = "FP" if prediction else "FN"
                print(f"  {kind} {point.line_no}: {point.raw}")

    tp, fp, fn, tn = totals
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / sum(totals) if sum(totals) else 0.0
    print(
        f"TOTAL\ttp={tp}\tfp={fp}\tfn={fn}"
        f"\tprecision={precision:.4f}\trecall={recall:.4f}\tf1={f1:.4f}\taccuracy={accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
