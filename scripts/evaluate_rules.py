from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


TARGET_CLASSES = [1, 2, 3, 6, 7, 8]
COORD_RE = re.compile(r"([XYZ])\s*=?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
GCODE_RE = re.compile(r"\bG0*([01])\b", re.IGNORECASE)
LABEL_RE = re.compile(r",\s*([+-]?\d+)\s*$")

# This is a reconstructed executable baseline from docs/gcode-classification-notes.md.
# The exact exploratory script/parameters from the previous computer were not committed.


@dataclass(frozen=True)
class GCodePoint:
    file: str
    line_no: int
    xyz: np.ndarray
    label: int
    raw: str


def parse_label(code: str) -> tuple[str, int]:
    match = LABEL_RE.search(code)
    if not match:
        return code, 0
    return code[: match.start()].rstrip(), int(match.group(1))


def parse_points(path: Path) -> list[GCodePoint]:
    points: list[GCodePoint] = []
    current_motion: str | None = None
    last_xyz: dict[str, float] = {}

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        code = raw_line.split(";", 1)[0].strip()
        if not code:
            continue

        for motion in GCODE_RE.findall(code):
            current_motion = f"G0{motion}"

        code, label = parse_label(code)
        coords = {axis.upper(): float(value) for axis, value in COORD_RE.findall(code)}
        if coords:
            last_xyz.update(coords)

        if current_motion != "G01" or not coords or not {"X", "Y", "Z"}.issubset(last_xyz):
            continue

        points.append(
            GCodePoint(
                file=path.name,
                line_no=line_no,
                xyz=np.array([last_xyz["X"], last_xyz["Y"], last_xyz["Z"]], dtype=float),
                label=label if label in TARGET_CLASSES else 0,
                raw=raw_line.rstrip(),
            )
        )

    return points


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    cosine = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def point_to_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    chord = end - start
    denom = float(np.linalg.norm(chord))
    if denom == 0:
        return float(np.linalg.norm(point - start))
    return float(np.linalg.norm(np.cross(chord, start - point)) / denom)


def signed_xz_turn(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[2] - a[2] * b[0])


def feature_table(points: list[GCodePoint]) -> list[dict[str, float]]:
    xyz = np.array([point.xyz for point in points])
    features: list[dict[str, float]] = [dict(valid=0.0) for _ in points]

    for i in range(2, len(points) - 2):
        if len({points[j].file for j in range(i - 2, i + 3)}) != 1:
            continue

        p_im2, p_im1, p_i, p_ip1, p_ip2 = xyz[i - 2], xyz[i - 1], xyz[i], xyz[i + 1], xyz[i + 2]
        v_prev = p_i - p_im1
        v_next = p_ip1 - p_i
        d_prev = float(np.linalg.norm(v_prev))
        d_next = float(np.linalg.norm(v_next))
        min_step = min(d_prev, d_next)
        max_step = max(d_prev, d_next)
        step_ratio = max_step / max(min_step, 1e-9)

        v1 = p_im1 - p_im2
        v2 = p_i - p_im1
        v3 = p_ip1 - p_i
        v4 = p_ip2 - p_ip1
        signs = [
            math.copysign(1.0, signed_xz_turn(v1, v2)) if abs(signed_xz_turn(v1, v2)) > 1e-9 else 0.0,
            math.copysign(1.0, signed_xz_turn(v2, v3)) if abs(signed_xz_turn(v2, v3)) > 1e-9 else 0.0,
            math.copysign(1.0, signed_xz_turn(v3, v4)) if abs(signed_xz_turn(v3, v4)) > 1e-9 else 0.0,
        ]

        turn1 = angle_degrees(v_prev, v_next)
        j2 = angle_degrees(p_i - p_im2, p_ip2 - p_i)
        d2 = point_to_line_distance(p_i, p_im2, p_ip2)
        local_decay = j2 / max(turn1, 1e-9)
        curvature_flip = signs[0] != 0 and signs[1] != 0 and signs[2] != 0 and signs[0] == signs[2] and signs[1] == -signs[0]

        features[i] = dict(
            valid=1.0,
            turn1=turn1,
            j2=j2,
            d2=d2,
            d_prev=d_prev,
            d_next=d_next,
            min_step=min_step,
            max_step=max_step,
            step_ratio=step_ratio,
            curvature_flip=float(curvature_flip),
            local_decay=local_decay,
        )

    return features


def suppress_to_local_maxima(candidates: list[int], scores: list[float], radius: int = 1) -> set[int]:
    kept: set[int] = set()
    candidate_set = set(candidates)
    for idx in candidates:
        lo = max(0, idx - radius)
        hi = min(len(scores), idx + radius + 1)
        neighbors = [j for j in range(lo, hi) if j in candidate_set]
        if not neighbors:
            continue
        best = max(neighbors, key=lambda j: (scores[j], -abs(j - idx)))
        if idx == best:
            kept.add(idx)
    return kept


def near_prediction(predictions: list[int], index: int, labels: set[int], radius: int) -> bool:
    lo = max(0, index - radius)
    hi = min(len(predictions), index + radius + 1)
    return any(predictions[j] in labels for j in range(lo, hi) if j != index)


def predict_rules(points: list[GCodePoint], features: list[dict[str, float]]) -> list[int]:
    predictions = [0 for _ in points]

    class7_candidates: list[int] = []
    class1_candidates: list[int] = []
    class1_scores = [0.0 for _ in points]

    def has_paired_right_turn(index: int) -> bool:
        for neighbor in (index - 1, index + 1):
            if not 0 <= neighbor < len(features):
                continue
            if points[neighbor].file != points[index].file or not features[neighbor].get("valid"):
                continue
            distance = float(np.linalg.norm(points[neighbor].xyz - points[index].xyz))
            if 80 <= features[neighbor]["turn1"] <= 100 and distance <= 2.0:
                return True
        return False

    for i, f in enumerate(features):
        if not f.get("valid"):
            continue

        if 80 <= f["turn1"] <= 100 and (
            has_paired_right_turn(i)
            or (f["j2"] >= 75.0 and f["d2"] >= 0.25 and f["max_step"] <= 1.5)
        ):
            class7_candidates.append(i)

        if (
            (
                f["curvature_flip"] and f["step_ratio"] <= 4.0 and f["turn1"] >= 4.0 and f["local_decay"] <= 1.35
            )
            or (
                f["turn1"] >= 10.0 and f["local_decay"] <= 0.9 and f["step_ratio"] <= 6.0 and f["j2"] <= 60.0
            )
        ):
            class1_candidates.append(i)
            class1_scores[i] = f["turn1"] + 0.25 * f["j2"]

    for idx in class7_candidates:
        predictions[idx] = 7

    for i, f in enumerate(features):
        if not f.get("valid") or predictions[i] != 0:
            continue
        if f["turn1"] >= 135:
            predictions[i] = 2

    for idx in suppress_to_local_maxima(class1_candidates, class1_scores, radius=1):
        if predictions[idx] == 0:
            predictions[idx] = 1

    for i, f in enumerate(features):
        if not f.get("valid") or predictions[i] != 0:
            continue
        if f["min_step"] <= 0.05 and f["step_ratio"] >= 20 and 2 <= f["turn1"] <= 60:
            predictions[i] = 3

    for i, f in enumerate(features):
        if not f.get("valid") or predictions[i] != 0:
            continue
        if f["turn1"] <= 105 and f["turn1"] >= 10 and (f["j2"] >= 30 or f["step_ratio"] >= 8) and f["d2"] >= 0.2:
            predictions[i] = 8

    for i, f in enumerate(features):
        if not f.get("valid") or predictions[i] != 0:
            continue
        if f["min_step"] <= 0.08 and f["step_ratio"] >= 8 and 2 <= f["turn1"] <= 80 and near_prediction(predictions, i, {7, 8}, radius=2):
            predictions[i] = 3

    for i, f in enumerate(features):
        if not f.get("valid") or predictions[i] != 0:
            continue
        in_transition = near_prediction(predictions, i, {8}, radius=2)
        strong_local_hint = f["j2"] >= 15 and f["d2"] >= 0.18 and f["step_ratio"] >= 2
        if in_transition and f["turn1"] <= 8 and strong_local_hint:
            predictions[i] = 6

    return predictions


def load_dataset(data_dir: Path) -> list[GCodePoint]:
    files = sorted(data_dir.glob("*.mpf")) + sorted(data_dir.glob("*.MPF"))
    if not files:
        raise FileNotFoundError(f"No MPF files found under {data_dir}")

    points: list[GCodePoint] = []
    for path in files:
        points.extend(parse_points(path))
    return points


def evaluate(
    points: list[GCodePoint], features: list[dict[str, float]], predictions: list[int]
) -> tuple[list[int], list[int], list[int]]:
    labels = [point.label for point in points]
    usable = [i for i, feature in enumerate(features) if feature.get("valid")]
    return [labels[i] for i in usable], [predictions[i] for i in usable], usable


def print_error_examples(points: list[GCodePoint], y_true: list[int], y_pred: list[int], indices: list[int], limit: int) -> None:
    if limit <= 0:
        return
    print("\nError examples:")
    shown = 0
    for index, truth, pred in zip(indices, y_true, y_pred):
        if truth == pred:
            continue
        point = points[index]
        print(f"- {point.file}:{point.line_no} true={truth} pred={pred} {point.raw}")
        shown += 1
        if shown >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based G-code point classification.")
    parser.add_argument("--data-dir", type=Path, default=Path("Gcode"), help="Directory containing labeled MPF files.")
    parser.add_argument("--errors", type=int, default=0, help="Print up to N misclassified points.")
    args = parser.parse_args()

    points = load_dataset(args.data_dir)
    features = feature_table(points)
    predictions = predict_rules(points, features)
    y_true, y_pred, indices = evaluate(points, features, predictions)

    print(f"Files directory: {args.data_dir}")
    print(f"Parsed G01 points: {len(points)}")
    print(f"Evaluated points with +/-2 context: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print()
    print(classification_report(y_true, y_pred, labels=TARGET_CLASSES, zero_division=0, digits=4))
    print("Confusion matrix rows=true, columns=pred for labels [0, 1, 2, 3, 6, 7, 8]:")
    labels = [0] + TARGET_CLASSES
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print_error_examples(points, y_true, y_pred, indices, args.errors)


if __name__ == "__main__":
    main()
