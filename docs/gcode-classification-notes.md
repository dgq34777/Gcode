# G-code Point Classification Notes

This document records the current working understanding from the analysis of the six labeled G-code files in `Gcode/`.

## Dataset

Files analyzed:

- `Gcode/3_wave_whole_123678.mpf`
- `Gcode/3-part1-whole-123678.MPF`
- `Gcode/3-part2-whole-123678.MPF`
- `Gcode/3-part3-whole-123678.MPF`
- `Gcode/3-part4-whole-123678.MPF`
- `Gcode/3-part5-whole-123678.MPF`

Labels:

- `1`, `2`, `3`, `6`: abnormal points
- `7`, `8`: feature points
- no label: normal point

The labels appear at line endings such as `,1`, `,3`, `,7`, `,8`; parsing should also accept `,+1` style if present.

## Current Physical Interpretation

### Class 1

Class 1 represents a local unsmooth transition or corner-like defect. The key pattern is a local curvature direction reversal.

For five consecutive points `P1 P2 P3 P4 P5`, define:

```text
t1 = normalize(P2 - P1)
t2 = normalize(P3 - P2)
t3 = normalize(P4 - P3)
t4 = normalize(P5 - P4)

k123 = t2 - t1
k234 = t3 - t2
k345 = t4 - t3
```

Class 1 is likely when `k234` points against the curvature direction on both sides. In a local 2D projection this often appears as signed curvature pattern:

```text
+ - +
or
- + -
```

Class 1 differs from class 8 because the effect is local. Its one-segment slope jump can be large, but the wider two- or three-segment jump tends to decay.

### Class 2

Class 2 represents a sudden feed-direction reversal anomaly.

The strongest feature is the angle between the incoming and outgoing feed vectors:

```text
turn1 = angle(Pi - P(i-1), P(i+1) - Pi)
```

Class 2 is likely when `turn1` is close to reverse direction, typically above about `135` degrees. This captures most class 2 points, but can over-predict nearby normal points if evaluated point-by-point rather than event-by-event.

### Class 3

Class 3 appears to be a short-segment or near-duplicate companion point around a feature boundary.

Typical properties:

- very small `min(dprev, dnext)`
- very large `step_ratio = max(dprev, dnext) / min(dprev, dnext)`
- often adjacent to class 8 or class 7
- often has small local turn, but a larger wider-window slope jump

Class 3 is not always covered by the short-segment rule, so it likely has multiple subtypes.

### Class 6

Class 6 appears to be a smooth companion point inside or near a transition region.

Typical properties:

- local `turn1` is small
- two-point or wider-window slope change can be large
- often adjacent to class 8 or another class 6
- many false negatives look locally normal, which suggests class 6 depends on regional context rather than a single-point geometric signature

### Class 7

Class 7 is a scanline boundary or row-turning point.

The machining path moves from one side to the other, turns roughly `90` degrees, advances by a small step, then turns roughly `90` degrees back along the opposite direction. Class 7 marks the two turning points for each scanline row, plus start/end boundary points.

Typical rule:

```text
turn1 ~= 90 degrees
turn2 or wider-window turn indicates a row reversal
slope jump is usually small
```

Class 7 is currently the most reliably recognized class.

### Class 8

Class 8 is a transition between a flatter/stable area and a curved or changing area.

The better mathematical definition is not just "large slope", but a persistent slope step across scale.

Useful features:

```text
J1 = one-segment slope jump around Pi
J2 = two-segment slope jump around Pi
D2 = distance from Pi to the chord P(i-2)-P(i+2)
rho = step_ratio
```

Current class 8 rule:

```text
not class 7
not class 1
turn1 <= 105 degrees
J1 >= 10 degrees
(J2 >= 30 degrees or rho >= 8)
D2 >= 0.2
```

This rule improved class 8 precision by filtering normal and class 3 points whose apparent slope jump is caused by tiny local segments.

## Experimental Results

All metrics below are from local rule-based experiments on usable G01 points with sufficient context window. Counts vary slightly depending on the window size.

### Class 1 Detector Evolution

| Rule | Precision | Recall | F1 |
|---|---:|---:|---:|
| Curvature flip only | 51.82% | 76.34% | 61.74% |
| + step ratio filter | 71.43% | 86.02% | 78.05% |
| + local NMS | 76.15% | 89.25% | 82.18% |

### 1/7/8 Rule System

After refining class 8 with persistent slope step and `D2`:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| 1 | 75.93% | 89.13% | 82.00% |
| 7 | 98.67% | 94.17% | 96.37% |
| 8 | 87.15% | 82.74% | 84.89% |

Macro-F1 for classes `1/7/8`: about `87.75%`.

### 1/2/3/6/7/8 Rule System

Current six-class rule baseline:

| Class | Precision | Recall | F1 | Normal false positives |
|---|---:|---:|---:|---:|
| 1 | 75.93% | 89.13% | 82.00% | 10 |
| 2 | 54.14% | 90.00% | 67.61% | 57 |
| 3 | 61.34% | 48.00% | 53.86% | 12 |
| 6 | 70.74% | 47.11% | 56.56% | 73 |
| 7 | 98.67% | 94.17% | 96.37% | 1 |
| 8 | 95.30% | 76.85% | 85.09% | 6 |

Accuracy: about `95.8%`.

Macro-F1 over `1/2/3/6/7/8`: about `73.58%`.

Note: `scripts/evaluate_rules.py` is a reconstructed executable version based on these notes. The exact exploratory script and parameter set that produced the reference baseline above were not committed from the previous computer.

## Main Error Sources

### Class 1

False positives are mostly normal points and some class 8 points that also have local curvature reversal. Tightening `D2` or `J2` constraints hurt recall too much, so the current class 1 rule should not be over-tightened.

### Class 2

The definition is correct for feed reversal, but point-level labeling is difficult. Neighboring normal points near a reversal event can also have very high reversal angle.

Using local NMS improves precision but reduces recall:

| Variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| Raw `turn1 >= 135` | 54.14% | 90.00% | 67.61% |
| With local NMS | 63.80% | 55.00% | 59.06% |

Use raw rule for point recall; use NMS for event-level detection.

### Class 3

The short-segment subtype is recognizable, but many class 3 points are not extremely short and look locally normal. They likely need regional context around class 8 and class 7.

### Class 6

Many class 6 false negatives are locally smooth:

```text
turn1 ~= 0.8 degrees
J2 ~= 1.7 degrees
D2 ~= 0.03
```

This suggests class 6 is partly a regional/semantic label, not just a local geometric anomaly.

### Class 8

The refined persistent slope-step rule works well. Remaining false negatives include weaker transition points or class 8 points absorbed by class 3.

## Suggested Next Step

Move from single-point rules to a regional two-stage recognizer:

1. Detect strong structural anchors: `7`, `1`, `2`, `8`.
2. Use detected `8` points to define transition regions.
3. Classify `3` and `6` inside those regions using local geometry plus position relative to nearest `8`.
4. Suppress `3/6` predictions outside transition regions unless the local signal is very strong.

This should reduce normal false positives and improve recall for classes `3` and `6`.
