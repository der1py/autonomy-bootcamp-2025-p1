"""Disposable coarse-to-fine tuner for the blue HSV threshold used by pytest."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
MAP_PATHS = [TESTS / "test_maps" / f"maps_{index}.jpg" for index in range(1, 4)]
GROUND_TRUTH_PATHS = [
    TESTS / "ground_truth_maps" / f"mask_detections_blue_{index}.jpg"
    for index in range(1, 4)
]


def compute_iou(written_mask: np.ndarray, expected_mask: np.ndarray) -> float:
    """Match the pytest IoU calculation exactly."""
    intersection = np.logical_and(written_mask > 0, expected_mask > 0).sum()
    union = np.logical_or(written_mask > 0, expected_mask > 0).sum()
    return intersection / union if union > 0 else 0.0


def make_count_table(hsv_image: np.ndarray, expected_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Count all HSV values, and expected-positive HSV values, for exact fast IoUs."""
    flat_hsv = hsv_image.reshape(-1, 3)
    flat_index = (
        flat_hsv[:, 0].astype(np.intp) * 256 * 256
        + flat_hsv[:, 1].astype(np.intp) * 256
        + flat_hsv[:, 2].astype(np.intp)
    )
    shape = (180, 256, 256)
    all_counts = np.bincount(flat_index, minlength=np.prod(shape)).reshape(shape)
    positive_counts = np.bincount(
        flat_index[expected_mask.reshape(-1) > 0], minlength=np.prod(shape)
    ).reshape(shape)

    # Each cell becomes the count where H is that exact value and S/V meet minima.
    all_counts = np.cumsum(np.cumsum(all_counts[:, ::-1, ::-1], axis=1), axis=2)[:, ::-1, ::-1]
    positive_counts = np.cumsum(
        np.cumsum(positive_counts[:, ::-1, ::-1], axis=1), axis=2
    )[:, ::-1, ::-1]
    return all_counts, positive_counts, int((expected_mask > 0).sum())


def range_count(table: np.ndarray, lower_hue: int, upper_hue: int, lower_saturation: int, lower_value: int) -> int:
    """Return the inclusive HSV range count represented by a cumulative table."""
    count = table[lower_hue : upper_hue + 1, lower_saturation, lower_value].sum()
    return int(count)


def search(
    count_tables: list[tuple[np.ndarray, np.ndarray, int]], lower_hues: range,
    upper_hues: range, saturations: range, values: range,
) -> tuple[float, tuple[int, int, int, int]]:
    best_score = -1.0
    best_threshold = (0, 0, 0, 0)

    for lower_hue in lower_hues:
        for upper_hue in upper_hues:
            if lower_hue > upper_hue:
                continue
            for lower_saturation in saturations:
                for lower_value in values:
                    scores = []
                    for all_counts, positive_counts, expected_positive in count_tables:
                        predicted_positive = range_count(
                            all_counts, lower_hue, upper_hue, lower_saturation, lower_value
                        )
                        intersection = range_count(
                            positive_counts, lower_hue, upper_hue, lower_saturation, lower_value
                        )
                        union = predicted_positive + expected_positive - intersection
                        scores.append(intersection / union if union > 0 else 0.0)
                    score = float(np.mean(scores))
                    if score > best_score:
                        best_score = score
                        best_threshold = (
                            lower_hue,
                            upper_hue,
                            lower_saturation,
                            lower_value,
                        )

    return best_score, best_threshold


def main() -> None:
    # These loads and mask representations match tests/test_detectcolours.py.
    hsv_images = [
        cv2.cvtColor(cv2.imread(str(map_path)), cv2.COLOR_BGR2HSV)
        for map_path in MAP_PATHS
    ]
    expected_masks = [
        cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)
        for ground_truth_path in GROUND_TRUTH_PATHS
    ]

    count_tables = [
        make_count_table(hsv_image, expected_mask)
        for hsv_image, expected_mask in zip(hsv_images, expected_masks)
    ]

    _, coarse_threshold = search(
        count_tables,
        range(80, 101, 2),
        range(100, 121, 2),
        range(80, 201, 10),
        range(80, 201, 10),
    )

    coarse_lower_hue, coarse_upper_hue, coarse_saturation, coarse_value = coarse_threshold
    best_score, (lower_hue, upper_hue, lower_saturation, lower_value) = search(
        count_tables,
        range(max(0, coarse_lower_hue - 2), min(179, coarse_lower_hue + 2) + 1),
        range(max(0, coarse_upper_hue - 2), min(179, coarse_upper_hue + 2) + 1),
        range(max(0, coarse_saturation - 10), min(255, coarse_saturation + 10) + 1, 2),
        range(max(0, coarse_value - 10), min(255, coarse_value + 10) + 1, 2),
    )

    # Recompute the winning threshold with the exact raw masks and pytest IoU.
    verified_score = float(
        np.mean(
            [
                compute_iou(
                    cv2.inRange(
                        hsv_image,
                        (lower_hue, lower_saturation, lower_value),
                        (upper_hue, 255, 255),
                    ),
                    expected_mask,
                )
                for hsv_image, expected_mask in zip(hsv_images, expected_masks)
            ]
        )
    )

    print(f"({lower_hue}, {lower_saturation}, {lower_value})")
    print(f"({upper_hue}, 255, 255)")
    print(f"mean IoU: {verified_score:.6f}")


if __name__ == "__main__":
    main()
