"""
BOOTCAMPERS TO COMPLETE.

Detects colours on a map of landing pads.
"""

from pathlib import Path
import cv2
import numpy as np


class DetectBlue:
    """
    Detects blue objects from an image.
    """

    __create_key = object()

    @classmethod
    def create(cls) -> "DetectBlue":
        """
        Factory method to create DetectBlue instance.
        """

        return DetectBlue(cls.__create_key)

    def __init__(self, class_create_private_key: object) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_create_private_key is DetectBlue.__create_key, "Use create() method"

    def run(self, image: str, output_path: Path, return_mask: bool = False) -> None | np.ndarray:
        """
        Detects blue from an image and shows the annotated result.

        image: The image to run the colour detection algorithm on.
        output_path: Path to output the resulting image with annotated detections.
        return_mask: Option to return the mask (black and white version of colour detection).
        """
        img = cv2.imread(image)

        # ============
        # ↓ BOOTCAMPERS MODIFY BELOW THIS COMMENT ↓
        # ============

        # Convert the image's colour to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set upper and lower bounds for colour detection, this is in HSV
        lower_blue = 78
        upper_blue = 112

        # Apply the threshold for the colour detection
        mask = cv2.inRange(hsv, (lower_blue, 150, 70), (upper_blue, 255, 255))

        # ============
        # ↑ BOOTCAMPERS MODIFY ABOVE THIS COMMENT ↑
        # ============

        # Annotate the colour detections
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # Show the annotated detection!
        cv2.imwrite(str(output_path), mask)  # change this shit back to img later

        # This parameter is needed to run tests
        return mask if return_mask else None


class DetectRed:
    """
    Detects red objects from an image.
    """

    __create_key = object()

    @classmethod
    def create(cls) -> "DetectRed":
        """
        Factory method to create DetectRed instance.
        """

        return DetectRed(cls.__create_key)

    def __init__(self, class_create_private_key: object) -> None:
        """
        Private constructor, use create() method.
        """
        assert class_create_private_key is DetectRed.__create_key, "Use create() method"

    def run(self, image: str, output_path: Path, return_mask: bool = False) -> None | np.ndarray:
        """
        Detects red from an image and shows the annotated result.

        image: The image to run the colour detection algorithm on.
        output_path: Path to output the resulting image with annotated detections.
        return_mask: Option to return the mask (black and white version of colour detection).
        """
        img = cv2.imread(image)

        # ============
        # ↓ BOOTCAMPERS MODIFY BELOW THIS COMMENT ↓
        # ============

        # Convert the image's colour to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set upper and lower bounds for colour detection, this is in HSV
        lower_red = 0
        upper_red = 10

        low_mask = cv2.inRange(hsv, (lower_red, 120, 120), (upper_red, 255, 255))

        # detect both hsv red ranges
        lower_red = 170
        upper_red = 180

        high_mask = cv2.inRange(hsv, (lower_red, 120, 120), (upper_red, 255, 255))

        mask = cv2.bitwise_or(low_mask, high_mask)

        # Annotate the colour detections
        # replace the '_' parameter with the appropiate variable
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # ============
        # ↑ BOOTCAMPERS MODIFY ABOVE THIS COMMENT ↑
        # ============

        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # Show the annotated detection!
        cv2.imwrite(str(output_path), img)

        # ============
        # ↓ BOOTCAMPERS MODIFY BELOW THIS COMMENT ↓
        # ============

        # Include the "return_mask" parameter if statement here, similar to how it is implemented in DetectBlue
        # Tests will not pass if this isn't included!
        return mask if return_mask else None

        # ============
        # ↑ BOOTCAMPERS MODIFY ABOVE THIS COMMENT ↑
        # ============
