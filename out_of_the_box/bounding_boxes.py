from typing import Tuple, Union, Optional
import cv2
import numpy as np


class Coordinates:
    """
        A class to represent a bounding box.

        Attributes:
            x1 (float): x-coordinate of top-left corner.
            y1 (float): y-coordinate of top-left corner.
            x2 (float): x-coordinate of bottom-right corner.
            y2 (float): y-coordinate of bottom-right corner.
            normalized (bool): Whether coordinates are normalized (0-1) or in pixels.
    """
    def __init__(self, x1: float, y1: float, x2: float, y2: float, normalized: bool = False):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.normalized = normalized
        self._validate_coordinates()

    def _validate_coordinates(self):
        """Validate that coordinates are in the correct order."""
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError("Coordinates are not in the correct order (x1 <= x2 and y1 <= y2)")

    def __str__(self):
        return f"({self.x1:.2f}, {self.y1:.2f}, {self.x2:.2f}, {self.y2:.2f})"

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_pixel(self, img_shape: Tuple[int, int]) -> 'Coordinates':
        """Convert normalized coordinates to pixel coordinates."""
        if not self.normalized:
            return self
        height, width = img_shape
        return Coordinates(
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
            normalized=False
        )

    def to_normalized(self, img_shape: Tuple[int, int]) -> 'Coordinates':
        """Convert pixel coordinates to normalized coordinates."""
        if self.normalized:
            return self
        height, width = img_shape
        return Coordinates(
            self.x1 / width,
            self.y1 / height,
            self.x2 / width,
            self.y2 / height,
            normalized=True
        )


class VOCBox(Coordinates):
    """VOC format bounding box (xmin, ymin, xmax, ymax)"""
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, normalized: bool = False):
        super().__init__(xmin, ymin, xmax, ymax, normalized)

    @property
    def xmin(self) -> float: return self.x1

    @property
    def ymin(self) -> float: return self.y1

    @property
    def xmax(self) -> float: return self.x2

    @property
    def ymax(self) -> float: return self.y2

    def __str__(self):
        return f"VOCBox(xmin={self.xmin:.2f}, ymin={self.ymin:.2f}, xmax={self.xmax:.2f}, ymax={self.ymax:.2f})"


class COCOBox(Coordinates):
    """COCO format bounding box (x, y, width, height)"""
    def __init__(self, x: float, y: float, width: float, height: float, normalized: bool = False):
        super().__init__(x, y, x + width, y + height, normalized)

    @property
    def x(self) -> float: return self.x1

    @property
    def y(self) -> float: return self.y1

    def __str__(self):
        return f"COCOBox(x={self.x:.2f}, y={self.y:.2f}, width={self.width:.2f}, height={self.height:.2f})"


class YOLOBox(Coordinates):
    """YOLO format bounding box (x_center, y_center, width, height)"""
    def __init__(self, x_center: float, y_center: float, width: float, height: float, normalized: bool = True):
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        super().__init__(x1, y1, x2, y2, normalized)

    def __str__(self):
        return f"YOLOBox(x_center={self.x_center:.2f}, y_center={self.y_center:.2f}, width={self.width:.2f}, height={self.height:.2f})"


class BoundingBox:
    def __init__(self, box: Union[VOCBox, COCOBox, YOLOBox], img_shape: Optional[Tuple[int, int]] = None):
        """
        Initialize BoundingBox object.

        Args:
            box: Bounding box in VOC, COCO, or YOLO format.
            img_shape: Image shape (height, width).

        Raises:
            ValueError: If image shape is not provided for normalized coordinates.
        """
        self.original_box = box
        self.img_shape = img_shape

        if box.normalized and img_shape is None:
            raise ValueError("Image shape must be provided for normalized coordinates")

        self.pixel_box = box.to_pixel(img_shape) if box.normalized else box
        self._area = self.pixel_box.width * self.pixel_box.height

    def __str__(self):
        return f"BoundingBox(VOC: {self.to_voc()}, COCO: {self.to_coco()}, YOLO: {self.to_yolo()}, Area: {self.area:.2f})"

    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self._area

    def to_voc(self, normalized: bool = False) -> VOCBox:
        """Convert to VOC format."""
        if normalized and self.img_shape is None:
            raise ValueError("Image shape must be provided for normalization")
        box = self.pixel_box.to_normalized(self.img_shape) if normalized else self.pixel_box
        return VOCBox(box.x1, box.y1, box.x2, box.y2, normalized)

    def to_coco(self, normalized: bool = False) -> COCOBox:
        """Convert to COCO format."""
        if normalized and self.img_shape is None:
            raise ValueError("Image shape must be provided for normalization")
        box = self.pixel_box.to_normalized(self.img_shape) if normalized else self.pixel_box
        return COCOBox(box.x1, box.y1, box.width, box.height, normalized)

    def to_yolo(self, normalized: bool = True) -> YOLOBox:
        """Convert to YOLO format."""
        if normalized and self.img_shape is None:
            raise ValueError("Image shape must be provided for normalized YOLO format")
        if normalized:
            box = self.pixel_box.to_normalized(self.img_shape)
        else:
            box = self.pixel_box
        return YOLOBox(box.x_center, box.y_center, box.width, box.height)

    def draw_on_image(self,
                      img: np.ndarray,
                      color: Tuple[int, int, int] = (255, 0, 0),
                      thickness: int = 2,
                      label: Optional[str] = None) -> np.ndarray:
        """Draw bounding box on image."""
        xmin, ymin, xmax, ymax = map(int, (self.pixel_box.x1, self.pixel_box.y1, self.pixel_box.x2, self.pixel_box.y2))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, label, (xmin, ymin - 10), font, 0.5, color, thickness, cv2.LINE_AA)
        return img

    @staticmethod
    def intersection(box1: 'BoundingBox', box2: 'BoundingBox') -> float:
        """Calculate the intersection area between two bounding boxes."""
        xi1 = max(box1.pixel_box.x1, box2.pixel_box.x1)
        yi1 = max(box1.pixel_box.y1, box2.pixel_box.y1)
        xi2 = min(box1.pixel_box.x2, box2.pixel_box.x2)
        yi2 = min(box1.pixel_box.y2, box2.pixel_box.y2)
        return max(0, xi2 - xi1) * max(0, yi2 - yi1)

    @staticmethod
    def union(box1: 'BoundingBox', box2: 'BoundingBox') -> float:
        """Calculate the union area between two bounding boxes."""
        return box1.area + box2.area - BoundingBox.intersection(box1, box2)

    def percentage_inside(self, other: 'BoundingBox') -> float:
        """Calculate the percentage of this box that is inside the other box."""
        intersection_area = BoundingBox.intersection(self, other)
        return intersection_area / self.area if self.area > 0 else 0

    @staticmethod
    def iou(box1: 'BoundingBox', box2: 'BoundingBox') -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        intersection_area = BoundingBox.intersection(box1, box2)
        union_area = BoundingBox.union(box1, box2)
        return intersection_area / union_area if union_area > 0 else 0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.pixel_box.x1 + self.pixel_box.x2) / 2,
                (self.pixel_box.y1 + self.pixel_box.y2) / 2)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return (self.pixel_box.x1 <= x <= self.pixel_box.x2 and
                self.pixel_box.y1 <= y <= self.pixel_box.y2)

    def overlap_percentage(self, other: 'BoundingBox') -> float:
        intersection = BoundingBox.intersection(self, other)
        return intersection / self.area


# Constants for common colors
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)


def main():
    img_shape = (500, 500)
    img = np.zeros((*img_shape, 3), dtype=np.uint8)

    # Create bounding boxes in different formats
    voc_box = VOCBox(100, 100, 300, 300)
    coco_box = COCOBox(100, 100, 200, 200)
    yolo_box = YOLOBox(0.4, 0.4, 0.2, 0.2)

    bb_voc = BoundingBox(voc_box, img_shape)
    bb_coco = BoundingBox(coco_box, img_shape)
    bb_yolo = BoundingBox(yolo_box, img_shape)

    # Print bounding boxes
    print("VOC bounding box:", bb_voc)
    print("COCO bounding box:", bb_coco)
    print("YOLO bounding box:", bb_yolo)

    # Convert between formats
    print("\nFormat conversions:")
    print("VOC to COCO:", bb_voc.to_coco())
    print("COCO to YOLO:", bb_coco.to_yolo())
    print("YOLO to VOC:", bb_yolo.to_voc())

    # Normalized and pixel coordinates
    print("\nNormalized and pixel coordinates:")
    print("VOC (normalized):", bb_voc.to_voc(normalized=True))
    print("COCO (pixel):", bb_yolo.to_coco(normalized=False))

    # Area calculation
    print("\nAreas:")
    print("VOC box area:", bb_voc.area)
    print("COCO box area:", bb_coco.area)
    print("YOLO box area:", bb_yolo.area)

    # Intersection and Union
    intersection = BoundingBox.intersection(bb_voc, bb_coco)
    union = BoundingBox.union(bb_voc, bb_coco)
    print("\nIntersection and Union:")
    print(f"Intersection between VOC and COCO: {intersection:.2f}")
    print(f"Union between VOC and COCO: {union:.2f}")

    # IoU calculation
    iou = BoundingBox.iou(bb_voc, bb_coco)
    print(f"IoU between VOC and COCO boxes: {iou:.2f}")

    # Percentage inside
    percentage = bb_coco.percentage_inside(bb_voc)
    print(f"\nPercentage of COCO box inside VOC box: {percentage:.2%}")

    # Center point
    print("\nCenter points:")
    print("VOC box center:", bb_voc.center)
    print("COCO box center:", bb_coco.center)
    print("YOLO box center:", bb_yolo.center)

    # Contains point
    test_point = (200, 200)
    print(f"\nPoint {test_point} contained in:")
    print("VOC box:", bb_voc.contains_point(test_point))
    print("COCO box:", bb_coco.contains_point(test_point))
    print("YOLO box:", bb_yolo.contains_point(test_point))

    # Overlap percentage
    overlap = bb_coco.overlap_percentage(bb_voc)
    print(f"\nOverlap percentage of COCO box with VOC box: {overlap:.2%}")

    # Draw the bounding boxes on the image
    img = bb_voc.draw_on_image(img, color=COLOR_RED, thickness=2, label="VOC")
    img = bb_coco.draw_on_image(img, color=COLOR_GREEN, thickness=2, label="COCO")
    img = bb_yolo.draw_on_image(img, color=COLOR_BLUE, thickness=2, label="YOLO")

    # Display the image
    cv2.imshow("Image with bounding boxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
