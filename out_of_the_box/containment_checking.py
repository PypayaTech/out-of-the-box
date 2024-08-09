from enum import Enum
from typing import Callable
from out_of_the_box.bounding_boxes import BoundingBox


class ContainmentMethod(Enum):
    IOU = "iou"
    PERCENTAGE_INSIDE = "percentage_inside"
    ADAPTIVE = "adaptive"


class BoxContainmentChecker:
    def __init__(self, method: ContainmentMethod = ContainmentMethod.ADAPTIVE, threshold: float = 0.5):
        if not isinstance(method, ContainmentMethod):
            raise ValueError(f"Invalid containment method: {method}")
        self.method = method
        self.threshold = threshold

    def is_contained(self, inner_box: BoundingBox, outer_box: BoundingBox) -> bool:
        iou = BoundingBox.iou(inner_box, outer_box)
        percentage_inside = inner_box.percentage_inside(outer_box)

        if self.method == ContainmentMethod.IOU:
            return iou >= self.threshold
        elif self.method == ContainmentMethod.PERCENTAGE_INSIDE:
            return percentage_inside >= self.threshold
        elif self.method == ContainmentMethod.ADAPTIVE:
            return iou >= self.threshold or percentage_inside >= self.threshold
        else:
            raise ValueError(f"Unknown containment method: {self.method}")

    @classmethod
    def custom(cls, check_function: Callable[[BoundingBox, BoundingBox], bool]) -> 'BoxContainmentChecker':
        checker = cls(ContainmentMethod.ADAPTIVE)
        checker.is_contained = check_function
        return checker
