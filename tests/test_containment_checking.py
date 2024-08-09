import pytest
from out_of_the_box.bounding_boxes import BoundingBox, VOCBox
from out_of_the_box.containment_checking import BoxContainmentChecker, ContainmentMethod


@pytest.fixture
def image_shape():
    return (100, 100)


@pytest.fixture
def inner_box(image_shape):
    return BoundingBox(VOCBox(20, 20, 40, 40), image_shape)


@pytest.fixture
def outer_box(image_shape):
    return BoundingBox(VOCBox(10, 10, 50, 50), image_shape)


@pytest.fixture
def non_containing_box(image_shape):
    return BoundingBox(VOCBox(60, 60, 90, 90), image_shape)


@pytest.fixture
def box_metrics(inner_box, outer_box, non_containing_box):
    return {
        "inner_outer_iou": BoundingBox.iou(inner_box, outer_box),
        "inner_outer_percentage": inner_box.percentage_inside(outer_box),
        "outer_inner_percentage": outer_box.percentage_inside(inner_box),
        "inner_non_containing_iou": BoundingBox.iou(inner_box, non_containing_box),
        "inner_non_containing_percentage": inner_box.percentage_inside(non_containing_box),
    }


def test_iou_method(inner_box, outer_box, non_containing_box, box_metrics):
    iou = box_metrics["inner_outer_iou"]

    checker_low = BoxContainmentChecker(ContainmentMethod.IOU, threshold=iou - 0.01)
    checker_high = BoxContainmentChecker(ContainmentMethod.IOU, threshold=iou + 0.01)

    assert checker_low.is_contained(inner_box, outer_box) == True
    assert checker_high.is_contained(inner_box, outer_box) == False
    assert checker_low.is_contained(inner_box, non_containing_box) == False


def test_percentage_inside_method(inner_box, outer_box, box_metrics):
    percentage = box_metrics["inner_outer_percentage"]

    checker_low = BoxContainmentChecker(ContainmentMethod.PERCENTAGE_INSIDE, threshold=percentage - 0.01)
    checker_high = BoxContainmentChecker(ContainmentMethod.PERCENTAGE_INSIDE, threshold=percentage + 0.01)

    assert checker_low.is_contained(inner_box, outer_box) == True
    assert checker_high.is_contained(inner_box, outer_box) == False
    assert checker_low.is_contained(outer_box, inner_box) == False


def test_adaptive_method(inner_box, outer_box, box_metrics):
    iou = box_metrics["inner_outer_iou"]
    percentage = box_metrics["inner_outer_percentage"]
    threshold = max(iou, percentage)

    checker_low = BoxContainmentChecker(ContainmentMethod.ADAPTIVE, threshold=threshold - 0.01)
    checker_high = BoxContainmentChecker(ContainmentMethod.ADAPTIVE, threshold=threshold + 0.01)

    assert checker_low.is_contained(inner_box, outer_box) == True
    assert checker_high.is_contained(inner_box, outer_box) == False
    assert checker_low.is_contained(outer_box, inner_box) == False


def test_custom_method(inner_box, outer_box):
    def custom_check(box1, box2):
        return box1.area < box2.area

    checker = BoxContainmentChecker.custom(custom_check)
    assert checker.is_contained(inner_box, outer_box) == True
    assert checker.is_contained(outer_box, inner_box) == False


def test_invalid_method():
    with pytest.raises(ValueError):
        BoxContainmentChecker("invalid_method")
