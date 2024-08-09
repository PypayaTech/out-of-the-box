import pytest
import numpy as np
from out_of_the_box.bounding_boxes import Coordinates, VOCBox, COCOBox, YOLOBox, BoundingBox


@pytest.fixture
def img_shape():
    return (500, 500)


def test_coordinates_initialization():
    coord = Coordinates(0.1, 0.2, 0.3, 0.4, normalized=True)
    assert coord.x1 == 0.1
    assert coord.y1 == 0.2
    assert coord.x2 == 0.3
    assert coord.y2 == 0.4
    assert coord.normalized == True


def test_coordinates_validation():
    with pytest.raises(ValueError):
        Coordinates(0.3, 0.4, 0.1, 0.2)


def test_coordinates_properties():
    coord = Coordinates(100, 100, 300, 300)
    assert coord.x_center == 200
    assert coord.y_center == 200
    assert coord.width == 200
    assert coord.height == 200


def test_coordinates_conversion(img_shape):
    norm_coord = Coordinates(0.2, 0.2, 0.6, 0.6, normalized=True)
    pixel_coord = norm_coord.to_pixel(img_shape)
    assert pixel_coord.x1 == 100
    assert pixel_coord.y1 == 100
    assert pixel_coord.x2 == 300
    assert pixel_coord.y2 == 300
    assert pixel_coord.normalized == False

    norm_coord_2 = pixel_coord.to_normalized(img_shape)
    assert norm_coord_2.x1 == pytest.approx(0.2)
    assert norm_coord_2.y1 == pytest.approx(0.2)
    assert norm_coord_2.x2 == pytest.approx(0.6)
    assert norm_coord_2.y2 == pytest.approx(0.6)
    assert norm_coord_2.normalized == True


def test_voc_box():
    voc = VOCBox(100, 100, 300, 300)
    assert voc.xmin == 100
    assert voc.ymin == 100
    assert voc.xmax == 300
    assert voc.ymax == 300


def test_coco_box():
    coco = COCOBox(100, 100, 200, 200)
    assert coco.x == 100
    assert coco.y == 100
    assert coco.width == 200
    assert coco.height == 200


def test_yolo_box():
    yolo = YOLOBox(0.5, 0.5, 0.2, 0.2)
    assert yolo.x_center == pytest.approx(0.5)
    assert yolo.y_center == pytest.approx(0.5)
    assert yolo.width == pytest.approx(0.2)
    assert yolo.height == pytest.approx(0.2)


def test_bounding_box_initialization(img_shape):
    voc = VOCBox(100, 100, 300, 300)
    bb = BoundingBox(voc, img_shape)
    assert isinstance(bb.pixel_box, VOCBox)
    assert bb.pixel_box.x1 == 100

    yolo = YOLOBox(0.5, 0.5, 0.2, 0.2)
    bb = BoundingBox(yolo, img_shape)
    assert isinstance(bb.pixel_box, Coordinates)
    assert bb.pixel_box.x1 == 200


def test_bounding_box_conversion(img_shape):
    voc = VOCBox(100, 100, 300, 300)
    bb = BoundingBox(voc, img_shape)

    coco = bb.to_coco()
    assert isinstance(coco, COCOBox)
    assert coco.x == 100
    assert coco.y == 100
    assert coco.width == 200
    assert coco.height == 200

    yolo = bb.to_yolo()
    assert isinstance(yolo, YOLOBox)
    assert yolo.x_center == pytest.approx(0.4)
    assert yolo.y_center == pytest.approx(0.4)
    assert yolo.width == pytest.approx(0.4)
    assert yolo.height == pytest.approx(0.4)


def test_bounding_box_iou():
    img_shape = (500, 500)
    bb1 = BoundingBox(VOCBox(100, 100, 300, 300), img_shape)
    bb2 = BoundingBox(VOCBox(200, 200, 400, 400), img_shape)
    iou = BoundingBox.iou(bb1, bb2)
    assert iou == pytest.approx(0.14285714285714285)


def test_bounding_box_draw_on_image():
    img_shape = (500, 500)
    img = np.zeros((*img_shape, 3), dtype=np.uint8)
    bb = BoundingBox(VOCBox(100, 100, 300, 300), img_shape)
    img = bb.draw_on_image(img, color=(255, 0, 0), thickness=2, label="Test")

    # Check if the bounding box is drawn
    assert np.any(img[100:300, 100:300] != 0)

    # Check if the label is drawn
    assert np.any(img[90:100, 100:150] != 0)


def test_bounding_box_normalized_conversion():
    img_shape = (500, 500)
    voc = VOCBox(100, 100, 300, 300)
    bb = BoundingBox(voc, img_shape)

    voc_norm = bb.to_voc(normalized=True)
    assert voc_norm.xmin == pytest.approx(0.2)
    assert voc_norm.ymin == pytest.approx(0.2)
    assert voc_norm.xmax == pytest.approx(0.6)
    assert voc_norm.ymax == pytest.approx(0.6)


def test_bounding_box_error_handling():
    with pytest.raises(ValueError):
        BoundingBox(YOLOBox(0.5, 0.5, 0.2, 0.2))  # Missing img_shape for normalized coordinates

    bb = BoundingBox(VOCBox(100, 100, 300, 300))
    with pytest.raises(ValueError):
        bb.to_yolo()  # Missing img_shape for YOLO conversion


def test_coordinates_edge_cases():
    # Test with minimum valid values
    coord = Coordinates(0, 0, 1, 1)
    assert coord.width == 1
    assert coord.height == 1

    # Test with very small differences
    coord = Coordinates(0.1, 0.1, 0.100001, 0.100001)
    assert coord.width > 0
    assert coord.height > 0


def test_bounding_box_conversions_consistency(img_shape):
    original_box = VOCBox(100, 100, 300, 300)
    bb = BoundingBox(original_box, img_shape)

    # Convert to different formats and back
    coco = bb.to_coco()
    yolo = bb.to_yolo()
    voc_from_coco = BoundingBox(coco, img_shape).to_voc()
    voc_from_yolo = BoundingBox(yolo, img_shape).to_voc()

    assert voc_from_coco.xmin == pytest.approx(original_box.xmin)
    assert voc_from_coco.ymin == pytest.approx(original_box.ymin)
    assert voc_from_coco.xmax == pytest.approx(original_box.xmax)
    assert voc_from_coco.ymax == pytest.approx(original_box.ymax)

    assert voc_from_yolo.xmin == pytest.approx(original_box.xmin)
    assert voc_from_yolo.ymin == pytest.approx(original_box.ymin)
    assert voc_from_yolo.xmax == pytest.approx(original_box.xmax)
    assert voc_from_yolo.ymax == pytest.approx(original_box.ymax)


def test_iou_edge_cases():
    img_shape = (500, 500)

    # Identical boxes
    bb1 = BoundingBox(VOCBox(100, 100, 300, 300), img_shape)
    bb2 = BoundingBox(VOCBox(100, 100, 300, 300), img_shape)
    assert BoundingBox.iou(bb1, bb2) == pytest.approx(1.0)

    # No overlap
    bb3 = BoundingBox(VOCBox(0, 0, 100, 100), img_shape)
    bb4 = BoundingBox(VOCBox(200, 200, 300, 300), img_shape)
    assert BoundingBox.iou(bb3, bb4) == pytest.approx(0.0)

    # Complete containment
    bb5 = BoundingBox(VOCBox(100, 100, 300, 300), img_shape)
    bb6 = BoundingBox(VOCBox(150, 150, 250, 250), img_shape)
    expected_iou = (100 * 100) / (200 * 200)
    assert BoundingBox.iou(bb5, bb6) == pytest.approx(expected_iou)


def test_yolo_box_conversion():
    img_shape = (500, 500)
    yolo = YOLOBox(0.5, 0.5, 0.2, 0.2)
    bb = BoundingBox(yolo, img_shape)

    voc = bb.to_voc()
    assert voc.xmin == pytest.approx(200)
    assert voc.ymin == pytest.approx(200)
    assert voc.xmax == pytest.approx(300)
    assert voc.ymax == pytest.approx(300)

    coco = bb.to_coco()
    assert coco.x == pytest.approx(200)
    assert coco.y == pytest.approx(200)
    assert coco.width == pytest.approx(100)
    assert coco.height == pytest.approx(100)
