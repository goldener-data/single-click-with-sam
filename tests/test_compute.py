import pytest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image as PILImage

from src.compute import (
    extract_connected_components_from_binary_mask,
    extract_bounding_boxes_from_connected_components,
    extract_random_points_from_connected_components,
    predict_sam_logits_from_single_click,
    threshold_single_click_sam_logits,
    compute_ious_from_sam_masks_and_connected_components,
    predict_sam_masks_from_single_click,
    sam_execution_time_for_single_click,
)


@pytest.fixture
def mask1() -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    return mask


@pytest.fixture
def mask2() -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[6:8, 6:8] = 1
    return mask


@pytest.fixture
def mask(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> np.ndarray:
    m = mask1 + mask2
    assert isinstance(
        m, np.ndarray
    )  # fix mypy: error: Returning Any from function declared to return "ndarray[Any, Any]"
    return m


@pytest.fixture
def connected_components(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> np.ndarray:
    return np.stack((mask1, mask2), axis=0)


@pytest.fixture
def boxes() -> np.ndarray:
    return np.array([[2, 2, 5, 5], [6, 6, 8, 8]], dtype=np.int64)


@pytest.fixture
def points() -> np.ndarray:
    return np.array([[[3, 3]], [[7, 7]]], dtype=np.int64)


@pytest.fixture
def mock_sam_model() -> MagicMock:
    mock_model = MagicMock()
    mock_model.set_image = MagicMock()

    def predict(
        box: np.ndarray | None,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        return_logits: bool,
        multimask_output: bool,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        None,
    ]:
        mask = np.zeros((10, 10))
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        mask[y1:y2, x1:x2] = 1
        logits = np.repeat(mask[np.newaxis, ...], 3, axis=0)

        return (logits, np.array([1, 0.5, 0.0]), None)

    mock_model.predict = predict
    return mock_model


@pytest.fixture
def sam_logits(
    connected_components: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    logits_list = []

    for cc, box in zip(connected_components, boxes, strict=True):
        logit = cc.astype(np.float32) - 1
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        logit[y1:y2, x1:x2] = 1
        logits = np.stack(
            [
                logit,
                np.ones_like(logit) * -1,
                np.ones_like(logit),
                np.ones_like(logit),
                np.ones_like(logit) * 0.5,
                np.zeros_like(logit),
            ],
            axis=0,
        )
        logits_list.append(logits[np.newaxis, ...])

    return np.stack(logits_list, axis=0).astype(np.float32)


class TestConnectedComponents:
    def test_connected_components_valid(
        self, mask1: np.ndarray, mask2: np.ndarray, mask: np.ndarray
    ) -> None:
        result = extract_connected_components_from_binary_mask(mask, min_area=1)
        assert result is not None
        assert np.all(np.isin(result, [0, 1]))
        assert result.shape == (2, 10, 10)
        assert np.all(result[0] == mask1)
        assert np.all(result[1] == mask2)

    def test_connected_components_none(self, mask1: np.ndarray) -> None:
        assert extract_connected_components_from_binary_mask(mask1, min_area=10) is None
        assert (
            extract_connected_components_from_binary_mask(
                np.zeros((10, 10), dtype=np.uint8), min_area=1
            )
            is None
        )

    def test_invalid_mask(self) -> None:
        with pytest.raises(ValueError):
            extract_connected_components_from_binary_mask(
                np.zeros((10, 10, 3), dtype=np.uint8), min_area=1
            )


class TestBoundingBoxes:
    def test_bounding_boxes_valid(
        self, connected_components: np.ndarray, boxes: np.ndarray
    ) -> None:
        extracted_boxes = extract_bounding_boxes_from_connected_components(
            connected_components
        )
        assert extracted_boxes is not None
        assert extracted_boxes.shape == (2, 4)
        assert np.all(extracted_boxes == boxes)

    def test_bounding_boxes_none(self) -> None:
        assert extract_bounding_boxes_from_connected_components(None) is None

    def test_bounding_boxes_invalid(self, mask1: np.ndarray, mask2: np.ndarray) -> None:
        with pytest.raises(ValueError):
            extract_bounding_boxes_from_connected_components(
                np.zeros((10, 10), dtype=np.uint8)
            )

        with pytest.raises(ValueError):
            extract_bounding_boxes_from_connected_components(
                np.zeros((2, 10, 10), dtype=np.uint8)
            )

        with pytest.raises(ValueError):
            extract_bounding_boxes_from_connected_components(
                (mask1 + mask2).reshape(1, 10, 10),
            )


class TestRandomPoints:
    def test_random_points_valid(self, connected_components: np.ndarray) -> None:
        points = extract_random_points_from_connected_components(
            connected_components, num_points=2
        )
        assert points.shape == (2, 2, 2)
        for cc_idx, cc_points in enumerate(points):
            low = 2 if cc_idx == 0 else 6
            high = 4 if cc_idx == 0 else 7
            for point in cc_points:
                assert low <= point[0] <= high and low <= point[0] <= high

    def test_random_points_none(self) -> None:
        assert extract_random_points_from_connected_components(None) is None

    def test_random_points_invalid(self, connected_components: np.ndarray) -> None:
        with pytest.raises(ValueError):
            extract_random_points_from_connected_components(
                connected_components, num_points=0
            )

        with pytest.raises(ValueError):
            extract_random_points_from_connected_components(
                np.zeros((10, 10), dtype=np.uint8)
            )

        with pytest.raises(ValueError):
            extract_random_points_from_connected_components(
                np.zeros((1, 10, 10), dtype=np.uint8)
            )

        with pytest.raises(ValueError):
            extract_random_points_from_connected_components(
                connected_components, num_points=20
            )


class TestPredictSamLogitsFromSingleClick:
    def test_predict_sam_logits_valid(
        self,
        mask: np.ndarray,
        connected_components: np.ndarray,
        mock_sam_model: MagicMock,
        boxes: np.ndarray,
        points: np.ndarray,
    ) -> None:
        img = MagicMock()

        result = predict_sam_logits_from_single_click(
            mock_sam_model, img, boxes, points
        )

        assert result is not None
        assert result.shape == (2, 1, 6, 10, 10)
        assert np.all(result[0, 0, 0:3, 2:5, 2:5] == 1)
        assert np.all(result[0, 0, -3] == 1)
        assert np.all(result[0, 0, -2] == 0.5)
        assert np.all(result[0, 0, -1] == 0.0)
        assert np.all(result[1, 0, 0:3, 6:8, 6:8] == 1)
        assert np.all(result[1, 0, -3] == 1)
        assert np.all(result[1, 0, -2] == 0.5)
        assert np.all(result[1, 0, -1] == 0.0)

    def test_predict_sam_logits_none_boxes(self, mock_sam_model: MagicMock) -> None:
        assert (
            predict_sam_logits_from_single_click(
                mock_sam_model, MagicMock(), None, None
            )
            is None
        )


class TestThresholdSingleClickSamLogits:
    def test_threshold_logits_valid(
        self, sam_logits: np.ndarray, connected_components: np.ndarray
    ) -> None:
        result = threshold_single_click_sam_logits(sam_logits, threshold=0.0)
        assert result is not None
        for sam_mask, cc in zip(result, connected_components, strict=True):
            assert np.all(sam_mask[0] == cc)

    def test_threshold_logits_none(self) -> None:
        assert threshold_single_click_sam_logits(None) is None

    def test_threshold_logits_invalid(self, sam_logits: np.ndarray) -> None:
        with pytest.raises(ValueError):
            threshold_single_click_sam_logits(
                np.zeros((2, 1, 5, 10, 10), dtype=np.float32)
            )

        with pytest.raises(ValueError):
            threshold_single_click_sam_logits(
                np.zeros((2, 1, 6, 10, 10, 1), dtype=np.float32), threshold=-1.0
            )


class TestComputeIousFromSamMasksAndConnectedComponents:
    def test_compute_ious_valid(self, connected_components: np.ndarray) -> None:
        preds_list = []
        for cc_idx, cc in enumerate(connected_components):
            cc_preds = []
            for i in range(2):
                pred = cc.copy()
                if i == 1:
                    if cc_idx == 0:
                        pred[3, 3] = 0
                    else:
                        pred[7, 7] = 0
                cc_preds.append(pred)
            preds_list.append(np.stack(cc_preds, axis=0))

        result = compute_ious_from_sam_masks_and_connected_components(
            connected_components, np.stack(preds_list, axis=0)
        )
        assert result is not None
        assert result.shape == (2, 2)
        for cc_result in result:
            for pred_idx, pred_result in enumerate(cc_result):
                if pred_idx == 0:
                    assert pred_result == 1.0  # Perfect match
                else:
                    assert pred_result < 1.0

    def test_compute_ious_none(self) -> None:
        assert compute_ious_from_sam_masks_and_connected_components(None, None) is None

    def test_compute_ious_shape_mismatch(
        self, connected_components: np.ndarray
    ) -> None:
        with pytest.raises(ValueError):
            compute_ious_from_sam_masks_and_connected_components(
                connected_components,
                None,
            )

        with pytest.raises(ValueError):
            compute_ious_from_sam_masks_and_connected_components(
                None, np.ones((2, 3, 10, 10), dtype=np.uint8)
            )

        with pytest.raises(ValueError):
            compute_ious_from_sam_masks_and_connected_components(
                np.ones((2, 1, 10, 10), dtype=np.uint8),
                np.ones((2, 3, 10, 10), dtype=np.uint8),
            )

        with pytest.raises(ValueError):
            compute_ious_from_sam_masks_and_connected_components(
                connected_components,
                np.ones((2, 3, 10, 10, 1), dtype=np.uint8),
            )

        with pytest.raises(ValueError):
            compute_ious_from_sam_masks_and_connected_components(
                connected_components,
                np.ones((3, 3, 10, 10), dtype=np.uint8),
            )

        with pytest.raises(ValueError):
            compute_ious_from_sam_masks_and_connected_components(
                np.ones((1, 10, 10), dtype=np.uint8),
                np.ones((2, 3, 10, 10), dtype=np.uint8),
            )


class TestPredictSamMasksFromSingleClick:
    def test_predict_masks_basic(
        self, mock_sam_model: MagicMock, boxes: np.ndarray, points: np.ndarray
    ) -> None:
        image = PILImage.new("RGB", (10, 10))
        masks = predict_sam_masks_from_single_click(
            model=mock_sam_model,
            image=image,
            bounding_boxes=boxes,
            random_points=points,
            threshold=0.0,
            use_bounding_box=True,
        )
        assert masks.shape == (2, 1, 10, 10)
        assert (masks[0, 0, 2:5, 2:5] == 1).all()
        assert (masks[1, 0, 6:8, 6:8] == 1).all()

    def test_none_points_returns_none(self, mock_sam_model: MagicMock) -> None:
        image = PILImage.new("RGB", (10, 10))
        masks = predict_sam_masks_from_single_click(
            model=mock_sam_model,
            image=image,
            bounding_boxes=None,
            random_points=None,
        )
        assert masks is None


class TestSamExecutionTimeForSingleClick:
    def test_execution_time_valid(
        self, mock_sam_model: MagicMock, boxes: np.ndarray, points: np.ndarray
    ) -> None:
        img = MagicMock()
        result = sam_execution_time_for_single_click(mock_sam_model, img, boxes, points)
        assert result is not None
        assert set(result.keys()) == {
            "img_transform_and_encoding",
            "mask_prediction",
            "total",
        }
        assert all(isinstance(v, float) for v in result.values())

    def test_execution_time_none_points(
        self, mock_sam_model: MagicMock, boxes: np.ndarray
    ) -> None:
        img = MagicMock()
        result = sam_execution_time_for_single_click(mock_sam_model, img, None, None)
        assert result is None
