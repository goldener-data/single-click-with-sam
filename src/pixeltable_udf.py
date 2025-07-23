from typing import Optional

from PIL.Image import Image

import pixeltable as pxt

from src.model import sam_cache
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


@pxt.udf
def connected_components(
    mask: pxt.Array, min_area: int
) -> Optional[pxt.Array[pxt.Int]]:
    """Find the connected components in the binary masks.

    see `utils.extract_connected_components_from_mask` for more details.
    """
    return extract_connected_components_from_binary_mask(mask=mask, min_area=min_area)


@pxt.udf
def bounding_boxes(connected_components: Optional[pxt.Array]) -> Optional[pxt.Array]:
    """Calculate bounding boxes for the connected components.

    see `utils.extract_bounding_boxes_from_connected_components` for more details.
    """
    return extract_bounding_boxes_from_connected_components(
        connected_components=connected_components
    )


@pxt.udf
def random_points(
    connected_components: Optional[pxt.Array],
    num_points: int = 1,
) -> Optional[pxt.Array]:
    """Sample random points within the connected components.

    See `utils.extract_random_points_from_connected_components` for more details.
    """
    return extract_random_points_from_connected_components(
        connected_components=connected_components,
        num_points=num_points,
    )


@pxt.udf
def sam_logits_from_single_click(
    model_id: str,
    image: Image,
    bounding_boxes: Optional[pxt.Array],
    random_points: Optional[pxt.Array],
    use_bounding_box: bool = True,
) -> Optional[pxt.Array]:
    """Predict logits and iou predictions after a single click using the SAM model.

    Args:
        model_id: The identifier of the SAM model to use.
        see `utils.predict_sam_logits_from_single_click for other arguments

    See `utils.predict_sam_logits_from_single_click` for more details.
    """
    if model_id not in sam_cache:
        raise ValueError(
            f"Model with id {model_id} is not loaded. Please load the sam first."
        )

    model = sam_cache[model_id]

    return predict_sam_logits_from_single_click(
        model=model,
        image=image,
        bounding_boxes=bounding_boxes,
        random_points=random_points,
        use_bounding_box=use_bounding_box,
    )


@pxt.udf
def sam_masks_from_single_click(
    model_id: str,
    image: Image,
    bounding_boxes: Optional[pxt.Array],
    random_points: Optional[pxt.Array],
    threshold: float = 0.0,
    use_bounding_box: bool = True,
) -> Optional[pxt.Array]:
    """Predict binary masks after a single click using the SAM model.

    Args:
        model_id: The identifier of the SAM model to use.
        see `utils.predict_sam_masks_from_single_click` for other arguments

    See `utils.predict_sam_masks_from_single_click` for more details.
    """
    if model_id not in sam_cache:
        raise ValueError(
            f"Model with id {model_id} is not loaded. Please load the sam first."
        )

    model = sam_cache[model_id]

    return predict_sam_masks_from_single_click(
        model=model,
        image=image,
        bounding_boxes=bounding_boxes,
        random_points=random_points,
        threshold=threshold,
        use_bounding_box=use_bounding_box,
    )


@pxt.udf
def masks_from_sam_logits(
    sam_logits: Optional[pxt.Array],
    threshold: float = 0.0,
) -> Optional[pxt.Array]:
    """Threshold the SAM logits to create binary masks.

    See `utils.threshold_single_click_sam_logits` for more details.
    """
    return threshold_single_click_sam_logits(
        sam_logits=sam_logits,
        threshold=threshold,
    )


@pxt.udf
def sam_segmentation_ious(
    connected_components: Optional[pxt.Array],
    sam_masks: Optional[pxt.Array],
) -> Optional[pxt.Array]:
    """Calculate the Intersection over Union (IoU) for the connected components and predicted masks.

    see `utils.compute_ious_from_sam_masks_and_connected_components` for more details.
    """
    return compute_ious_from_sam_masks_and_connected_components(
        connected_components=connected_components,
        sam_masks=sam_masks,
    )


@pxt.udf
def sam_execution_time(
    model_id: str,
    image: Image,
    bounding_boxes: Optional[pxt.Array],
    random_points: Optional[pxt.Array],
    use_bounding_box: bool = True,
) -> Optional[dict[str, float]]:
    """Measure the execution time of the SAM model mask prediction.

    Args:
        model_id: The identifier of the SAM model to use.
        see `utils.sam_execution_time_for_single_click` for other arguments

    Returns:
        An optional dictionary containing the execution time in milliseconds for image encoding (including transformation),
        binary mask prediction and full pass. None if random_points is None.
    """
    if model_id not in sam_cache:
        raise ValueError(
            f"Model with id {model_id} is not loaded. Please load the sam first."
        )

    model = sam_cache[model_id]

    return sam_execution_time_for_single_click(
        model=model,
        image=image,
        bounding_boxes=bounding_boxes,
        random_points=random_points,
        use_bounding_box=use_bounding_box,
    )
