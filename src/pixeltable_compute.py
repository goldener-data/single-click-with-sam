from logging import getLogger
from typing import Callable, Any, Optional

from omegaconf import DictConfig
import pixeltable as pxt
from pixeltable import catalog
from pixeltable.exprs import Expr

from src.pixeltable_udf import (
    connected_components,
    bounding_boxes,
    random_points,
    masks_from_sam_logits,
    segmentation_ious,
    sam_logits_from_single_click,
)

logger = getLogger(__name__)


def pxt_column_with_error(
    pxt_table: catalog.Table,
    col: Expr,
) -> bool:
    """Check if a column in the pixeltable table has errors.

    An error means that the column has been computed but some entries are missing or have errors.
    """
    return pxt_table.where(col.errormsg != None).count() > 0  # noqa: E711


def compute_pxt_column(
    pxt_table: catalog.Table,
    column_name: str,
    pxt_udf: Callable,
    **kwargs: Any,
) -> None:
    """Compute a column in the pixeltable if it does not exist or contains errors."""
    pxt_columns = pxt_table.columns()
    if column_name not in pxt_columns:
        logger.info(f"Adding {column_name} column to the Pixeltable table")
        pxt_table.add_computed_column(**{column_name: pxt_udf(**kwargs)})
        return

    col = getattr(pxt_table, column_name)
    if pxt_column_with_error(pxt_table, col):
        logger.info(
            f"The {column_name} column already exists but had some errors. "
            "Recomputing the missing entries."
        )
        pxt_table.recompute_columns(
            getattr(pxt_table, column_name),
            errors_only=True,
        )
    else:
        logger.info(f"The {column_name} column is complete, skipping recomputation.")


def compute_data_extraction(
    pxt_table: catalog.Table,
    cfg: DictConfig,
) -> None:
    """Generate columns allowing to access information useful to study the single click segmentation task.

    3 columns are generated:
    - `connected_components`: connected components extracted from the binary mask
    - `bounding_boxes`: bounding boxes for each connected component
    - `random_points`: random points sampled from each connected component. The single click segmentation task
    will be simulated by using these points as clicks for the SAM model.
    """
    # Extract connected components from the binary mask
    # Each object in the mask will then be processed separately
    compute_pxt_column(
        pxt_table,
        "connected_components",
        connected_components,
        mask=pxt_table.mask,
        min_area=cfg.pipeline.min_area,
    )

    # Extract the bounding box for each connected component
    compute_pxt_column(
        pxt_table,
        "bounding_boxes",
        bounding_boxes,
        connected_components=pxt_table.connected_components,
    )

    # For each connected component, SAM will generate multiple segmentation masks based on  a single click.
    # The goal is to simulate the random choice made by an annotator
    compute_pxt_column(
        pxt_table,
        "random_points",
        random_points,
        connected_components=pxt_table.connected_components,
        num_points=cfg.pipeline.num_points,
    )


def compute_segmentation_with_sam(
    pxt_table: catalog.Table,
    cfg: DictConfig,
) -> None:
    """Compute the segmentation using the SAM model.

    3 columns are generated:
    - `sam_logits`: logits predicted by the SAM model for each random point
    - `sam_masks`: segmentation masks extracted from the logits
    - `sam_ious`: IoU between the predicted masks and the ground truth mask
    """

    # Make a prediction with the SAM model using the random points as single clicks.
    # SAM is output logits expressing a score regarding the presence of the same object in each pixel.
    compute_pxt_column(
        pxt_table,
        "sam_logits",
        sam_logits_from_single_click,
        model_id=cfg.model.hf_id,
        image=pxt_table.image,
        boxes=pxt_table.bounding_boxes,
        points=pxt_table.random_points,
    )

    # Keep the most probable segmentation mask from the logits.
    # This means only the mask with the highest predicted IoU will be kept.
    compute_pxt_column(
        pxt_table,
        "sam_masks",
        masks_from_sam_logits,
        sam_logits=pxt_table.sam_logits,
        threshold=cfg.model.threshold,
    )

    # Compute the IoU between the predicted masks and the ground truth mask.
    compute_pxt_column(
        pxt_table,
        "sam_ious",
        segmentation_ious,
        connected_components=pxt_table.connected_components,
        predicted_masks=pxt_table.sam_masks,
    )


@pxt.uda
class mean_sam_iou(pxt.Aggregator):
    """Compute the mean IoU of the SAM model masks.

    Args:
        sam_ious: An optional array of shape (M, num_points) containing the IoU values for each sampled points
        taken for each connected component.
    """

    def __init__(self) -> None:
        self.iou_sum: int = 0
        self.iou_count: int = 0

    def update(self, sam_ious: Optional[pxt.Array]) -> None:
        """Aggregate value from each row of the Pixeltable."""
        if sam_ious is None:
            return

        for box_sam_iou in sam_ious:
            for sam_iou in box_sam_iou:
                self.iou_sum += sam_iou.item()
                self.iou_count += 1

    def value(self) -> Optional[float]:
        """Return the mean IoU."""
        return self.iou_sum / self.iou_count if self.iou_count > 0 else None


def get_mean_sam_iou(
    pxt_table: catalog.Table,
) -> None | float:
    """Get the mean IoU from the SAM model predictions.

    Args:
        pxt_table: The Pixeltable table containing the ious from the sam model.

    Returns: The mean IoU value or None if no IoU values are present.
    """
    mean_iou_expr = pxt_table.select(mean_sam_iou(pxt_table.sam_ious))
    mean_iou = mean_iou_expr.show()[0]["mean_sam_iou"]
    return float(mean_iou) if mean_iou is not None else None
