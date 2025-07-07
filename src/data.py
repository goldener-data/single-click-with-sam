from __future__ import annotations

import os
from functools import partial

from typing import Any, Optional, Callable

import cv2
import fiftyone as fo

import numpy as np
import puremagic
import torch
from PIL.Image import Image

from any_gold import AnyVisionSegmentationDataset, AnyRawDataset
from bson import ObjectId
from fiftyone.utils import data as foud
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import RandomSampler, DataLoader
from torchvision.transforms.v2 import functional as F


import pixeltable as pxt
from pixeltable import catalog, exprs
from pixeltable.type_system import ColumnType

from src.utils import logger


def load_segmentation_dataset_for_sam_single_click(
    cfg: DictConfig,
) -> AnyVisionSegmentationDataset:
    """Load the Segmentation dataset for the single click experiment.

    Args:
        cfg: Configuration specifying the dataset to load.

    Returns: the segmentation dataset specified in configuration.
    """
    dataset_args = dict(cfg.dataset.args)
    logger.info(f"Loading the dataset: {cfg.dataset.args._target_}")
    dataset = instantiate(dataset_args, transforms=None)

    return dataset


def format_for_sam_single_click_pixeltable(value: Any, is_image: bool) -> Any:
    """Format each sample of a dataset into expected format from Pixeltable table."""
    if isinstance(value, torch.Tensor):
        if is_image:
            return F.to_pil_image(value.cpu())
        else:
            if value.dim() == 0:
                return value.item()

            return value.numpy().astype(np.uint8)
    else:
        return value


def import_data_in_table_for_sam_single_click(
    cfg: DictConfig,
    dataset: AnyVisionSegmentationDataset,
    pxt_table: catalog.Table,
    num_samples: int | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> None:
    """Load the data from the dataset inside the Pixeltable table for the single click experiments.

    Only data not present in the table are added.
    The label value or column might be selected from the conifuration. The value
    is prioritary compared to the column.

    Args:
        cfg: Configuration specifying the label or label column if needed.
        dataset: The any_gold segmentation dataset to import from.
        pxt_table: The PixelTable table to import the data into.
        num_samples: the number of sample to include from the dataset.
        if None, the full dataset will be added.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses to use for data loading.
    """
    # Check if the data is already there
    already_loaded = [
        dataset_idx for dataset_idx in pxt_table.select(pxt_table.index).collect()
    ]
    expected_size = num_samples if num_samples is not None else len(dataset)
    if len(already_loaded) == expected_size:
        logger.info("All data is already in the pixeltable table")
        return

    # set a label value or column is required (specified label is prioritary on specified column to find label)
    if cfg.dataset.label is not None:
        label = cfg.dataset.label
    elif cfg.dataset.label_col is not None:
        label = cfg.dataset.label_col
    else:
        label = None

    # add the data in the Pixeltable table
    logger.info("Adding the data of dataset into the Pixeltable table")
    raw_dataset = AnyRawDataset(dataset)
    sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples)
    dataloader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )
    for batch in dataloader:
        for sample in batch:
            # skip already added data
            if sample["index"] in already_loaded:
                continue

            # format sample to the pixeltable table format and insert it
            to_insert = {}
            for key, value in sample.items():
                value_for_pxt = format_for_sam_single_click_pixeltable(
                    value, key == "image"
                )
                # if the label is a column name, rename this column `label`
                if key == label:
                    key = "label"

                # add missing column from table definition
                if key not in pxt_table.columns():
                    pxt_table.add_columns(
                        {
                            key: ColumnType.from_python_type(
                                type(value_for_pxt),
                                nullable_default=True,
                                allow_builtin_types=True,
                            )
                        }
                    )
                if key == "mask":
                    value_for_pxt = np.squeeze(value_for_pxt).astype(np.int64)
                to_insert[key] = value_for_pxt

            # missing label column mean the label should have the specified value
            if "label" not in to_insert:
                to_insert["label"] = label

            pxt_table.insert(**to_insert)


class PxtSAMSingleClickDatasetImporter(foud.LabeledImageDatasetImporter):
    """
    Implementation of a FiftyOne `DatasetImporter` that reads image data from a Pixeltable table.
    """

    def __init__(
        self,
        pxt_table: pxt.Table,
        image: exprs.Expr,
        label: exprs.Expr,
        connected_components: exprs.Expr,
        bounding_boxes: exprs.Expr,
        random_points: exprs.Expr,
        sam_logits: exprs.Expr | None,
        sam_masks: exprs.Expr | None,
        tmp_dir: str,
        dataset_dir: Optional[os.PathLike] = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir, shuffle=None, seed=None, max_samples=None
        )
        self._labels = {
            "connected_components": (connected_components, fo.Segmentation),
            "bounding_boxes": (bounding_boxes, fo.Detection),
            "random_points": (random_points, fo.Keypoint),
        }
        if sam_logits is not None:
            self._labels["sam_logits"] = (sam_logits, fo.Heatmap)

        if sam_masks is not None:
            self._labels["sam_masks"] = (sam_masks, fo.Segmentation)

        self.tmp_dir = tmp_dir

        selection = [image, image.localpath, label] + [
            expr for expr, _ in self._labels.values()
        ]

        df = pxt_table.select(*selection)
        self._row_iter = (
            df._output_row_iterator()
        )  # iterator over the table rows, to be convered to FiftyOne samples

    def _get_conversion_func(
        self,
        label_name: str,
        image_size: tuple[int, int],
    ) -> Callable:
        """Get the conversion function for a specific label name."""
        if label_name == "connected_components":
            return self.connected_components_to_fo
        elif label_name == "bounding_boxes":
            return partial(
                self.bounding_boxes_to_fo,
                img_size=image_size,
            )
        elif label_name == "random_points":
            return partial(
                self.random_points_to_fo,
                img_size=image_size,
            )
        elif label_name == "sam_logits":
            return partial(
                self.sam_logits_to_fo,
                save_dir=self.tmp_dir,
            )
        elif label_name == "sam_masks":
            return self.sam_masks_to_fo
        else:
            raise ValueError(f"Unknown label name: {label_name}. ")

    def __next__(self) -> tuple[str, fo.ImageMetadata, dict[str, fo.Label]]:
        """Access the next row in the Pixeltable and return infos to create a FiftyOne sample.

        Each element in the pixeltable row is converted to a FiftyOne object.
        The label is used to tag all the FiftyOne objects.
        """
        row = next(self._row_iter)
        img = row[0]
        file = row[1]
        label = row[2]

        assert isinstance(img, Image), "Image data must be a PIL Image"
        metadata = fo.ImageMetadata(
            size_bytes=os.path.getsize(file),
            mime_type=puremagic.from_file(file, mime=True),
            width=img.width,
            height=img.height,
            filepath=file,
            num_channels=len(img.getbands()),
        )

        labels: dict[str, fo.Label] = {}
        for idx, (label_name, (_, label_cls)) in enumerate(
            self._labels.items(),
            start=3,  # the 2 first columns are the image, path and the label
        ):
            label_data = row[idx]
            label_fo = self._get_conversion_func(label_name, img.size)(label_data)

            if label_fo is not None:
                labels.update(**label_fo)

        for label_fo in labels.values():
            label_fo.tags.append(label)

        return (
            file,
            metadata,
            labels,
        )

    @staticmethod
    def connected_components_to_fo(
        connected_components: np.ndarray,
    ) -> dict[str, fo.Segmentation] | None:
        """Convert connected components masks to FiftyOne segmentations.

        Args:
            connected_components: An optional array of shape (M, H, W) corresponding to the connected component masks.

        Returns: A dictionary with a Segmentation object for each connected component.
        None if no connected components are provided.
        """
        if connected_components is None:
            return None

        segmentations = {}
        for idx_mask, mask in enumerate(connected_components):
            label = f"ground_truth_{idx_mask}"
            segmentations[label] = fo.Segmentation(
                mask=mask,
                label=label,
            )

        return segmentations

    @staticmethod
    def bounding_boxes_to_fo(
        bounding_boxes: np.ndarray | None, img_size: tuple[int, int]
    ) -> dict[str, fo.Detection] | None:
        """Convert bounding boxes to FiftyOne detections.

        Args:
            bounding_boxes: An optional array of bounding boxes with shape (M, 4), where each box is represented
            with coordinates (x1, y1, x2, y2) corresponding to the left(x)-top(y) and right(x)-bottom(y) corners.
            img_size: A tuple representing the size of the image (width, height).

        Returns: A dictionary with a Detection object for each bounding box, with coorindates normalized to the image size.
        None if no bounding boxes are provided.
        """
        if bounding_boxes is None:
            return None

        w, h = img_size
        detections = {}
        for idx_box, box in enumerate(bounding_boxes):
            label = f"bounding_box_{idx_box}"
            detections[label] = fo.Detection(
                label=label,
                bounding_box=[
                    box[0] / w,  # left
                    box[1] / h,  # top
                    (box[2] - box[0]) / w,  # width
                    (box[3] - box[1]) / h,  # height
                ],
            )

        return detections

    @staticmethod
    def random_points_to_fo(
        random_points: np.ndarray | None, img_size: tuple[int, int]
    ) -> dict[str, fo.Keypoint] | None:
        """Convert random points to FiftyOne keypoints.

        Args:
            random_points: An optional array of shape (M, N, 2) where M is the number of boxes, N is the number of points per box,
            and each point is represented by its (x, y) coordinates.
            img_size: A tuple representing the size of the image (width, height).

        Returns: A dictionary with a Keypoint object for each point, with coordinates normalized to the image size.
        None if no random points are provided.
        """
        if random_points is None:
            return None

        w, h = img_size
        points = {}
        for idx_box, box_points in enumerate(random_points):
            for idx_point, point in enumerate(box_points):
                label = f"random_point_{idx_box}_{idx_point}"
                points[label] = fo.Keypoint(
                    points=[(point[0] / w, point[1] / h)],  # x, y
                    label=label,
                )

        return points

    @staticmethod
    def sam_logits_to_fo(
        sam_logits: np.ndarray | None,
        save_dir: str,
    ) -> dict[str, fo.Heatmap | fo.Classification] | None:
        """Convert SAM logits to FiftyOne heatmaps and classifications.

        Args:
            sam_logits: An optional array of shape (M, N, 6, H, W) where M is the number of boxes,
            N is the number of points per box, and each point has 6 logits (3 for the heatmaps and 3 for the ious).
            save_dir: Directory to save the heatmaps as images.

        Returns: A dictionary with heatmaps and classifications for each logits and ious.
        None if no SAM logits are provided.
        """
        if sam_logits is None:
            return None

        def make_heatmap(arr: np.ndarray) -> np.ndarray:
            """Normalize the array to the range [0, 1]."""
            min_val = arr.min()
            max_val = arr.max()

            heatmap = (255 * ((arr - min_val) / (max_val - min_val + 1e-8))).astype(
                np.uint8
            )
            assert isinstance(heatmap, np.ndarray)
            return heatmap

        heatmaps_and_classifications = {}
        for idx_box, box_logits in enumerate(sam_logits):
            for idx_point, logits in enumerate(box_logits):
                for idx_logits in range(3):
                    label_iou = f"sam_iou_{idx_box}_{idx_point}_{idx_logits}"
                    heatmaps_and_classifications[label_iou] = fo.Classification(
                        label=label_iou,
                        confidence=logits[idx_logits + 3].max(),
                    )

                    label_logit = f"sam_logit_{idx_box}_{idx_point}_{idx_logits}"
                    heatmap = make_heatmap(logits[idx_logits])
                    map_path = f"{save_dir}/{label_logit}_{ObjectId()}.png"
                    cv2.imwrite(map_path, heatmap)
                    heatmaps_and_classifications[label_logit] = fo.Heatmap(
                        map_path=map_path,
                        range=[0, 255],
                        label=label_logit,
                    )

        return heatmaps_and_classifications

    @staticmethod
    def sam_masks_to_fo(
        sam_masks: None | np.ndarray,
    ) -> dict[str, fo.Segmentation] | None:
        """Convert SAM masks to FiftyOne segmentations.

        Args:
            sam_masks: An optional array of shape (M, N, H, W) where M is the number of boxes,
            N is the number of points per box, and each point has a binary mask of shape (H, W).

        Returns: A dictionary with a Segmentation object for each mask.
        None if no SAM masks are provided.
        """
        if sam_masks is None:
            return None

        segmentations = {}
        for idx_box, box_mask in enumerate(sam_masks):
            for idx_point, point_mask in enumerate(box_mask):
                label = f"sam_mask_{idx_box}_{idx_point}"
                segmentations[label] = fo.Segmentation(
                    mask=point_mask,
                    label=label,
                )

        return segmentations

    @property
    def label_cls(self) -> dict[str, type[fo.Label]]:
        return {
            label_name: label_cls for label_name, (_, label_cls) in self._labels.items()
        }

    @property
    def has_dataset_info(self) -> bool:
        return False

    @property
    def has_image_metadata(self) -> bool:
        return True

    def setup(self) -> None:
        pass

    def get_dataset_info(self) -> dict:
        pass

    def close(self, *args: Any) -> None:
        pass
