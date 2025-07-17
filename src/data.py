from __future__ import annotations

import os

from typing import Any, Optional

import cv2
import fiftyone as fo

import numpy as np
import puremagic
import torch
from PIL.Image import Image

from any_gold import AnyVisionSegmentationDataset, AnyRawDataset
from fiftyone.utils import data as foud
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import RandomSampler, DataLoader
from torchvision.transforms.v2 import functional as F


import pixeltable as pxt
from pixeltable import catalog
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
    dataset_args = cfg.dataset.args
    logger.info(f"Loading the dataset: {dataset_args._target_}")
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
    """FiftyOne `DatasetImporter` loading data from a Pixeltable table during sam single click experiment.

    The importer will use the Pixeltable table to create FiftyOne samples:
    - connected_components and bounding_boxes are converted to FiftyOne Detections
    - random_points are converted to FiftyOne Keypoints
    - sam_logits are converted to FiftyOne Heatmaps with predicted iou as attribute
    - sam_masks are converted to FiftyOne Segmentations
    - sam_ious are used to set the confidence of the ground_truth (allow easy visualization) and added as iou value
     for detections, keypoints, segmentations.
    """

    def __init__(
        self,
        pxt_table: pxt.Table,
        show_sam_logits: bool,
        show_sam_masks: bool,
        tmp_dir: str,
        dataset_dir: Optional[os.PathLike] = None,
    ):
        super().__init__(
            dataset_dir=dataset_dir, shuffle=None, seed=None, max_samples=None
        )

        self.tmp_dir = tmp_dir

        inputs = (
            {
                "image": pxt_table.image,
                "file": pxt_table.image.localpath,
                "label": pxt_table.label,
                "index": pxt_table.index,
                "connected_components": pxt_table.connected_components,
                "bounding_boxes": pxt_table.bounding_boxes,
                "random_points": pxt_table.random_points,
                "sam_ious": pxt_table.sam_ious,
            }
            | (
                {
                    "sam_logits": pxt_table.sam_logits,
                }
                if show_sam_logits
                else {}
            )
            | (
                {
                    "sam_masks": pxt_table.sam_masks,
                }
                if show_sam_masks
                else {}
            )
        )

        df = pxt_table.select(*[i for i in inputs.values()])

        self.input_positions = {
            input_type: pos for pos, input_type in enumerate(inputs)
        }
        self._row_iter = (
            df._output_row_iterator()
        )  # iterator over the table rows, to be converted to FiftyOne samples

    def __next__(self) -> tuple[str, fo.ImageMetadata, dict[str, fo.Label]]:
        """Access the next row in the Pixeltable and return infos to create a FiftyOne sample.

        Each element in the pixeltable row is converted to a FiftyOne object.
        The label is used to tag all the FiftyOne objects.
        """
        row = next(self._row_iter)
        img = row[self.input_positions["image"]]
        img_size = img.size
        file = row[self.input_positions["file"]]
        label = row[self.input_positions["label"]]
        index = row[self.input_positions["index"]]
        connected_components = row[self.input_positions["connected_components"]]
        bounding_boxes = row[self.input_positions["bounding_boxes"]]
        random_points = row[self.input_positions["random_points"]]
        sam_ious = row[self.input_positions["sam_ious"]]
        sam_logits: None | np.ndarray = (
            row[self.input_positions["sam_logits"]]
            if "sam_logits" in self.input_positions
            else None
        )
        sam_masks: None | np.ndarray = (
            row[self.input_positions["sam_masks"]]
            if "sam_masks" in self.input_positions
            else None
        )

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
        if connected_components is not None:
            labels = (
                self.ground_truth_to_fo(
                    connected_components=connected_components,
                    bounding_boxes=bounding_boxes,
                    sam_ious=sam_ious,
                    label=label,
                    index=index,
                    save_dir=self.tmp_dir,
                    img_size=img_size,
                )
                | self.random_points_to_fo(
                    random_points=random_points,
                    sam_ious=sam_ious,
                    img_size=img_size,
                )
                | (
                    self.sam_logits_to_fo(
                        sam_logits=sam_logits,
                        index=index,
                        save_dir=self.tmp_dir,
                    )
                    if sam_logits is not None
                    else {}
                )
                | (
                    self.sam_masks_to_fo(
                        sam_masks=sam_masks,
                        sam_ious=sam_ious,
                        index=index,
                        save_dir=self.tmp_dir,
                    )
                    if sam_masks is not None
                    else {}
                )
            )

        return (
            file,
            metadata,
            labels,
        )

    @staticmethod
    def ground_truth_to_fo(
        connected_components: np.ndarray | None,
        bounding_boxes: np.ndarray | None,
        sam_ious: np.ndarray | None,
        label: str,
        index: int,
        save_dir: str,
        img_size: tuple[int, int],
    ) -> dict[str, fo.Detections] | None:
        """Convert connected components and bounding boxes to FiftyOne segmentations.

        Args:
            connected_components: An optional array of shape (M, H, W) corresponding to the connected component masks.
            bounding_boxes: An optional array of shape (M, 4) where M is the number of boxes and each box is represented by its
            (left, top, right, bottom) coordinates.
            sam_ious: An optional array of shape (M, N) where M is the number of boxes and N is the number of points per box,
            label: The label to assign to the detections.
            index: The index of the sample in the dataset.
            save_dir: Directory to save the masks as images.
            img_size: A tuple representing the size of the image (width, height).

        Returns: A dictionary with a Detections object for each connected component and bounding boxes.
        The confidence is set to the minimum IoU value for each box. None if no connected components are provided.
        """
        if connected_components is None:
            return None

        w, h = img_size
        detections = []
        for idx_truth, (connected_component, bounding_box, box_sam_ious) in enumerate(
            zip(connected_components, bounding_boxes, sam_ious, strict=True)
        ):
            left, top, right, bottom = bounding_box
            mask = connected_component[top:bottom, left:right]
            mask_path = f"{save_dir}/{index}_{label}_{idx_truth}.png"
            cv2.imwrite(mask_path, mask)
            detections.append(
                fo.Detection(
                    mask_path=mask_path,
                    bounding_box=[
                        left / w,  # left
                        top / h,  # top
                        (right - left) / w,  # width
                        (bottom - top) / h,  # height
                    ],
                    label=label,
                    confidence=box_sam_ious.min(),
                )
            )

        return {"ground_truth": fo.Detections(detections=detections)}

    @staticmethod
    def random_points_to_fo(
        random_points: np.ndarray | None,
        sam_ious: np.ndarray | None,
        img_size: tuple[int, int],
    ) -> dict[str, fo.Keypoints] | None:
        """Convert random points to FiftyOne keypoints.

        Args:
            random_points: An optional array of shape (M, N, 2) where M is the number of boxes, N is the number of points per box,
            and each point is represented by its (x, y) coordinates.
            sam_ious: An optional array of shape (M, N) where M is the number of boxes and N is the number of points per box,
            img_size: A tuple representing the size of the image (width, height).

        Returns: A dictionary with a Keypoints object for each connected component, with coordinates normalized to the image size.
        """
        if random_points is None:
            return None

        w, h = img_size
        points_per_box = {}
        for idx_box, (box_points, box_ious) in enumerate(
            zip(random_points, sam_ious, strict=True)
        ):
            points = []
            for idx_point, (point, iou) in enumerate(
                zip(box_points, box_ious, strict=True)
            ):
                label = f"random_point_{idx_box}_{idx_point}"
                points.append(
                    fo.Keypoint(
                        points=[(point[0] / w, point[1] / h)],  # x, y
                        label=label,
                        iou=iou.item(),
                    )
                )
            points_per_box[f"random_points_{idx_box}"] = fo.Keypoints(keypoints=points)

        return points_per_box

    @staticmethod
    def sam_logits_to_fo(
        sam_logits: np.ndarray | None,
        index: int,
        save_dir: str,
    ) -> dict[str, fo.Heatmap | fo.Classification] | None:
        """Convert SAM logits to FiftyOne heatmaps and classifications.

        Args:
            sam_logits: An optional array of shape (M, N, 6, H, W) where M is the number of boxes,
            N is the number of points per box, and each point has 6 logits (3 for the heatmaps and 3 for the ious).
            index: The index of the sample in the dataset.
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

        heatmaps = {}
        for idx_box, box_logits in enumerate(sam_logits):
            for idx_point, logits in enumerate(box_logits):
                for idx_logits in range(3):
                    predicted_iou = logits[idx_logits + 3].max().item()
                    label_logit = f"sam_logit_{idx_box}_{idx_point}_{idx_logits}"

                    heatmap = make_heatmap(logits[idx_logits])
                    map_path = f"{save_dir}/{index}_{label_logit}.png"
                    cv2.imwrite(map_path, heatmap)

                    heatmaps[label_logit] = fo.Heatmap(
                        map_path=map_path,
                        range=[0, 255],
                        label=label_logit,
                        predicted_iou=predicted_iou,
                    )

        return heatmaps

    @staticmethod
    def sam_masks_to_fo(
        sam_masks: None | np.ndarray,
        sam_ious: None | np.ndarray,
        index: int,
        save_dir: str,
    ) -> dict[str, fo.Segmentation] | None:
        """Convert SAM masks to FiftyOne segmentations.

        Args:
            sam_masks: An optional array of shape (M, N, H, W) where M is the number of boxes,
            N is the number of points per box, and each point has a binary mask of shape (H, W).
            sam_ious: An optional array of shape (M, N) where M is the number of boxes and N is the number of points per box,
            index: The index of the sample in the dataset.
            save_dir: Directory to save the masks as images.

        Returns: A dictionary with a Segmentation object for each connected components.
        None if no SAM masks are provided.
        """
        if sam_masks is None:
            return None

        segmentations = {}
        for idx_box, (box_mask, box_ious) in enumerate(
            zip(sam_masks, sam_ious, strict=True)
        ):
            for idx_point, (point_mask, point_iou) in enumerate(
                zip(box_mask, box_ious, strict=True)
            ):
                label = f"sam_mask_{idx_box}_{idx_point}"
                mask_path = f"{save_dir}/{index}_{label}.png"
                cv2.imwrite(mask_path, point_mask)
                segmentations[label] = fo.Segmentation(
                    mask_path=mask_path,
                    label=label,
                    iou=point_iou.item(),
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
