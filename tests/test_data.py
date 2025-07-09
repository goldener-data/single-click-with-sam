from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import PIL
import fiftyone as fo

import numpy as np
import torch
from PIL.Image import Image
from omegaconf import OmegaConf

from src import data


class TestFormatForPixeltable:
    def test_tensor_image(self) -> None:
        t = torch.ones(3, 8, 8)
        img = data.format_for_sam_single_click_pixeltable(t, is_image=True)
        from PIL.Image import Image

        assert isinstance(img, Image)

    def test_tensor_scalar(self) -> None:
        t = torch.tensor(5)
        val = data.format_for_sam_single_click_pixeltable(t, is_image=False)
        assert val == 5

    def test_tensor_array(self) -> None:
        t = torch.arange(6).reshape(2, 3)
        arr = data.format_for_sam_single_click_pixeltable(t, is_image=False)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3)

    def test_other(self) -> None:
        val = data.format_for_sam_single_click_pixeltable(42, is_image=False)
        assert val == 42


class TestImportAnyDatasetToPixeltable:
    def test_import(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {
                    "run_name": "run",
                    "dir_name": "dir",
                    "table_name": "tbl",
                },
                "dataset": {
                    "label": None,
                    "label_col": None,
                    "args": {"_target_": "foo.Bar", "split": "train"},
                },
                "sam": {"name": "sam"},
                "pipeline": {"seed": 1, "k_shots": 2, "num_points": 3, "min_area": 4},
            }
        )
        dataset = MagicMock()
        dataset.__len__.return_value = 2
        dataset.get_raw.side_effect = lambda idx: {
            "image": torch.ones(3, 8, 8),
            "mask": torch.ones(1, 8, 8),
            "label": "foo",
            "index": idx,
        }

        table = MagicMock()
        table.inserted = []
        table._columns = set()
        table.columns.return_value = table._columns
        table.add_columns.side_effect = lambda d: table._columns.update(d.keys())

        def insert_side_effect(**kwargs: Any) -> None:
            table.inserted.append(kwargs)

        table.insert.side_effect = insert_side_effect

        data.import_data_in_table_for_sam_single_click(
            cfg, dataset, table, num_samples=2, batch_size=1, num_workers=0
        )
        assert len(table.inserted) == 2

        for row in table.inserted:
            assert isinstance(row["image"], Image)
            assert np.all(row["mask"] == np.ones((8, 8)))
            assert row["label"] == "foo"


class TestPxtSAMSingleClickDatasetImporter:
    def test_importer_next(self, tmp_path: Path) -> None:
        # Setup mocks for pxt_table and exprs
        pxt_table = MagicMock()
        image_expr = MagicMock()
        label_expr = MagicMock()
        index_expr = MagicMock()
        connected_components_expr = MagicMock()
        bounding_boxes_expr = MagicMock()
        random_points_expr = MagicMock()
        sam_logits_expr = MagicMock()
        sam_masks_expr = MagicMock()

        # Mock the image expression to have .col.is_stored and .localpath
        image_expr.col.is_stored = True
        image_expr.localpath = str(tmp_path / "image.png")

        # Mock the DataFrame returned by pxt_table.select
        df_mock = MagicMock()

        # Create a dummy image and save it to the local path
        img = PIL.Image.new("RGB", (10, 10))
        img.save(image_expr.localpath)

        # Prepare a fake row: [image, file, label, index, connected_components, boxes, points, sam_logits, sam_masks,]
        row = [
            img,
            image_expr.localpath,
            "label",
            0,
            np.ones((1, 10, 10)),
            np.array([[2, 2, 7, 7]]),  # bounding box
            np.ones((1, 1, 2)),
            np.ones((1, 1, 6, 10, 10)),
            np.ones((1, 1, 10, 10)),
        ]
        df_mock._output_row_iterator.return_value = iter([row])
        pxt_table.select.return_value = df_mock

        importer = data.PxtSAMSingleClickDatasetImporter(
            pxt_table=pxt_table,
            image=image_expr,
            index=index_expr,
            label=label_expr,
            connected_components=connected_components_expr,
            bounding_boxes=bounding_boxes_expr,
            random_points=random_points_expr,
            sam_logits=sam_logits_expr,
            sam_masks=sam_masks_expr,
            tmp_dir=str(tmp_path),
            dataset_dir=None,
        )

        result = next(importer)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], fo.ImageMetadata)
        assert isinstance(result[2], dict)
        assert len(result[2]) == 9
        assert isinstance(result[2]["ground_truth"], fo.Detections)
        assert isinstance(result[2]["random_points_0"], fo.Keypoints)
        assert isinstance(result[2]["sam_logit_0_0_0"], fo.Heatmap)
        assert isinstance(result[2]["sam_logit_0_0_1"], fo.Heatmap)
        assert isinstance(result[2]["sam_logit_0_0_2"], fo.Heatmap)
        assert isinstance(result[2]["sam_iou_0_0_0"], fo.Classification)
        assert isinstance(result[2]["sam_iou_0_0_1"], fo.Classification)
        assert isinstance(result[2]["sam_iou_0_0_2"], fo.Classification)
        assert isinstance(result[2]["sam_mask_0_0"], fo.Segmentation)
