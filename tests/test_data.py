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
        pxt_table.image = MagicMock()
        pxt_table.label = MagicMock()
        pxt_table.index = MagicMock()
        pxt_table.connected_components = MagicMock()
        pxt_table.bounding_boxes = MagicMock()
        pxt_table.random_points = MagicMock()
        pxt_table.sam_ious = MagicMock()
        pxt_table.sam_logits = MagicMock()
        pxt_table.sam_masks = MagicMock()

        # Mock the image expression to have .col.is_stored and .localpath
        pxt_table.image.col.is_stored = True
        pxt_table.image.localpath = str(tmp_path / "image.png")

        # Mock the DataFrame returned by pxt_table.select
        df_mock = MagicMock()

        # Create a dummy image and save it to the local path
        img = PIL.Image.new("RGB", (10, 10))
        img.save(pxt_table.image.localpath)

        # Prepare a fake row: [image, file, label, index, connected_components, boxes, points, sam_logits, sam_masks,]
        row = [
            img,
            pxt_table.image.localpath,
            "label",
            0,
            np.ones((1, 10, 10)),
            np.array([[2, 2, 7, 7]]),  # bounding box
            np.ones((1, 1, 2)),
            np.ones((1, 1)),
            np.ones((1, 1, 6, 10, 10)),
            np.ones((1, 1, 10, 10)),
        ]
        df_mock._output_row_iterator.return_value = iter([row])
        pxt_table.select.return_value = df_mock

        importer = data.PxtSAMSingleClickDatasetImporter(
            pxt_table=pxt_table,
            show_sam_logits=True,
            show_sam_masks=True,
            tmp_dir=str(tmp_path),
            dataset_dir=None,
        )

        result = next(importer)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], fo.ImageMetadata)
        assert isinstance(result[2], dict)
        assert len(result[2]) == 6
        assert isinstance(result[2]["ground_truth"], fo.Detections)
        assert isinstance(result[2]["random_points_0"], fo.Keypoints)
        assert isinstance(result[2]["sam_logit_0_0_0"], fo.Heatmap)
        assert isinstance(result[2]["sam_logit_0_0_1"], fo.Heatmap)
        assert isinstance(result[2]["sam_logit_0_0_2"], fo.Heatmap)
        assert isinstance(result[2]["sam_mask_0_0"], fo.Segmentation)
