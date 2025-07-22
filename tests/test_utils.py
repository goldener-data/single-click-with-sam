from omegaconf import OmegaConf


import src.utils as utils


class TestGetPxtRunName:
    def test_with_run_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"run_name": "run", "dir_name": "dir"},
                "dataset": {"args": {"_target_": "foo.Bar", "split": "train"}},
                "sam": {"name": "sam"},
            }
        )
        assert (
            utils.get_pxt_run_name_for_sam_single_click(cfg)
            == "sam_single_click_dir.run"
        )

    def test_without_run_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"run_name": None, "dir_name": "dir"},
                "dataset": {
                    "args": {
                        "_target_": "foo.Bar",
                        "split": "train",
                    }
                },
                "sam": {"name": "sam"},
            }
        )
        assert (
            utils.get_pxt_run_name_for_sam_single_click(cfg)
            == "sam_single_click_dir.Bar_train"
        )

    def test_without_run_name_but_with_category(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"run_name": None, "dir_name": "dir"},
                "dataset": {
                    "args": {"_target_": "foo.Bar", "split": "train", "category": "cat"}
                },
                "sam": {"name": "sam"},
            }
        )
        assert (
            utils.get_pxt_run_name_for_sam_single_click(cfg)
            == "sam_single_click_dir.Bar_train_cat"
        )


class TestGetPxtTableName:
    def test_with_table_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"table_name": "tbl"},
                "pipeline": {"seed": 1, "k_shots": 2, "num_points": 3, "min_area": 4},
                "model": {"hf_id": "foo/bar.baz-2"},
            }
        )
        assert utils.get_pxt_table_name_for_sam_single_click(cfg, "run") == "run.tbl"

    def test_without_table_name(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {"table_name": None},
                "pipeline": {"seed": 1, "k_shots": 2, "num_points": 3, "min_area": 4},
                "model": {"hf_id": "foo/bar.baz-2"},
            }
        )
        assert (
            utils.get_pxt_table_name_for_sam_single_click(cfg, "run")
            == "run.model_bar_baz_2_seed_1_k_2_n_3_amin_4"
        )


class TestGetPxtTablePath:
    def test_table_path(self) -> None:
        cfg = OmegaConf.create(
            {
                "pixeltable": {
                    "run_name": "run",
                    "dir_name": "dir",
                    "table_name": "tbl",
                },
                "dataset": {"args": {"_target_": "foo.Bar", "split": "train"}},
                "sam": {"name": "sam"},
                "pipeline": {"seed": 1, "k_shots": 2, "num_points": 3, "min_area": 4},
            }
        )
        assert (
            utils.get_pxt_table_path_for_sam_single_click(cfg)
            == "sam_single_click_dir.run.tbl"
        )


class TestStrAsValidPythonIdentifier:
    def test_basic(self) -> None:
        assert utils.str_as_valid_python_identifier("foo-bar") == "foo_bar"
        assert utils.str_as_valid_python_identifier("123abc") == "_123abc"
        assert utils.str_as_valid_python_identifier("class") == "class_"
        assert utils.str_as_valid_python_identifier("") == "_"
        assert utils.str_as_valid_python_identifier("foo@bar!") == "foo_bar_"
        assert utils.str_as_valid_python_identifier("with space") == "with_space"
        assert utils.str_as_valid_python_identifier("def") == "def_"
        assert (
            utils.str_as_valid_python_identifier("_already_valid") == "_already_valid"
        )


class TestCheckBoundingBoxesAndPointsForSam:
    def test_valid_inputs(self) -> None:
        import numpy as np
        from src.utils import check_bounding_boxes_and_points_for_sam

        boxes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        points = np.zeros((2, 3, 2))
        # Should not raise
        check_bounding_boxes_and_points_for_sam(boxes, points)

    def test_none_boxes_and_points(self) -> None:
        from src.utils import check_bounding_boxes_and_points_for_sam

        # Should not raise
        check_bounding_boxes_and_points_for_sam(None, None)

    def test_none_boxes_with_points(self) -> None:
        import numpy as np
        from src.utils import check_bounding_boxes_and_points_for_sam

        points = np.zeros((2, 3, 2))
        import pytest

        with pytest.raises(ValueError):
            check_bounding_boxes_and_points_for_sam(None, points)

    def test_boxes_with_none_points(self) -> None:
        import numpy as np
        from src.utils import check_bounding_boxes_and_points_for_sam

        boxes = np.array([[1, 2, 3, 4]])
        import pytest

        with pytest.raises(ValueError):
            check_bounding_boxes_and_points_for_sam(boxes, None)

    def test_invalid_boxes_shape(self) -> None:
        import numpy as np
        from src.utils import check_bounding_boxes_and_points_for_sam

        boxes = np.array([1, 2, 3, 4])
        points = np.zeros((1, 3, 2))
        import pytest

        with pytest.raises(ValueError):
            check_bounding_boxes_and_points_for_sam(boxes, points)

    def test_invalid_points_shape(self) -> None:
        import numpy as np
        from src.utils import check_bounding_boxes_and_points_for_sam

        boxes = np.array([[1, 2, 3, 4]])
        points = np.zeros((1, 3))
        import pytest

        with pytest.raises(ValueError):
            check_bounding_boxes_and_points_for_sam(boxes, points)

    def test_mismatched_m(self) -> None:
        import numpy as np
        from src.utils import check_bounding_boxes_and_points_for_sam

        boxes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        points = np.zeros((1, 3, 2))
        import pytest

        with pytest.raises(ValueError):
            check_bounding_boxes_and_points_for_sam(boxes, points)
