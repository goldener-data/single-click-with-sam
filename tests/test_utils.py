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
