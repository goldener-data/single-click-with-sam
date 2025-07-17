from logging import getLogger

import hydra
from omegaconf import DictConfig

import pixeltable as pxt

from src.utils import (
    get_pxt_dir_name_for_sam_single_click,
    get_ground_truth_labels_for_sam_single_click,
)

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def show_pixeltable_status(cfg: DictConfig) -> None:
    pxt_dir_name = get_pxt_dir_name_for_sam_single_click(
        cfg,
    )
    logger.info(f"Searching runs in {pxt_dir_name}.")
    pxt_run_names = pxt.list_dirs(pxt_dir_name)
    for pxt_run_name in pxt_run_names:
        logger.info(f"Searching tables in {pxt_run_name}.")
        pxt_table_names = pxt.list_tables(pxt_run_name, recursive=False)
        logger.info(f"Found {len(pxt_table_names)} tables in {pxt_run_name}.")
        for pxt_table_name in pxt_table_names:
            pxt_table = pxt.get_table(pxt_table_name)
            size = pxt_table.count()
            table_stats = {
                "size": size,
                "labels": get_ground_truth_labels_for_sam_single_click(pxt_table),
            }
            if size > 0:
                row = pxt_table.sample(1).collect()[0]
                table_stats["num_points"] = row["random_points"].shape[1]

            logger.info(f"Stats for {pxt_table_name}: {table_stats}")


if __name__ == "__main__":
    show_pixeltable_status()
