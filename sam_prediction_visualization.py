from logging import getLogger
from tempfile import TemporaryDirectory

import hydra
from omegaconf import DictConfig

import pixeltable as pxt
import fiftyone as fo

from src.utils import (
    get_pxt_run_name_for_sam_single_click,
    get_pxt_table_path_for_sam_single_click,
)
from src.data import PxtSAMSingleClickDatasetImporter

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def show_in_fiftyone(cfg: DictConfig) -> None:
    pxt_run_name = get_pxt_run_name_for_sam_single_click(
        cfg,
    )
    existing_pxt_tables = pxt.list_tables(pxt_run_name, recursive=False)

    pxt_table_name = get_pxt_table_path_for_sam_single_click(cfg)
    if pxt_table_name not in existing_pxt_tables:
        raise ValueError(
            f"PixelTable {pxt_table_name} does not exist. Please run the experiment first."
        )

    logger.info(f"Loading PixelTable: {pxt_table_name}")
    pxt_table = pxt.get_table(pxt_table_name)

    with TemporaryDirectory() as tmp_dir:
        logger.info(
            f"Using temporary directory: {tmp_dir} to create custom fiftyone importer"
        )
        importer = PxtSAMSingleClickDatasetImporter(
            pxt_table=pxt_table,
            show_sam_logits=cfg.visualization.show_sam_logits,
            show_sam_masks=cfg.visualization.show_sam_mask,
            tmp_dir=tmp_dir,
        )
        fo_dataset = fo.Dataset.from_importer(importer)

        logger.info("Running fiftyone session")
        session = fo.launch_app(fo_dataset)
        session.wait()


if __name__ == "__main__":
    show_in_fiftyone()
