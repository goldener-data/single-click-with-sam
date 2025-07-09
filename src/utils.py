from __future__ import annotations

import os
import random
from logging import getLogger

import numpy as np
import torch
from omegaconf import DictConfig

import pixeltable as pxt
from pixeltable import catalog


logger = getLogger(__name__)

SEED = int(os.environ.get("SEED", 42))


def force_seed(seed: int = SEED) -> None:
    """Force the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_pxt_dir_name_for_sam_single_click(cfg: DictConfig) -> str:
    """Generate a unique experiment name for the PixelTable directory based on the configuration."""
    if cfg.pixeltable.dir_name is None:
        return "sam_single_click"

    return f"sam_single_click_{cfg.pixeltable.dir_name}"


def get_pxt_run_name_for_sam_single_click(cfg: DictConfig) -> str:
    """Generate a unique name for the PixelTable directory for the run based on the configuration."""
    dir_name = get_pxt_dir_name_for_sam_single_click(cfg)

    if cfg.pixeltable.run_name is not None:
        return f"{dir_name}.{cfg.pixeltable.run_name}"

    dataset = cfg.dataset.args._target_.split(".")[-1]
    split = cfg.dataset.args.split

    run_name = f"{dir_name}.{dataset}_{split}"
    if "category" in cfg.dataset.args:
        run_name += f"_{cfg.dataset.args.category}"

    return run_name


def get_pxt_table_name_for_sam_single_click(
    cfg: DictConfig,
    run_name: str,
) -> str:
    """Generate a unique name for the PixelTable table based on the configuration."""
    table_name = (
        cfg.pixeltable.table_name
        if cfg.pixeltable.table_name is not None
        else f"seed_{cfg.pipeline.seed}_k_{cfg.pipeline.k_shots}_n_{cfg.pipeline.num_points}_amin_{cfg.pipeline.min_area}"
    )

    return f"{run_name}.{table_name}"


def get_pxt_table_path_for_sam_single_click(
    cfg: DictConfig,
) -> str:
    """Generate the path to the PixelTable table based on the configuration."""
    run_name = get_pxt_run_name_for_sam_single_click(cfg)
    return get_pxt_table_name_for_sam_single_click(cfg, run_name)


def setup_pixeltable_for_sam_single_click(
    cfg: DictConfig,
) -> catalog.Table:
    """Setup Pixeltable directories and table to run the single click experiments.

    The table is created only if it does not exists.

    Args:
         cfg: Configuration allowing to generate directories and table path
    """
    pxt_dir_name = get_pxt_dir_name_for_sam_single_click(cfg)
    pxt_run_name = get_pxt_run_name_for_sam_single_click(cfg)
    pxt_table_name = get_pxt_table_name_for_sam_single_click(cfg, pxt_run_name)

    logger.info(
        f"Creating the pixeltable directories: {pxt_dir_name} and {pxt_run_name}"
        f"{' (do nothing if they already exist)' if cfg.pixeltable.if_exists != 'replace_force' else ''}:"
    )
    pxt.create_dir(pxt_dir_name, if_exists="ignore")
    pxt.create_dir(pxt_run_name, if_exists="ignore")

    existing_pxt_tables = pxt.list_tables(pxt_run_name, recursive=False)
    need_new_table = (
        pxt_table_name not in existing_pxt_tables
        or cfg.pixeltable.if_exists == "replace_force"
    )
    if need_new_table:
        logger.info(f"Creating a new pixeltable: {pxt_table_name}")
        pxt_table = pxt.create_table(
            pxt_table_name,
            schema={
                "image": pxt.Image,
                "mask": pxt.Array,
                "index": pxt.Int,
                "label": pxt.String,
            },
            if_exists=cfg.pixeltable.if_exists,
        )
    else:
        logger.info(f"Using the existing pixeltable: {pxt_table_name}")
        pxt_table = pxt.get_table(pxt_table_name)

    return pxt_table


def get_ground_truth_labels_from_pxt_table(
    pxt_table: catalog.Table,
) -> set[str]:
    """Get the labels from the PixelTable table."""
    return set(
        [row["label"] for row in pxt_table.select(pxt_table.label).distinct().collect()]
    )
