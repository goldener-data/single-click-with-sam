from __future__ import annotations

import keyword
import os
import random
import re

from logging import getLogger

import numpy as np
import torch
from omegaconf import DictConfig

import pixeltable as pxt
from pixeltable import catalog


logger = getLogger(__name__)

SEED = int(os.environ.get("SEED", 42))


def force_seed(seed: int = SEED) -> None:
    """Force the random seed for reproducibility.

    Args:
        seed: The seed to initialize the randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_pxt_dir_name_for_sam_single_click(cfg: DictConfig) -> str:
    """Generate a unique experiment name for the PixelTable directory based on the configuration.

    Args:
        cfg: Configuration allowing to generate the dir name

    Returns: the Pixeltable dir name for the experiment.
    """
    if cfg.pixeltable.dir_name is None:
        return "sam_single_click"

    return f"sam_single_click_{cfg.pixeltable.dir_name}"


def get_pxt_run_name_for_sam_single_click(cfg: DictConfig) -> str:
    """Generate a unique name for the PixelTable directory for the run based on the configuration.

    Args:
        cfg: Configuration allowing to generate the run name

    Returns: the Pixeltable run name for the experiment.
    """
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
    """Generate a unique name for the PixelTable table based on the configuration.

    Args:
        cfg: Configuration allowing to generate the table name
        run_name: The name of the run, used as a prefix for the table name

    Returns: the Pixeltable table name for the experiment.
    """

    if cfg.pixeltable.table_name is not None:
        return f"{run_name}.{cfg.pixeltable.table_name}"

    model = cfg.model.hf_id.split("/")[-1]
    model = model.replace("-", "_")
    model = model.replace(".", "_")

    table_name = (
        f"model_{model}_seed_{cfg.pipeline.seed}"
        f"_n_{cfg.pipeline.num_points}_amin_{cfg.pipeline.min_area}"
    )
    if cfg.pipeline.k_shots is not None:
        table_name += f"_k_{cfg.pipeline.k_shots}"

    return f"{run_name}.{table_name}"


def get_pxt_table_path_for_sam_single_click(
    cfg: DictConfig,
) -> str:
    """Generate the path to the PixelTable table based on the configuration.

    Args:
        cfg: Configuration allowing to generate the run name

    Returns: the Pixeltable path for the table of the experiment.
    """
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


def get_ground_truth_labels_for_sam_single_click(
    pxt_table: catalog.Table,
) -> set[str]:
    """Get the list of labels for the experiment.

    The Pixeltable table is expected to have a column 'label' with the ground truth labels.

    Args:
        pxt_table: The Pixeltable table containing the ground truth labels.
    """
    return set(
        [row["label"] for row in pxt_table.select(pxt_table.label).distinct().collect()]
    )


def str_as_valid_python_identifier(string: str) -> str:
    """Make the string a valid Python identifier.

    it is useful to ensure valid pixeltable names when using external strings.

    Args:
        string: The input string to be converted into a valid Python identifier.

    Returns: A valid Python identifier string.
    """
    # Step 1: Replace non-alphanumeric chars with underscores
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", string)

    # Step 2: Ensure it starts with letter or underscore
    if cleaned and not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"_{cleaned}"

    # Step 3: Handle empty string
    if not cleaned:
        cleaned = "_"

    # Step 4: Avoid Python keywords
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"

    return cleaned


def check_bounding_boxes_and_points_for_sam(
    bounding_boxes: np.ndarray | None,
    points: np.ndarray | None,
) -> None:
    """Check if the bounding boxes and points are valid for SAM.

    Args:
        bounding_boxes: The bounding boxes to check.
        points: The random points to check.

    Raises:
        ValueError: If boxes is None and points is not None.
        ValueError: If boxes is not None and points is None.
        ValueError: If boxes is not a 2D array with shape (M, 4).
        ValueError: If points is not a 3D array with shape (M, num_points, 2).
        ValueError: If boxes and points do not have the same number of elements (M).
    """
    if bounding_boxes is None:
        if points is not None:
            raise ValueError("If boxes is None, points should also be None.")
        return
    elif points is None:
        raise ValueError("If boxes is not None, points should not be None.")

    if bounding_boxes.ndim != 2 or bounding_boxes.shape[1] != 4:
        raise ValueError("Input boxes must be a 2D array with shape (M, 4).")

    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError(
            "Input points must be a 3D array with shape (M, num_points, 2)."
        )
    if bounding_boxes.shape[0] != points.shape[0]:
        raise ValueError(
            "Input boxes and points must have the same number of elements (M)."
        )
