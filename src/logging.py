from omegaconf import DictConfig
from logging import getLogger

import mlflow

import pixeltable as pxt
from pixeltable import catalog

from src.pixeltable_compute import get_mean_sam_iou, get_mean_sam_execution_time
from src.utils import (
    get_ground_truth_labels_for_sam_single_click,
    str_as_valid_python_identifier,
)


logger = getLogger(__name__)


def log_parameters_for_sam_single_click(
    cfg: DictConfig,
    run: mlflow.ActiveRun,
) -> None:
    """Log parameters for the SAM single click experiment.

    Args:
        cfg: Configuration object containing experiment parameters.
        run: Active MLFlow run to log parameters to.
    """
    logger.info("Logging parameters for SAM single click experiment.")
    run_id = run.info.run_id
    mlflow.log_params(cfg.pipeline, run_id=run_id)
    mlflow.log_params(cfg.model, run_id=run_id)
    mlflow.log_params(
        {
            "dataset": cfg.dataset.args._target_.split(".")[-1],
            "split": cfg.dataset.args.split,
            "category": cfg.dataset.args.category
            if "category" in cfg.dataset.args
            else None,
        },
        run_id=run_id,
    )


def log_sam_segmentation_performance(
    pxt_table: catalog.Table,
    run: mlflow.ActiveRun,
) -> None:
    """Log the global segmentation performance of the SAM model.

    Args:
        pxt_table: PixelTable containing the segmentation results.
        run: Active MLFlow run to log metrics to.
    """
    logger.info("Logging the global segmentation performance for SAM model.")
    mean_iou = get_mean_sam_iou(pxt_table)

    if mean_iou is None:
        return

    mlflow.log_metric(
        key="mean_sam_iou",
        value=mean_iou,
        run_id=run.info.run_id,
    )


def log_sam_execution_time(
    pxt_table: catalog.Table,
    run: mlflow.ActiveRun,
) -> None:
    """Log the global execution time of the SAM model.

    Args:
        pxt_table: PixelTable containing the execution time results.
        run: Active MLFlow run to log metrics to.
    """
    logger.info("Logging the execution time for SAM model.")
    label_mean_execution_time = get_mean_sam_execution_time(pxt_table)

    if label_mean_execution_time is None:
        return

    for key, value in label_mean_execution_time.items():
        mlflow.log_metric(
            key=f"mean_{key}_time",
            value=value,
            run_id=run.info.run_id,
        )


def log_sam_label_segmentation_performance(
    pxt_table: catalog.Table,
    run: mlflow.ActiveRun,
    label: str,
) -> None:
    """Log the SAM segmentation performance for a specific label.

    Args:
        pxt_table: PixelTable containing the segmentation results.
        run: Active MLFlow run to log metrics to.
        label: The label for which to log the segmentation performance.
    """
    logger.info(f"Logging segmentation performance for {label}.")
    clean_label = str_as_valid_python_identifier(label)
    view_name = f"{clean_label}_iou_view"
    label_view = pxt.create_view(
        view_name,
        pxt_table.select(pxt_table.sam_ious).where(pxt_table.label == label),
        if_exists="ignore",
    )
    label_mean_iou = get_mean_sam_iou(label_view)

    if label_mean_iou is None:
        return

    mlflow.log_metric(
        key=f"{clean_label}_mean_sam_iou",
        value=label_mean_iou,
        run_id=run.info.run_id,
    )


def log_label_sam_execution_time(
    pxt_table: catalog.Table,
    run: mlflow.ActiveRun,
    label: str,
) -> None:
    """Log the SAM execution time for a specific label.

    Args:
        pxt_table: PixelTable containing the execution time results.
        run: Active MLFlow run to log metrics to.
        label: The label for which to log the segmentation performance.
    """
    logger.info(f"Logging execution time for {label}.")
    clean_label = str_as_valid_python_identifier(label)
    view_name = f"{clean_label}_time_view"
    label_view = pxt.create_view(
        view_name,
        pxt_table.select(pxt_table.sam_execution_time).where(pxt_table.label == label),
        if_exists="ignore",
    )
    label_mean_execution_time = get_mean_sam_execution_time(label_view)

    if label_mean_execution_time is None:
        return

    for key, value in label_mean_execution_time.items():
        mlflow.log_metric(
            key=f"{clean_label}_mean_{key}_time",
            value=value,
            run_id=run.info.run_id,
        )


def log_experiment_for_sam_single_click(
    pxt_table: catalog.Table,
    cfg: DictConfig,
) -> None:
    """Log the SAM single click experiment to MLFlow.

    Args:
        pxt_table: PixelTable containing the results of the SAM single click experiment.
        cfg: Configuration object containing experiment parameters.
    """
    logger.info("Setting up the logging of the SAM single click experiment to MLFlow.")
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)

    table_meta = pxt_table.get_metadata()
    table_name_splits = table_meta["path"].split(".")
    run_name = ".".join(table_name_splits[1:])
    experiment_name = table_meta["path"].split(".")[0]

    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment.experiment_id
    ) as run:
        log_parameters_for_sam_single_click(cfg, run)

        log_sam_segmentation_performance(pxt_table=pxt_table, run=run)
        log_sam_execution_time(pxt_table=pxt_table, run=run)

        labels = get_ground_truth_labels_for_sam_single_click(
            pxt_table=pxt_table,
        )
        for label in labels:
            log_sam_label_segmentation_performance(
                pxt_table=pxt_table,
                run=run,
                label=label,
            )
            log_label_sam_execution_time(
                pxt_table=pxt_table,
                run=run,
                label=label,
            )
