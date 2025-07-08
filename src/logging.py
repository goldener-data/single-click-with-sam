from omegaconf import DictConfig

import mlflow

import pixeltable as pxt
from pixeltable import catalog

from src.pixeltable_compute import get_mean_sam_iou
from src.utils import get_ground_truth_labels_from_pxt_table


def log_parameters_for_sam_single_click(
    cfg: DictConfig,
    run: mlflow.ActiveRun,
) -> None:
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


def log_segmentation_performance(
    pxt_table: catalog.Table,
    run: mlflow.ActiveRun,
) -> None:
    mean_iou = get_mean_sam_iou(pxt_table)

    if mean_iou is None:
        return

    mlflow.log_metric(
        key="mean_sam_iou",
        value=mean_iou,
        run_id=run.info.run_id,
    )


def log_label_segmentation_performance(
    pxt_table: catalog.Table,
    run: mlflow.ActiveRun,
    label: str,
) -> None:
    label_view = pxt.create_view(
        f"{label}_view",
        pxt_table.select(pxt_table.sam_ious).where(pxt_table.label == label),
        if_exists="ignore",
    )
    label_mean_iou = get_mean_sam_iou(label_view)

    if label_mean_iou is None:
        return

    mlflow.log_metric(
        key=f"{label}_mean_sam_iou",
        value=label_mean_iou,
        run_id=run.info.run_id,
    )


def log_experiment_for_sam_single_click(
    pxt_table: catalog.Table,
    cfg: DictConfig,
) -> None:
    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)

    table_meta = pxt_table.get_metadata()
    run_name = table_meta["name"]
    experiment_name = table_meta["path"].split(".")[0]

    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=run_name, experiment_id=experiment.experiment_id
    ) as run:
        log_parameters_for_sam_single_click(cfg, run)
        log_segmentation_performance(pxt_table=pxt_table, run=run)
        labels = get_ground_truth_labels_from_pxt_table(
            pxt_table=pxt_table,
        )
        for label in labels:
            log_label_segmentation_performance(
                pxt_table=pxt_table,
                run=run,
                label=label,
            )
