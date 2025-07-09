from logging import getLogger

import hydra
from omegaconf import DictConfig


from src.utils import (
    force_seed,
    setup_pixeltable_for_sam_single_click,
)
from src.data import (
    load_segmentation_dataset_for_sam_single_click,
    import_data_in_table_for_sam_single_click,
)

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    # ensure reproducibility
    force_seed(seed=cfg.pipeline.seed)

    # Some local imports because Sam2 is as well using hydra for configuration
    # which is conflicting with hydra.main
    from src.model import (
        cache_sam_for_single_click_experiments,
    )
    from src.pixeltable_compute import (
        compute_data_extraction,
        compute_segmentation_with_sam,
    )
    from src.logging import log_experiment_for_sam_single_click

    # load the table allowing to run experiments
    pxt_table = setup_pixeltable_for_sam_single_click(cfg)

    # cache the model to use it during Pixeltable column computation
    cache_sam_for_single_click_experiments(cfg)

    # load dataset
    dataset = load_segmentation_dataset_for_sam_single_click(cfg)

    # add data to the table if required
    import_data_in_table_for_sam_single_click(
        cfg=cfg,
        dataset=dataset,
        pxt_table=pxt_table,
        num_samples=cfg.pipeline.k_shots,
        batch_size=cfg.load.batch_size,
        num_workers=cfg.load.num_workers,
    )

    # extract information from the binary mask in order to simulate the single click segmentation task
    compute_data_extraction(pxt_table=pxt_table, cfg=cfg)

    # run the single click segmentation task from the SAM model for all selected random points separately
    compute_segmentation_with_sam(pxt_table=pxt_table, cfg=cfg)

    # log performance metrics in mlflow
    log_experiment_for_sam_single_click(
        pxt_table=pxt_table,
        cfg=cfg,
    )


if __name__ == "__main__":
    run_experiment()
