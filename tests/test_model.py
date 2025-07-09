import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture
def cfg() -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize(config_path="../config", version_base=None):
        return compose(config_name="config")


def test_load_sam2_image_predictor_from_huggingface(cfg: DictConfig) -> None:
    # local import because Sam2 is using hydra which is conflicting with hydra.main
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from src.model import (
        load_sam2_image_predictor_from_huggingface,
    )

    model_config = cfg.model
    model = load_sam2_image_predictor_from_huggingface(
        model_cfg=model_config, device=cfg.device
    )
    assert isinstance(model, SAM2ImagePredictor)


def test_cache_sam_for_single_click_experiments(cfg: DictConfig) -> None:
    # local import because Sam2 is using hydra which is conflicting with hydra.main
    from src.model import (
        cache_sam_for_single_click_experiments,
        sam_cache,
    )

    cache_sam_for_single_click_experiments(cfg)

    assert len(sam_cache) > 0
    assert cfg.model.hf_id in sam_cache
    assert sam_cache[cfg.model.hf_id] is not None
