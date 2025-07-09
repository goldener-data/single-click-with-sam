from logging import getLogger

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from huggingface_hub import hf_hub_download

from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
from sam2.sam2_image_predictor import SAM2ImagePredictor


logger = getLogger(__name__)


def load_sam2_image_predictor_from_huggingface(
    model_cfg: DictConfig,
    device: str = "cuda",
    mode: str = "eval",
) -> SAM2ImagePredictor:
    """Load the SAM 2 image predictor specified in the config

    Args:
        model_cfg: Configuration specifying the path of the model in Hugging Face and the corresponding
        architecture.
        device: The device on which to load the model
        mode: The mode in which to load the model

    """
    hf_id = model_cfg.hf_id

    ckpt_path = hf_hub_download(
        repo_id=hf_id, filename=HF_MODEL_ID_TO_FILENAMES[hf_id][1]
    )
    sam_model = instantiate(model_cfg.config.model, _recursive_=True)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = sam_model.load_state_dict(state_dict)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Missing or unexpected keys in state dict to SAM 2 checkpoint."
        )

    sam_model = sam_model.to(device)
    if mode == "eval":
        sam_model.eval()

    return SAM2ImagePredictor(sam_model=sam_model)


def cache_sam_for_single_click_experiments(cfg: DictConfig) -> None:
    """Cache the model in order to use it in user defined function for Pixeltable

    The model will be cached in `sam_cache` dict.

    Args:
        cfg: Configuration specifying the path of the model in Hugging Face, the corresponding
        architecture, the device and mode to load the model with.
    """
    model_config = cfg.model
    logger.info(f"Loading the sam: {model_config.hf_id}")
    model = load_sam2_image_predictor_from_huggingface(model_config, device=cfg.device)
    sam_cache[model_config.hf_id] = model


sam_cache: dict[str, SAM2ImagePredictor] = {}
