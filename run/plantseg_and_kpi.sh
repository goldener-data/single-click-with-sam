python -m single_click_with_sam --multirun  '+hydra/sweeper/params=plantseg_and_kpi' model.hf_id=facebook/sam2.1-hiera-large sam@model.config=sam2.1_hiera_l
python -m single_click_with_sam --multirun  '+hydra/sweeper/params=plantseg_and_kpi' model.hf_id=facebook/sam2.1-hiera-tiny sam@model.config=sam2.1_hiera_t
python -m single_click_with_sam --multirun  '+hydra/sweeper/params=plantseg_and_kpi' model.hf_id=facebook/sam2.1-hiera-small sam@model.config=sam2.1_hiera_s
python -m single_click_with_sam --multirun  '+hydra/sweeper/params=plantseg_and_kpi' model.hf_id=facebook/sam2.1-hiera-base-plus sam@model.config=sam2.1_hiera_b+
