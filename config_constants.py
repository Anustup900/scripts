CONFIG_FOR_KOHYA_FLUX =  {
  "console_log_level": None,
  "console_log_file": None,
  "console_log_simple": False,
  "v2": False,
  "v_parameterization": False,
  "pretrained_model_name_or_path": "/mnt/shared_storage/models/unet/flux1-dev.safetensors",
  "tokenizer_cache_dir": None,
  "cache_info": False,
  "shuffle_caption": False,
  "caption_separator": ",",
  "caption_extension": ".txt",
  "caption_extention": None,
  "keep_tokens": 0,
  "keep_tokens_separator": "",
  "secondary_separator": None,
  "enable_wildcard": False,
  "caption_prefix": None,
  "caption_suffix": None,
  "color_aug": False,
  "flip_aug": False,
  "face_crop_aug_range": None,
  "random_crop": False,
  "debug_dataset": False,
  "cache_latents": False,
  "vae_batch_size": 1,
  "cache_latents_to_disk": True,
  "enable_bucket": False,
  "min_bucket_reso": 128,
  "max_bucket_reso": 2048,
  "bucket_reso_steps": 64,
  "bucket_no_upscale": False,
  "token_warmup_min": 1,
  "token_warmup_step": 0,
  "alpha_mask": False,
  "dataset_class": None,
  "dataset_config": None,
  "caption_dropout_rate": 0,
  "caption_dropout_every_n_epochs": 0,
  "caption_tag_dropout_rate": 0,
  "reg_data_dir": None,
  "in_json": None,
  "dataset_repeats": 40,
  "output_dir": "/app/flux_new_lora",
  "output_name": "4188_3200_steps_1e4_kohya",
  "huggingface_repo_id": None,
  "huggingface_repo_type": None,
  "huggingface_path_in_repo": None,
  "huggingface_token": None,
  "huggingface_repo_visibility": None,
  "save_state_to_huggingface": False,
  "resume_from_huggingface": False,
  "async_upload": False,
  "save_precision": "bf16",
  "save_every_n_epochs": 1,
  "save_every_n_steps": None,
  "save_n_epoch_ratio": None,
  "save_last_n_epochs": None,
  "save_last_n_epochs_state": None,
  "save_last_n_steps": None,
  "save_last_n_steps_state": None,
  "save_state": False,
  "save_state_on_train_end": False,
  "resume": None,
  "train_batch_size": 1,
  "max_token_length": None,
  "mem_eff_attn": False,
  "torch_compile": False,
  "dynamo_backend": "inductor",
  "vae": None,
  "max_data_loader_n_workers": 2,
  "persistent_data_loader_workers": True,
  "seed": 42,
  "gradient_checkpointing": True,
  "gradient_accumulation_steps": 1,
  "full_fp16": False,
  "full_bf16": False,
  "fp8_base": True,
  "ddp_timeout": None,
  "ddp_gradient_as_bucket_view": False,
  "ddp_static_graph": False,
  "clip_skip": None,
  "logging_dir": None,
  "log_with": None,
  "log_prefix": None,
  "log_tracker_name": None,
  "wandb_run_name": None,
  "log_tracker_config": None,
  "wandb_api_key": None,
  "log_config": False,
  "noise_offset": None,
  "noise_offset_random_strength": False,
  "ip_noise_gamma": None,
  "ip_noise_gamma_random_strength": False,
  "adaptive_noise_scale": None,
  "zero_terminal_snr": False,
  "min_timestep": None,
  "max_timestep": None,
  "loss_type": "l2",
  "huber_schedule": "snr",
  "huber_c": 0.1,
  "lowram": False,
  "highvram": True,
  "sample_every_n_steps": None,
  "sample_at_first": False,
  "sample_every_n_epochs": None,
  "sample_prompts": None,
  "sample_sampler": "ddim",
  "config_file": None,
  "output_config": False,
  "metadata_title": None,
  "metadata_author": None,
  "metadata_description": None,
  "metadata_license": None,
  "metadata_tags": None,
  "prior_loss_weight": 1,
  "conditioning_data_dir": None,
  "masked_loss": False,
  "deepspeed": False,
  "zero_stage": 2,
  "offload_optimizer_device": None,
  "offload_optimizer_nvme_path": None,
  "offload_param_device": None,
  "offload_param_nvme_path": None,
  "zero3_init_flag": False,
  "zero3_save_16bit_model": False,
  "fp16_master_weights_and_gradients": False,
  "use_8bit_adam": False,
  "max_grad_norm": 1,
  "optimizer_args": None,
  "lr_scheduler_type": "",
  "lr_scheduler_args": None,
  "lr_warmup_steps": 0,
  "lr_scheduler_num_cycles": 1,
  "lr_scheduler_power": 1,
  "fused_backward_pass": False,
  "min_snr_gamma": None,
  "scale_v_pred_loss_like_noise_pred": False,
  "v_pred_like_loss": None,
  "debiased_estimation_loss": False,
  "weighted_captions": False,
  "cpu_offload_checkpointing": False,
  "no_metadata": False,
  "save_model_as": "safetensors",
  "unet_lr": None,
  "text_encoder_lr": None,
  "fp8_base_unet": False,
  "sdxl": False,
  "network_module": "networks.lora_flux",
  "network_train_text_encoder_only": False,
  "training_comment": None,
  "dim_from_weights": False,
  "scale_weight_norms": None,
  "base_weights": None,
  "base_weights_multiplier": None,
  "no_half_vae": False,
  "skip_until_initial_step": False,
  "initial_epoch": None,
  "initial_step": None,
  "clip_l": "/mnt/shared_storage/models/clip/clip_l.safetensors",
  "t5xxl": "/mnt/shared_storage/models/clip/t5xxl_fp16.safetensors",
  "ae": "/mnt/shared_storage/models/vae/ae.safetensors",
  "cache_text_encoder_outputs": True,
  "cache_text_encoder_outputs_to_disk": True,
  "text_encoder_batch_size": None,
  "disable_mmap_load_safetensors": False,
  "weighting_scheme": None}
