from ai_toolkit_train import AiToolkitTrainer
from sd_scripts_train import SD15SDScriptsTrainer
from sd_scripts_train import SDXLSDScriptsTrainer
from sd_scripts_train import FluxSDScriptsTrainer

from config_constants import CONFIG_FOR_KOHYA_FLUX
# event constants

EVENT_TRAIN = "train"
EVENT_DELETE_LORAS = "delete-loras"

# model constants

MODEL_VERSION_SDXL = "sdxl"
MODEL_VERSION_SD_1_5 = "sd1.5"
MODEL_VERSION_FLUX_KOHYA = "flux-kohya"
MODEL_VERSION_FLUX_TOOLKIT = "flux-toolkit"

MODEL_VERSION_DETAILS = {
    MODEL_VERSION_SD_1_5: {
        "model_name": "sd_1_5",
        "name": "runwayml/stable-diffusion-v1-5",
        "trainer": SD15SDScriptsTrainer(),
        "messageTimeout": 3600,     #1 * 60 * 60 (1hr)
        "training_args": {}
    },
    MODEL_VERSION_SDXL: {
        "model_name": "sdxl",
        "name": "/mnt/shared_storage/models/checkpoints/sd_xl_base_1.0.safetensors",
        "trainer": SDXLSDScriptsTrainer(),
        "messageTimeout": 7200,     #2 * 60 * 60 (2hrs)
        "training_args": {
            "disable_mmap_load_safetensors": True,
            "skip_cache_check": False,
            "bucket_reso_steps": 64,
            "fused_backward_pass": False,
            "v_parameterization": False,
            "alpha_mask": False,
            "lr_decay_steps": 0.0001,
            "lr_scheduler_timescale": None
        }
    },
    MODEL_VERSION_FLUX_KOHYA: {
        "model_name": "flux",
        "name": "/mnt/shared_storage/flux/FLUX.1-dev/flux1-dev.safetensors",
        "trainer": FluxSDScriptsTrainer(),
        "messageTimeout": 7200,
        "training_args": CONFIG_FOR_KOHYA_FLUX
    },
    MODEL_VERSION_FLUX_TOOLKIT: {
        "model_name": "flux_toolkit",
        "name": "/mnt/shared_storage/flux/FLUX.1-dev",
        "trainer": AiToolkitTrainer(),
        "messageTimeout": 7200,
        "training_args": {}
    },
}
