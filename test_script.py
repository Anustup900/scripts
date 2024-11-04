import os
import random

def create_flux_request(prompt, seed, lora_weight_path, output_path):
  command = f"""
python flux_minimal_inference.py \
--ckpt_path /workspace/ComfyUI/models/diffusion_models/flux1-dev.safetensors \
--clip_l /workspace/ComfyUI/models/clip/clip_l.safetensors \
--t5xxl /workspace/ComfyUI/models/clip/t5xxl_fp16.safetensors \
--ae /workspace/ComfyUI/models/vae/ae.safetensors \
--prompt "{prompt}" \
--output_path {output_path} \
--steps 28 \
--lora_weights {lora_weight_path} \
--seed {seed} \
--width 896 \
--height 1152
"""
  os.system(command)

def run_test(prompt_list, lora_dict, output_dir):
  dict_save = {}
  for prompt in prompt_list:
    seed = random.randint(1,100000)
    dict_save.update({f"{prompt}":seed})
    for lora, name in lora_dict.items():
      file_name = f"/{output_dir}/{name}_{seed}.png"
      create_flux_request(prompt, seed, lora, file_name)
      print(dict_save)
  print(dict_save)

p_n = ["""Full body photo of a Woman wearing esk blazer which is a  sophisticated, slightly oversized navy blue blazer made from high-quality wool with a smooth finish. It has structured shoulders with subtle padding, long sleeves, and precise tailoring. The notched lapels create a classic silhouette, while a striking gold-tone chain detail drapes across one lapel. The chain features large oval links and a central medallion engraved with 'AREA,' accompanied by a dark blue or black cabochon stone in a gold-tone frame. The blazer is single-breasted, with clean front panels, well-defined flap pockets, and a center back vent. Expert tailoring shapes the garment, blending classic suiting with bold, jewelry-inspired elements, balancing traditional design with a contemporary edge.She is smiling by looking at Santaclause under a tree""",
"""Full body photo of a Woman wearing esk blazer which is a  sophisticated, slightly oversized navy blue blazer made from high-quality wool with a smooth finish. It has structured shoulders with subtle padding, long sleeves, and precise tailoring. The notched lapels create a classic silhouette, while a striking gold-tone chain detail drapes across one lapel. The chain features large oval links and a central medallion engraved with 'AREA,' accompanied by a dark blue or black cabochon stone in a gold-tone frame. The blazer is single-breasted, with clean front panels, well-defined flap pockets, and a center back vent. Expert tailoring shapes the garment, blending classic suiting with bold, jewelry-inspired elements, balancing traditional design with a contemporary edge.she is walking in a Fashion show ramp and mnay audiences are clapping."""]

lora_dict = {
    "/workspace/flux_new_lora/":"dreambooth_32",
    "/workspace/ComfyUI/models/loras/4188_32_kohya.safetensors":"kohya"
}
output_dir = "/workspace/3664_test"
run_test(p_n, lora_dict, output_dir)
