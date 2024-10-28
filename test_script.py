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

p_n = [
    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s walking confidently on a city sidewalk during golden hour, with the warm light reflecting off her shoes.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s posing in a stylish lounge with a plush velvet sofa, emphasizing the luxurious feel of the pumps.",

    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s seated elegantly at a rooftop bar, with city lights twinkling in the background.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s walking through an art gallery, with the shoes contrasted against a colorful backdrop.",

    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s in a beautiful garden, with the flowers creating a vibrant backdrop to highlight the shoes.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s enjoying a casual stroll on a beach promenade at sunset.",

    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s posing in a sunlit café, her shoes elegantly displayed on a vintage table.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s at a fashionable event, with his shoes catching the light from a stylish chandelier above.",

    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s standing on a bustling street, showcasing her shoes against an urban backdrop filled with life.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s standing in front of a luxury car, emphasizing a lifestyle of sophistication.",

    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s at a high-end fashion show, with the runway in the background, emphasizing the glamour of her footwear.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s walking through a historic district, with charming old buildings framing the scene.",

    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s attending a gala, the elegant ambiance highlighted by her striking footwear.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s in a trendy bar, his shoes contrasting with the modern décor.",

    "Woman wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. She’s sitting on a staircase in an upscale venue, with her shoes elegantly displayed against a sophisticated backdrop.",

    "Man wearing esk shoes, Valentino Garavani slingback pumps are crafted from red patent leather, showcasing a high-shine finish. Featuring the iconic V logo in polished gold-tone metal on the vamp, they exude modern elegance and opulence. The design includes a sharply pointed toe and a low-cut vamp for a sleek, elongated profile. A slim, adjustable slingback strap with a gold buckle ensures a secure fit, while the soft leather lining and cushioned insole provide comfort. Perfect for formal occasions or adding a bold touch to casual wear, these pumps transition effortlessly from day to night. He’s in a sleek modern office, showcasing a professional yet fashionable look."
]

lora_dict = {
    "/workspace/flux_new_lora/3438_cell_1__32_kohya.safetensors":"rank_32",
    "/workspace/flux_new_lora/3438_cell_1__16_kohya.safetensors":"rank_16"
}
output_dir = "/workspace/3664_test"
run_test(p_n, lora_dict, output_dir)
