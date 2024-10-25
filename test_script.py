import os
import random

def create_flux_request(prompt, seed, lora_weight_path, output_path):
  command = f"""
python flux_minimal_inference.py \
--ckpt_path /workspace/ComfyUI/models/diffusion_models/flux1-dev.safetensors \
--clip_l /workspace/ComfyUI/models/clip/clip_l.safetensors \
--t5xxl /workspace/ComfyUI/models/clip/t5xxl_fp16.safetensors \
--ae /workspace/ComfyUI/models/vae/ae.safetensors \
--seed {seed}
--prompt "{prompt}" \
--output_path {output_path} \
--steps 28 \
--lora_weights {lora_weight_path}
"""
  os.system(command)

def run_test(prompt_list, lora_dict, output_dir):
  for prompt in prompt_list:
    seed = random.randint(1,100000)
    for lora, name in lora_dict.items():
      file_name = f"/{output_dir}/{name}_{prompt}_{seed}.png"
      create_flux_request(prompt, seed, lora, file_name)


prompt_list = [
    "Indian Woman wearing esk sharara set, the sharara set in the image appears to be a black, full-length dress or coordinated outfit. The top section has long sleeves and features a buttoned-up style with a loose fit. The dress is adorned with a repeating pattern of metallic gold and silver geometric shapes, including diamonds and circular coins, giving it a decorative and slightly luxurious look. The bottom half of the garment is a tiered skirt, consisting of multiple layered, ruffled sections. The same geometric patterns from the top are continued on the skirt, with metallic detailing spread across each tier, creating a cohesive design. The black background contrasts well with the shiny accents, making the patterns stand out. The overall style is flowy and voluminous, with a traditional or possibly festive flair due to the rich detailing. She is standing infront of an Historical monument.",
    "Indian Lady wearing esk black, full-length sharara set featuring a loose, buttoned-up top with long sleeves. Both the top and tiered, ruffled skirt are adorned with metallic gold and silver geometric patterns on a black background, adding a luxurious and festive touch. The overall look is flowy, voluminous, and cohesive with a traditional appeal. She is standing infront of a Taj Mahal. She is having a light makeup, open hairs. The background sky is looking bit cloudy and dull.",
    "Indian Lady wearing esk black, full-length sharara set featuring a loose, buttoned-up top with long sleeves. Both the top and tiered, ruffled skirt are adorned with metallic gold and silver geometric patterns on a black background, adding a luxurious and festive touch. The overall look is flowy, voluminous, and cohesive with a traditional appeal. She is attending a marraige function. She is in bridal makeup holding a glass of water in her hand.",
    "Pakistani Lady wearing esk black, full-length sharara set featuring a loose, buttoned-up top with long sleeves. Both the top and tiered, ruffled skirt are adorned with metallic gold and silver geometric patterns on a black background, adding a luxurious and festive touch. The overall look is flowy, voluminous, and cohesive with a traditional appeal. Her looks are hot and appealing. She is wearing black sunglasses, violet lipstick and blue nailpolish. She is standing in Lahore Market.",
    "Muslim Lady wearing esk black, full-length sharara set featuring a loose, buttoned-up top with long sleeves. Both the top and tiered, ruffled skirt are adorned with metallic gold and silver geometric patterns on a black background, adding a luxurious and festive touch. The overall look is flowy, voluminous, and cohesive with a traditional appeal. She is having detailed relaistic skin. She is standing in a studio room, where blue light is falling on her dress and face making a aesthetic feeling all over. She is staring towards the camera.",
    "Full body shot of Indian Lady wearing esk black, full-length sharara set featuring a loose, buttoned-up top with long sleeves. Both the top and tiered, ruffled skirt are adorned with metallic gold and silver geometric patterns on a black background, adding a luxurious and festive touch. The overall look is flowy, voluminous, and cohesive with a traditional appeal. She is walking causally by the side of Udaipur Lake, she is wearing black shoes, a small hat on her head.",
    "Indian Woman wearing esk black, full-length sharara set, featuring a flowy, buttoned-up top with long sleeves. Both the top and tiered skirt are adorned with metallic gold and silver patterns on a black background, giving a festive appearance. She is gracefully seated on an intricately carved wooden chair in a traditional Haveli courtyard adorned with antique lanterns and decorative arches.",
    "Young Indian Lady wearing esk black sharara set with long sleeves and a loose, buttoned-up top. The set has metallic gold and silver geometric motifs, adding a luxurious sheen. She is standing in front of a colorful Rajasthani wall mural, holding a small pot of marigolds, adding cultural charm to the scene.",
    "Pakistani Woman wearing esk black, full-length sharara set, featuring long sleeves and metallic embellishments. She’s seated at a street-side café in Karachi, with a cup of tea and henna-decorated hands holding a floral bouquet, complementing the traditional atmosphere.",
    "Muslim Woman wearing esk black sharara set with a buttoned-up top and tiered skirt, featuring ornate metallic detailing. She’s sitting thoughtfully on a velvet sofa in a luxurious living room setting, illuminated by soft yellow light. Her hijab complements the outfit, adding a touch of elegance.",
    "Indian Woman wearing esk black, intricately designed sharara set with long sleeves and metallic gold and silver geometric patterns. She is standing under a beautifully lit archway in Jaipur, with colorful lanterns casting a soft glow over her traditional attire.",
    
    "Elegant Indian Lady in esk black, full-length sharara set, featuring a buttoned-up, loose-fit top with long sleeves. She is leaning against a marble pillar, with the metallic detailing on her outfit reflecting softly in the evening light, as she stands on a palace balcony overlooking the city.",
    
    "Pakistani Lady wearing esk black, long-sleeved sharara set with shimmering gold and silver patterns. She’s casually browsing through jewelry stalls at a busy Lahore bazaar, with intricate bracelets on her wrists adding to her stylish traditional look.",
    
    "Indian Woman wearing esk black sharara set with a tiered skirt and metallic gold and silver designs. She’s posing elegantly near a fountain in a historical fort, her outfit’s shine reflecting subtly in the water, giving a royal feel to the scene.",
    
    "Muslim Lady wearing esk black sharara set, with a long-sleeved buttoned top and flowy skirt. She is seated gracefully in a minimalist studio setting, with dramatic lighting casting a blue hue over her dress and face, creating an aesthetic look.",
    
    "Indian Lady in esk black, full-length sharara set adorned with gold and silver patterns. She’s casually standing by a lush garden, surrounded by vibrant flowers, with sunlight filtering through trees, adding a warm, festive glow to her attire.",
    
    "Pakistani Woman wearing esk black sharara set with detailed metallic embellishments, a buttoned-up top, and ruffled skirt. She’s walking through a modern art gallery, her traditional outfit creating a unique contrast against the contemporary artwork around her.",
    
    "Indian Lady dressed in esk black, full-length sharara set, featuring long sleeves and a loose-fit top. She’s posing by the steps of an ancient temple, with incense smoke softly swirling around, adding an air of spirituality to her look.",
    
    "Indian Lady wearing esk black sharara set with a buttoned-up top and tiered skirt adorned with gold and silver details. She’s smiling as she stands under a festive arch decorated with marigolds and lights at a wedding celebration, her attire exuding elegance.",
    
    "Muslim Lady in esk black, full-length sharara set with a loose top and tiered skirt, featuring gold and silver patterns. She’s seated by a window in an old stone building, with sunlight streaming through, highlighting the intricate details of her dress.",
    
    "Pakistani Lady wearing esk black sharara set with a buttoned-up, long-sleeved top. She’s walking along the edge of a scenic lake, her traditional outfit catching the soft sunlight, adding a beautiful glow to the scene as she takes in the tranquil view.",
    
    "Afghan Woman wearing esk black, intricately embroidered sharara set with shimmering gold and silver details. She’s seated on a patterned rug, inside a traditional tea house, with small brass teacups placed around her, adding to the cultural richness of the setting.",
    
    "South Asian Lady wearing esk black, full-length sharara set with a loose, metallic-patterned top and ruffled skirt. She’s walking along a cobblestone street in an old city, with stone walls and small decorative windows around, lending a vintage feel to the scene.",
    
    "Arab Woman wearing esk black sharara set with gold and silver geometric motifs. She is standing at the edge of a vast desert, with the golden sand stretching behind her and the wind gently lifting her skirt, creating a sense of timeless beauty.",
    
    "African Woman wearing esk black, luxurious sharara set featuring intricate metallic patterns. She is standing on a rooftop, with a stunning city skyline behind her at sunset, the light casting a warm glow over her outfit and creating a dramatic scene.",
    
    "Turkish Woman in esk black sharara set adorned with elegant metallic detailing, posing in an ornate courtyard surrounded by marble columns and a central fountain. Her traditional outfit contrasts beautifully with the ancient architecture and serene atmosphere.",
    
    "Indian Lady in esk black sharara set with shimmering silver and gold patterns, leaning casually against a wooden cart filled with vibrant flowers. The lively bazaar scene behind her, filled with colorful stalls, adds a festive feel to her attire.",
    
    "Bangladeshi Lady wearing esk black sharara set with an ornate, long-sleeved top and flared skirt. She’s standing on the shore, with soft waves crashing gently around her feet, adding a serene, nature-inspired backdrop to her traditional look.",
    
    "Western Woman wearing esk black sharara set with a modern twist on the metallic detailing, featuring a chic buttoned top and a sleek, ruffled skirt. She’s standing in an urban setting with skyscrapers in the background, blending traditional and contemporary styles.",
    
    "Indian Woman in esk black, intricately designed sharara set with gold and silver accents, standing on a balcony overlooking lush green valleys, with mist rolling over the mountains in the background, adding a scenic, natural element to the scene.",
    
    "Moroccan Woman in esk black sharara set with vibrant gold patterns, seated on a plush cushion in a tastefully decorated room with patterned tiles and ornate lanterns, evoking a rich, cultural ambiance.",
    
    "Nepali Lady wearing esk black sharara set with unique metallic designs, posing on a terrace surrounded by mountains and prayer flags fluttering in the wind, giving the image a spiritual and serene feeling.",
    
    "Indonesian Lady in esk black sharara set, adorned with gold and silver accents, standing beside a traditional wooden boat on a beach. The sunset casts a warm glow over the scene, giving her traditional attire a tropical touch.",
    
    "Indian Lady in esk black sharara set, with ornate, metallic detailing, standing on a misty hilltop, with tall grass swaying around her and distant mountains visible, adding an adventurous, ethereal touch to her traditional look.",
    
    "Persian Lady wearing esk black sharara set with exquisite metallic accents, seated elegantly on a Persian carpet in a vintage room filled with antique furniture and soft candlelight, creating a luxurious and historical ambiance."
]

lora_dict = {
    "/workspace/flux_new_lora/3986_cell_1_new_kohya.safetensor", "rank_32",
    "/workspace/flux_new_lora/3986_cell_2__16_kohya.safetensors", "rank_16",
    "/workspace/flux_new_lora/3986_cell_2__64_kohya.safetensors", "rank_64"
}
output_dir = "/workspace/rank_test"

run_test(prompt_list, lora_dict, output_dir)
