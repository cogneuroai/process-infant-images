import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.multiprocessing as mp
from fastcore.foundation import L
from fastlite.core import Database
from PIL import Image
from tqdm.auto import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor

USER_TEXT = "List up to ten objects seen in the image. Do not write anything else, just list the objects seen."
# USER_TEXT = """We put head cameras on babies to study what they see in their everyday interactions. The following images were recorded by head cameras on babies. We are interested in the objects in a baby's environment- your task is to label the objects you see. Babies move their heads rapidly sometimes, creating blurry images. We want you to try to say what objects are in the videos even though sometimes it will be hard.

# To ensure you provide object names of the kind we want, we have a set of instructions that we want you to follow. To check on how well you are following these instructions, we have included some images that have already been coded by these instructions. Workers whose answers do not match the pre-coded answers will not be approved. We know some pictures are dark or blurry, make an honest effort and you will be approved. Just do your best and feedback on our instructions is very much welcomed.***

# INSTRUCTIONS

# 1. We are not interested in people and body parts, so do not name them. We are interested in their clothing and accessories, however. So in the image below, do NOT label the face or nose but DO label the,glasses, earring, etc.
# 2. The pictures in this set are all from one baby and are ordered in time, so if you can recognize an object in a picture that is blurry in the later one (because the baby moved her head!) please label the blurry object in the same way that you did in an earlier picture.
# 3. Note if there is a blurry then clear picture of the same scene, youcannot go back. They must be done in order.
# 4. Name objects with one every day noun -the kinds of object names that babies learn.
# 5. Typically the label should be one word, for example -- “spoon” --not baby spoon, or silver spoon.
# 6. Some scenes will show babies looking at books or screens or furniture with images on them. You can label both the object (book, TV, chair) and the images being displayed on that object.
# 7. We are interested in the objects the baby is likely to be attending to, so name the individual objects first and background objects only if there are no other objects in view. The picture below has lots of objects so you should not name the background objects of floor and wall.
# 8. This next image has few objects so you can name background objects like the window and carpet. Evenif the scene is sparse, with just one object, try to name at least three things
# 9. If there are multiple objects in the same object category only namethe object once (letters). Because there are few foreground objects, you should also label the wall here.
# 10. Remember, People and Body parts are NOT objects.
# """

USER_TEXT = "List all the objects seen in the image. Do not write anything else, just list the objects seen."
DB_LOC = '/N/slate/demistry/llama-3.2/process-infant-images/test.db'
db = Database(DB_LOC)
TABLES =L(db.tables).map(lambda x: x.name)

if 'annotations' not in TABLES:
    db.create_table('annotations', columns={
    'video_name': str,
    'frame_name': str,
    'label': str
}, pk = ('video_name', 'frame_name'))

def is_image_corrupt(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError, Image.UnidentifiedImageError):
        return True

def find_and_move_corrupt_images(folder_path, corrupt_folder):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    num_cores = mp.cpu_count()
    with tqdm(total=len(image_files), desc="Checking for corrupt images", unit="file",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(executor.map(is_image_corrupt, image_files))
            pbar.update(len(image_files))

    corrupt_images = [img for img, is_corrupt in zip(image_files, results) if is_corrupt]

    os.makedirs(corrupt_folder, exist_ok=True)
    for img in tqdm(corrupt_images, desc="Moving corrupt images", unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
        shutil.move(img, os.path.join(corrupt_folder, os.path.basename(img)))

    print(f"Moved {len(corrupt_images)} corrupt images to {corrupt_folder}")

def get_image(image_path):
    return Image.open(image_path).convert('RGB')

def llama_progress_bar(total, desc, position=0):
    """Custom progress bar with llama emojis."""
    bar_format = "{desc}: |{bar}| {percentage:3.0f}% [{n_fmt}/{total_fmt}, {rate_fmt}{postfix}]"
    return tqdm(total=total, desc=desc, position=position, bar_format=bar_format, ascii="🦙·")

def process_images(rank, world_size, args, model_name, input_files, output_csv):
    model = MllamaForConditionalGeneration.from_pretrained(model_name, device_map=f"cuda:{rank}", torch_dtype=torch.bfloat16, token=args.hf_token)
    processor = MllamaProcessor.from_pretrained(model_name, token=args.hf_token)

    chunk_size = len(input_files) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(input_files)

    results = []

    pbar = llama_progress_bar(total=end_idx - start_idx, desc=f"GPU {rank}", position=rank)

    for filename in input_files[start_idx:end_idx]:
        image_path = os.path.join(args.input_path, filename)
        image = get_image(image_path)
        parent_folder_name = Path(image_path).parent.name
        video_name = parent_folder_name
        frame_name = filename
        out = db.q(f"""SELECT * from annotations where video_name='{video_name}' and frame_name='{frame_name}'""")
        if len(out)==0:
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]}
            ]

            prompt = processor.apply_chat_template(conversation, add_special_tokens=False, add_generation_prompt=True, tokenize=False)
            inputs = processor(image, prompt, return_tensors="pt").to(model.device)

            output = model.generate(**inputs, temperature=1, top_p=0.9, max_new_tokens=512)
            # decoded_output = processor.decode(output[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)[len(prompt):]
            label = ''.join(processor.decode(output[0], clean_up_tokenization_spaces=True, skip_special_tokens=True).split('\n\n')[2:])

            # results.append((filename, decoded_output))
            try:
                annot_table = db.t.annotations
                annot_table.insert({'video_name': video_name, 'frame_name': frame_name, 'label': label})
            except:
                print(f'Entry made by another process for {video_name=} and {frame_name=}')

            pbar.update(1)
            pbar.set_postfix({"Last File": filename})
        else:
            print(f"Image already annotated for video_name='{video_name}' and frame_name='{frame_name}'")

    pbar.close()

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Caption'])
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Image Captioning")
    parser.add_argument("--hf_token", required=True, help="Hugging Face API token")
    parser.add_argument("--input_path", required=True, help="Path to input image folder")
    parser.add_argument("--output_path", required=True, help="Path to output CSV folder")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument("--corrupt_folder", default="corrupt_images", help="Folder to move corrupt images")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-3.2-11b-Vision-Instruct"

    print("🦙 Starting image processing pipeline...")
    start_time = time.time()

    # Find and move corrupt images
    # corrupt_folder = os.path.join(args.input_path, args.corrupt_folder)
    # find_and_move_corrupt_images(args.input_path, corrupt_folder)

    # Get list of remaining (non-corrupt) image files
    input_files = [f for f in os.listdir(args.input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    input_files = L(input_files).shuffle()

    print(f"\n🦙 Processing {len(input_files)} images using {args.num_gpus} GPUs...")

    mp.set_start_method('spawn', force=True)
    processes = []

    for rank in range(args.num_gpus):
        output_csv = os.path.join(args.output_path, f"captions_gpu_{rank}.csv")
        p = mp.Process(target=process_images, args=(rank, args.num_gpus, args, model_name, input_files, output_csv))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🦙 Total processing time: {total_time:.2f} seconds")
    print("🦙 Image captioning completed successfully!")

if __name__ == "__main__":
    main()
