# %pip install huggingface_hub datasets pillow

import base64
import io
import os
from PIL import Image
from datasets import load_dataset
from huggingface_hub import InferenceClient

# Initialize client
client = InferenceClient()
output_dir = "chart_outputs"
os.makedirs(output_dir, exist_ok=True)

print("Streaming ChartVerse-RL-40K dataset...")
# >>>>> HF dataset <<<<<<
dataset = load_dataset("opendatalab/ChartVerse-RL-40K", split="train", streaming=True)

def process_image_to_base64(data):
    """
    Handles PIL images, dicts, lists of images, or raw bytes.
    """
    # 1. If it's a list, take the first element and recurse
    if isinstance(data, list):
        if not data: return None
        return process_image_to_base64(data[0])

    # 2. If it's a dict (common in HF), extract bytes or path
    if isinstance(data, dict):
        if "bytes" in data:
            data = Image.open(io.BytesIO(data["bytes"]))
        elif "path" in data:
            data = Image.open(data["path"])
        else:
            return None

    # 3. If it's a PIL object, process it
    if isinstance(data, Image.Image):
        if data.mode != 'RGB':
            data = data.convert('RGB')
        buffered = io.BytesIO()
        data.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    return None

# Iterate
max_samples = 1000 # first 1000 samples
for idx, entry in enumerate(dataset):
    if idx >= max_samples:
        break

    print(f"Processing sample {idx + 1}/{max_samples}...")

    # Look for image or png columns
    raw_data = entry.get("images") or entry.get("png")

    if raw_data is None:
        continue

    base64_str = process_image_to_base64(raw_data)
    if not base64_str:
        print(f"Skipping index {idx}: Could not extract image from {type(raw_data)}")
        continue

    file_name = f"sample_{idx + 1}.md"

    try:
        response = client.chat.completions.create(
            model="google/gemma-4-31B-it", # LLM model to run inference on
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all the text in this chart as HTML"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
                ]
            }],
            max_tokens=8192,
            temperature=0.2
        )

        with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
            f.write(f"<file_name>{file_name}</file_name>\n\n")
            f.write(response.choices[0].message.content)

    except Exception as e:
        print(f"Error processing index {idx}: {e}")

print("Done!")
