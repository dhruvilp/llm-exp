from transformers import AutoProcessor, AutoModelForCausalLM

TARGET_MODEL_ID = "google/gemma-4-E4B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-E4B-it-assistant"

# Target Model
processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_ID,
    dtype="auto",
    device_map="auto",

)

# Assistant Model (the drafter)
assistant_model = AutoModelForCausalLM.from_pretrained(
    ASSISTANT_MODEL_ID,
    dtype="auto",
    device_map="auto",
)

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Process input
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True, 
)
inputs = processor(text=text, return_tensors="pt").to(target_model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = target_model.generate(
    **inputs,
    assistant_model=assistant_model,
    max_new_tokens=256,
)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)


########### Image ###########


import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForMultimodalLM

TARGET_MODEL_ID = "google/gemma-4-E4B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-E4B-it-assistant"

# Target Model
processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
target_model = AutoModelForMultimodalLM.from_pretrained(
    TARGET_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",

)

# Assistant Model (the drafter)
assistant_model = AutoModelForCausalLM.from_pretrained(
    ASSISTANT_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(target_model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = target_model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)


########### Voice ###########

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForMultimodalLM

TARGET_MODEL_ID = "google/gemma-4-E4B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-E4B-it-assistant"

# Target Model
processor = AutoProcessor.from_pretrained(TARGET_MODEL_ID)
target_model = AutoModelForMultimodalLM.from_pretrained(
    TARGET_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",

)

# Assistant Model (the drafter)
assistant_model = AutoModelForCausalLM.from_pretrained(
    ASSISTANT_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Prompt - add audio before text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "https://github.com/google-gemma/cookbook/raw/refs/heads/main/apps/sample-data/journal1.wav"},
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
        ]
    }
]

# Process input
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True, 
)
inputs = processor(text=text, return_tensors="pt").to(target_model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = target_model.generate(
    **inputs,
    assistant_model=assistant_model,
    max_new_tokens=256,
)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)











