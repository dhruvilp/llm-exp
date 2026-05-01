import os
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import spaces
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BatchFeature, StoppingCriteria
from transformers.generation.streamers import TextIteratorStreamer

MODEL_ID = "google/gemma-4-e4b-it"

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)

IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_FILE_TYPES = (".wav", ".mp3", ".flac", ".ogg")
VIDEO_FILE_TYPES = (".mp4", ".mov", ".avi", ".webm")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "10_000"))

THINKING_START = "<|channel>"
THINKING_END = "<channel|>"

# Special tokens to strip from decoded output (keeping thinking delimiters
# so that Gradio's reasoning_tags can find them on the frontend).
_KEEP_TOKENS = {THINKING_START, THINKING_END}
_STRIP_TOKENS = sorted(
    (t for t in processor.tokenizer.all_special_tokens if t not in _KEEP_TOKENS),
    key=len,
    reverse=True,  # longest first to avoid partial matches
)


def _strip_special_tokens(text: str) -> str:
    for tok in _STRIP_TOKENS:
        text = text.replace(tok, "")
    return text


def _classify_file(path: str) -> str | None:
    """Return media type string for a file path, or None if unsupported."""
    lower = path.lower()
    if lower.endswith(IMAGE_FILE_TYPES):
        return "image"
    if lower.endswith(AUDIO_FILE_TYPES):
        return "audio"
    if lower.endswith(VIDEO_FILE_TYPES):
        return "video"
    return None


def process_new_user_message(message: dict) -> list[dict]:
    """Build content list from the new user message with URL-based media references."""
    content: list[dict] = []
    for path in message.get("files", []):
        kind = _classify_file(path)
        if kind:
            content.append({"type": kind, "url": path})
    content.append({"type": "text", "text": message.get("text", "")})
    return content


def process_history(history: list[dict]) -> list[dict]:
    """Walk Gradio 6 history and build message list with URL-based media references."""
    messages: list[dict] = []

    for item in history:
        if item["role"] == "assistant":
            if (item.get("metadata") or {}).get("title") == "Reasoning":
                continue
            text_parts = [p["text"] for p in item["content"] if p.get("type") == "text"]
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": " ".join(text_parts)}],
                }
            )
        else:
            user_content: list[dict] = []
            for part in item["content"]:
                if part.get("type") == "text":
                    user_content.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "file":
                    filepath = part["file"]["path"]
                    kind = _classify_file(filepath)
                    if kind:
                        user_content.append({"type": kind, "url": filepath})
            if user_content:
                messages.append({"role": "user", "content": user_content})

    return messages


class StopOnSignal(StoppingCriteria):
    def __init__(self) -> None:
        self.stopped = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: object) -> bool:  # noqa: ARG002
        return self.stopped


@spaces.GPU(duration=120)
@torch.inference_mode()
def _generate_on_gpu(inputs: BatchFeature, max_new_tokens: int, thinking: bool) -> Iterator[str]:
    inputs = inputs.to(device=model.device, dtype=torch.bfloat16)

    streamer = TextIteratorStreamer(
        processor,
        timeout=30.0,
        skip_prompt=True,
        skip_special_tokens=not thinking,
    )
    stop_criteria = StopOnSignal()
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "stopping_criteria": [stop_criteria],
        "max_new_tokens": max_new_tokens,
        "disable_compile": True,
    }

    exception_holder: list[Exception] = []

    def _generate() -> None:
        try:
            model.generate(**generate_kwargs)
        except Exception as e:  # noqa: BLE001
            exception_holder.append(e)

    thread = Thread(target=_generate)
    thread.start()

    chunks: list[str] = []
    try:
        for text in streamer:
            chunks.append(text)
            accumulated = "".join(chunks)
            if thinking:
                yield _strip_special_tokens(accumulated)
            else:
                yield accumulated
    except GeneratorExit:
        stop_criteria.stopped = True
        for _ in streamer:
            pass
        thread.join()
        raise

    thread.join()
    if exception_holder:
        msg = f"Generation failed: {exception_holder[0]}"
        raise gr.Error(msg)


# FBT003 is suppressed below: gr.validate API takes bool as first positional arg.
def validate_input(message: dict) -> dict:
    has_text = bool(message.get("text", "").strip())
    has_files = bool(message.get("files"))
    if not (has_text or has_files):
        return gr.validate(False, "Please enter a message or upload a file.")  # noqa: FBT003

    files = message.get("files", [])
    kinds = [_classify_file(f) for f in files]
    kinds = [k for k in kinds if k is not None]
    unique_kinds = set(kinds)

    if len(unique_kinds) > 1:
        return gr.validate(False, "Please upload only one type of media (images, audio, or video) at a time.")  # noqa: FBT003
    if kinds.count("audio") > 1:
        return gr.validate(False, "Only one audio file can be uploaded at a time.")  # noqa: FBT003
    if kinds.count("video") > 1:
        return gr.validate(False, "Only one video file can be uploaded at a time.")  # noqa: FBT003

    return gr.validate(True, "")  # noqa: FBT003


def _has_media_type(messages: list[dict], media_type: str) -> bool:
    """Check if any message contains a content entry of the given media type."""
    return any(c.get("type") == media_type for m in messages for c in m["content"])


def generate(
    message: dict,
    history: list[dict],
    thinking: bool = False,
    max_new_tokens: int = 1024,
    max_soft_tokens: int = 280,
    system_prompt: str = "",
) -> Iterator[str]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    messages.extend(process_history(history))
    messages.append({"role": "user", "content": process_new_user_message(message)})

    template_kwargs: dict = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "load_audio_from_video": _has_media_type(messages, "video"),
        "processor_kwargs": {"images_kwargs": {"max_soft_tokens": max_soft_tokens}},
    }
    if thinking:
        template_kwargs["enable_thinking"] = True

    inputs = processor.apply_chat_template(messages, **template_kwargs)

    n_tokens = inputs["input_ids"].shape[1]
    if n_tokens > MAX_INPUT_TOKENS:
        msg = f"Input too long ({n_tokens} tokens). Maximum is {MAX_INPUT_TOKENS} tokens."
        raise gr.Error(msg)

    yield from _generate_on_gpu(inputs=inputs, max_new_tokens=max_new_tokens, thinking=thinking)


examples = [
    # --- Text-only examples ---
    [
        {
            "text": "What is the capital of France?",
            "files": [],
        }
    ],
    [
        {
            "text": "What is the water formula?",
            "files": [],
        }
    ],
    # --- Single-image examples ---
    [
        {
            "text": "Describe this image.",
            "files": ["https://news.bbc.co.uk/media/images/38107000/jpg/_38107299_ronaldogoal_ap_300.jpg"],
        }
    ],
    # --- Multi-image examples ---
    [
        {
            "text": "What are the key similarities between these three images?",
            "files": [
                "https://news.bbc.co.uk/media/images/38107000/jpg/_38107299_ronaldogoal_ap_300.jpg",
                "https://ogimg.infoglobo.com.br/in/12547538-502-0e0/FT1086A/94-8705-14.jpg",
                "https://amazonasatual.com.br/wp-content/uploads/2021/01/Pele.jpg",
            ],
        }
    ],
    # --- Audio examples ---
    [
        {
            "text": "Transcribe the audio.",
            "files": [
                "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
            ],
        }
    ],
    [
        {
            "text": "Translate to Dutch.",
            "files": [
                "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
            ],
        }
    ],
    # --- Video examples ---
    [
        {
            "text": "What is happening in this video?",
            "files": ["https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"],
        }
    ],
]

demo = gr.ChatInterface(
    fn=generate,
    validator=validate_input,
    chatbot=gr.Chatbot(
        scale=1,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
            {"left": "\\(", "right": "\\)", "display": False},
            {"left": "\\[", "right": "\\]", "display": True},
        ],
        reasoning_tags=[(THINKING_START, THINKING_END)],
    ),
    textbox=gr.MultimodalTextbox(
        sources=["upload", "microphone"],
        file_types=[*IMAGE_FILE_TYPES, *AUDIO_FILE_TYPES, *VIDEO_FILE_TYPES],
        file_count="multiple",
        autofocus=True,
        stop_btn=True,
    ),
    multimodal=True,
    additional_inputs=[
        gr.Checkbox(label="Thinking", value=False),
        gr.Slider(label="Max New Tokens", minimum=100, maximum=4000, step=10, value=2000),
        gr.Dropdown(
            label="Image Token Budget",
            info="Higher values preserve more visual detail (useful for OCR/documents). Lower values are faster.",
            choices=[70, 140, 280, 560, 1120],
            value=280,
        ),
        gr.Textbox(label="System Prompt", value=""),
    ],
    additional_inputs_accordion=gr.Accordion("Settings", open=True),
    title="Gemma 4 E4B It",
    examples=examples,
    run_examples_on_click=False,
    cache_examples=False,
    delete_cache=(1800, 1800),
)

if __name__ == "__main__":
    demo.launch(css_paths="style.css", max_file_size="20MB")
