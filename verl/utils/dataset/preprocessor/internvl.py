# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocessor for InternVL models."""

from __future__ import annotations

import base64
import copy
import logging
import os
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from qwen_vl_utils import fetch_video

from .base_processor import BasicPreprocessor
from .registry import PREPROCESSOR_REGISTER

__all__ = ["InternVLPreprocessor"]


logger = logging.getLogger(__name__)


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


@PREPROCESSOR_REGISTER.register()
class InternVLPreprocessor(BasicPreprocessor):
    """Processor wrapper that normalises inputs for InternVL models."""

    def __init__(self, processor, image_key: str = "image", video_key: str = "video") -> None:
        super().__init__(processor, image_key=image_key, video_key=video_key)
        self.max_video_frames = self._read_env_int("VERL_INTERNVL_MAX_VIDEO_FRAMES", 12)
        self.min_video_frames = max(1, self._read_env_int("VERL_INTERNVL_MIN_VIDEO_FRAMES", 1))
        max_images = self._read_env_int("VERL_INTERNVL_MAX_IMAGES", 0)
        self.max_images = max_images if max_images > 0 else None
        self._warned_image_cap = False

    @staticmethod
    def _read_env_int(key: str, default: int) -> int:
        value = os.environ.get(key)
        if value is None:
            return default
        try:
            return max(int(value), 0)
        except ValueError:
            return default

    def process_image(self, image: Any, **_: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            image_obj = image
        elif isinstance(image, dict) and "bytes" in image:
            with BytesIO(image["bytes"]) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
        elif isinstance(image, dict) and "image" in image:
            # Some datasets keep the image path/value under the "image" key.
            return self.process_image(image["image"])
        elif isinstance(image, str) and image.startswith(("http://", "https://")):
            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                with BytesIO(response.content) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        elif isinstance(image, str) and image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif isinstance(image, str) and image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                with BytesIO(data) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
            else:
                raise ValueError("Unsupported data URI without base64 payload.")
        elif isinstance(image, str):
            image_obj = Image.open(image)
        else:
            raise TypeError(f"Unsupported image payload type: {type(image)}")

        return image_obj.convert("RGB")

    def process_video(self, video: Any, **kwargs: Any):
        nframes = kwargs.get("nframes")
        fps = kwargs.get("fps")
        fps_min_frames = kwargs.get("fps_min_frames")
        fps_max_frames = kwargs.get("fps_max_frames")

        if not isinstance(video, dict) or "video" not in video:
            raise NotImplementedError(VIDEO_FORMAT_HELP)
        if nframes is not None and fps is not None:
            raise AssertionError("Can't use both `nframes` and `fps`.")

        video_args = dict(video)

        contains_sampling_rules = "nframes" in video_args or "fps" in video_args
        if not contains_sampling_rules:
            if nframes is not None:
                video_args["nframes"] = nframes
            elif fps is not None:
                video_args["fps"] = fps
                if fps_min_frames is not None:
                    video_args["min_frames"] = fps_min_frames
                if fps_max_frames is not None:
                    video_args["max_frames"] = fps_max_frames

        max_frames_cap = self.max_video_frames if self.max_video_frames > 0 else None
        if max_frames_cap is not None:
            if "nframes" in video_args:
                video_args["nframes"] = min(video_args["nframes"], max_frames_cap)
            else:
                existing_max = video_args.get("max_frames")
                if existing_max is None:
                    video_args["max_frames"] = max_frames_cap
                else:
                    video_args["max_frames"] = min(existing_max, max_frames_cap)

                if "min_frames" in video_args:
                    video_args["min_frames"] = min(video_args["min_frames"], video_args["max_frames"])
                else:
                    video_args["min_frames"] = max(1, min(self.min_video_frames, video_args["max_frames"]))

                if "fps_max_frames" in video_args:
                    video_args["fps_max_frames"] = min(video_args["fps_max_frames"], video_args["max_frames"])
            # Ensure whichever sampling strategy is chosen respects the cap.
            if "fps_max_frames" in video_args:
                video_args["fps_max_frames"] = min(video_args["fps_max_frames"], max_frames_cap)

        return fetch_video(video_args)

    def process_audio(self, audio, **_: Any):
        raise ValueError("InternVL does not support audio inputs.")

    def __call__(self, messages, row_dict: dict):
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}

        images = None
        if self.image_key in row_dict:
            images = [self.process_image(image) for image in row_dict.pop(self.image_key)]
            if self.max_images is not None and len(images) > self.max_images:
                if not self._warned_image_cap:
                    logger.warning(
                        "InternVLPreprocessor: truncating images per sample to %s to keep prompt within context.",
                        self.max_images,
                    )
                    self._warned_image_cap = True
                images = images[: self.max_images]
            multi_modal_data["image"] = images

        videos = None
        if self.video_key in row_dict:
            videos = [self.process_video(video) for video in row_dict.pop(self.video_key)]
            multi_modal_data["video"] = [video.numpy() for video in videos]

        aligned_prompt = self._align_image_placeholders(raw_prompt, len(images) if images is not None else 0)
        processor_prompt = self._convert_placeholder_for_processor(aligned_prompt)

        model_inputs = self.processor(
            text=[processor_prompt],
            images=images,
            videos=videos,
            return_tensors="pt",
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        return row_dict, model_inputs, input_ids, attention_mask, aligned_prompt

    def _align_image_placeholders(self, prompt: str, num_images: int) -> str:
        canonical_placeholder = "<image>"
        tokenizer = getattr(self.processor, "tokenizer", None)
        context_token = None
        if tokenizer is not None:
            context_token = getattr(tokenizer, "context_image_token", None)

        processed = prompt
        legacy_tokens = {canonical_placeholder, "<IMG_CONTEXT>"}
        if context_token:
            legacy_tokens.add(context_token)
        for token in legacy_tokens:
            if token and token != canonical_placeholder:
                processed = processed.replace(token, canonical_placeholder)

        if num_images <= 0:
            return processed.replace(canonical_placeholder, "")

        count = processed.count(canonical_placeholder)
        if count == num_images:
            return processed

        if count > num_images:
            excess = count - num_images
            for _ in range(excess):
                idx = processed.rfind(canonical_placeholder)
                if idx == -1:
                    break
                processed = processed[:idx] + processed[idx + len(canonical_placeholder) :]
            return processed

        # count < num_images
        missing = num_images - count
        addition = "\n".join(canonical_placeholder for _ in range(missing))
        if count == 0:
            if processed.strip():
                processed = f"{addition}\n\n{processed.lstrip()}"
            else:
                processed = addition
        else:
            if processed and not processed.endswith("\n"):
                processed = processed + "\n"
            processed = processed + addition
        return processed

    def _convert_placeholder_for_processor(self, prompt: str) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        target_placeholder = "<image>"
        if tokenizer is not None:
            context_token = getattr(tokenizer, "context_image_token", None)
            if context_token:
                target_placeholder = context_token

        if target_placeholder == "<image>":
            return prompt

        return prompt.replace("<image>", target_placeholder)
