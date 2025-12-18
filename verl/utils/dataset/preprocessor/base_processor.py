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
"""Base class for multimodal preprocessors."""

from __future__ import annotations

from typing import Any, Tuple


class BasicPreprocessor:
    """Base preprocessor that wraps a Hugging Face Processor."""

    def __init__(self, processor, image_key: str = "image", video_key: str = "video", audio_key: str = "audio") -> None:
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
        self.audio_key = audio_key

    # The following methods are intentionally kept simple so subclasses can override them.
    def process_image(self, image, **_: Any):
        raise NotImplementedError("Subclasses must implement process_image.")

    def process_video(self, video, **_: Any):
        raise NotImplementedError("Subclasses must implement process_video.")

    def process_audio(self, audio, **_: Any):
        raise NotImplementedError("Subclasses must implement process_audio.")

    def __call__(self, messages, row_dict: dict) -> Tuple[dict, dict, Any, Any, str]:
        """Apply chat template, preprocess media, and build processor inputs."""

        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}

        images = None
        if self.image_key in row_dict:
            images = [self.process_image(image) for image in row_dict.pop(self.image_key)]
            multi_modal_data["image"] = images

        videos = None
        if self.video_key in row_dict:
            videos = [self.process_video(video) for video in row_dict.pop(self.video_key)]
            multi_modal_data["video"] = [video.numpy() for video in videos]

        model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        # Drop unused keys.
        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # Multi-modal inputs must be a plain dict, not a BatchFeature.
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        return row_dict, model_inputs, input_ids, attention_mask, raw_prompt
