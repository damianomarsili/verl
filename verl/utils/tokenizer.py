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
"""Utils for tokenization."""

import math
import os
import re
import warnings
from transformers import AutoConfig

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    trust_remote_code = True
    kwargs["trust_remote_code"] = trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)

    config = None
    try:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    except Exception:
        warnings.warn(
            f"Failed to load config for tokenizer '{name_or_path}'. InternVL specific tweaks are skipped.",
            stacklevel=1,
        )

    if config is not None and re.match(r"internvl", getattr(config, "model_type", ""), re.IGNORECASE):
        tokenizer.context_image_token = "<IMG_CONTEXT>"
        tokenizer.end_image_token = "</img>"
        tokenizer.start_image_token = "<img>"
        tokenizer.video_token = "<video>"
        # InternVL processors expect *_token_id attributes to be present.
        tokenizer.context_image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.context_image_token)
        tokenizer.start_image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.start_image_token)
        tokenizer.end_image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.end_image_token)
        tokenizer.video_token_id = tokenizer.convert_tokens_to_ids(tokenizer.video_token)
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + ''}}"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image' %}{{ '<image>' }}"
            "{% elif content['type'] == 'video' %}{{ '<video>' }}"
            "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "{{'<|im_end|>'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant' }}{% endif %}"
        )

    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def _get_internvl_processor_overrides(config=None):
    """Read optional InternVL processor overrides from environment variables.

    The defaults depend on the InternVL vision configuration so that the number of
    textual image context tokens matches the actual number of vision features.
    """

    def _int_env(key, default):
        try:
            return int(os.environ.get(key, default))
        except ValueError:
            return default

    vision_config = getattr(config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", 14)
    base_image_res = getattr(vision_config, "image_size", 448)
    force_image_size = getattr(config, "force_image_size", None)
    downsample_ratio = getattr(config, "downsample_ratio", 0.5)

    if isinstance(force_image_size, int) and force_image_size > 0:
        base_image_res = force_image_size

    max_patches = _int_env("VERL_INTERNVL_MAX_PATCHES", 1)
    if max_patches <= 0:
        max_patches = 1

    if max_patches <= 1:
        default_image_res = min(base_image_res, 320)
    else:
        default_image_res = base_image_res

    image_res = _int_env("VERL_INTERNVL_IMAGE_RES", default_image_res)
    # Ensure at least one patch per dimension before downsampling.
    patches_per_dim = max(image_res // patch_size, 1)
    default_seq_float = (patches_per_dim ** 2) * (downsample_ratio ** 2)
    default_seq_len = max(1, int(math.floor(default_seq_float + 1e-6)))
    image_seq_length = _int_env("VERL_INTERNVL_IMAGE_SEQ_LEN", default_seq_len)

    return image_seq_length, image_res, max_patches


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    trust_remote_code = True
    kwargs["trust_remote_code"] = trust_remote_code
    config = None
    try:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    except Exception:
        warnings.warn(
            f"Failed to load config for processor '{name_or_path}'. InternVL specific tweaks are skipped.",
            stacklevel=1,
        )

    try:
        if config is not None and re.match(r"internvl", getattr(config, "model_type", ""), re.IGNORECASE):
            from transformers.models.got_ocr2 import GotOcr2ImageProcessorFast
            from transformers.models.internvl import InternVLProcessor
            from transformers.models.internvl.video_processing_internvl import InternVLVideoProcessor

            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]

            image_seq_length, image_res, max_patches = _get_internvl_processor_overrides(config)
            if max_patches is None or max_patches <= 0:
                max_patches = 1

            image_processor = GotOcr2ImageProcessorFast(
                crop_to_patches=False,
                data_format="channels_first",
                default_to_square=True,
                    do_convert_rgb=True,
                    do_normalize=True,
                    do_rescale=True,
                    do_resize=True,
                    rescale_factor=0.00392156862745098,
                    size={"height": image_res, "width": image_res},
                    max_patches=max_patches,
                    min_patches=min(1, max_patches),
                    resample=3,
                    return_tensors=None,
                    image_mean=imagenet_mean,
                    image_std=imagenet_std,
            )
            video_processor = InternVLVideoProcessor()
            tokenizer = hf_tokenizer(name_or_path, trust_remote_code=trust_remote_code)
            processor = InternVLProcessor(
                image_processor=image_processor,
                image_seq_length=image_seq_length,
                tokenizer=tokenizer,
                chat_template=tokenizer.chat_template,
                video_processor=video_processor,
            )
        else:
            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
