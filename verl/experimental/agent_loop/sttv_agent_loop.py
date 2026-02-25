# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BOX_COLOR = (0, 0, 255)
BOX_FILL_RGBA = (0, 0, 255, 25)
FONT_SCALE = 0.022
BOX_OUTLINE_SCALE = 0.005
LABEL_PADDING = 2
BBOX_2D_ENTRY_PATTERN = re.compile(
    r'^\s*label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*\[(?P<coords>[^\]]+)\]\s*$',
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LocEntry:
    image_index: int
    label: str
    coords: Tuple[float, ...]


@register("sttv_agent")
class SttvAgentLoop(AgentLoopBase):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        instruction_mode: str = "box",
        depth_enabled: bool = False,
        max_image_side: int = 768,
        verifier_image_side: int = 1024,
        loc_verifier_rounds: int = 3,
        verifier_max_new_tokens: int = 96,
        max_steps: int = 8,
        max_new_tokens_per_chunk: int = 256,
        max_final_answer_tokens: int = 64,
        **kwargs: Any,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

        self.instruction_mode = instruction_mode.lower()
        if self.instruction_mode != "box":
            raise ValueError(f"Only box mode is supported; got instruction_mode={instruction_mode}")
        self.depth_enabled = self._coerce_bool(depth_enabled)

        self.max_image_side = int(max_image_side)
        self.verifier_image_side = int(verifier_image_side)
        self.loc_verifier_rounds = max(0, int(loc_verifier_rounds))
        self.verifier_max_new_tokens = int(verifier_max_new_tokens)

        self.max_steps = int(max_steps)
        self.max_new_tokens_per_chunk = int(max_new_tokens_per_chunk)
        self.max_final_answer_tokens = int(max_final_answer_tokens)

        prompts_dir = self._resolve_prompts_dir()
        if self.depth_enabled:
            prompt_file = prompts_dir / "sttv_verifier_depth.txt"
        else:
            prompt_file = prompts_dir / "sttv_verifier_single_turn.txt"
        self.prompt_template = self._read_prompt_file(prompt_file)
        self.instruction_text = self._read_prompt_file(prompts_dir / "instructions_box.txt")
        self.depth_instruction_text = None
        if self.depth_enabled:
            depth_file = prompts_dir / "instructions_depth_box.txt"
            self.depth_instruction_text = self._read_prompt_file(depth_file)
        self.verifier_template = self._read_prompt_file(prompts_dir / "verifier_instructions.txt")

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if value is None:
            return False
        normalized = str(value).strip().lower()
        return normalized not in {"0", "false", "no", "off", ""}

    def _resolve_prompts_dir(self) -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "lmms-eval" / "prompts").exists():
                return parent / "lmms-eval" / "prompts"
            if (parent / "STTV" / "lmms-eval" / "prompts").exists():
                return parent / "STTV" / "lmms-eval" / "prompts"
        raise FileNotFoundError("Could not locate lmms-eval/prompts relative to the repo.")

    def _read_prompt_file(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Prompt file is empty: {path}")
        return text.replace("<plan>", "<reason>").replace("</plan>", "</reason>")

    def _build_prompted_context(self, query: str) -> str:
        if self.depth_enabled:
            if self.depth_instruction_text is None:
                raise ValueError("Depth mode enabled but depth instructions are missing.")
            return self.prompt_template.format(self.instruction_text, self.depth_instruction_text, query.strip())
        return self.prompt_template.format(self.instruction_text, query.strip())

    def _resize_longest_side(self, image: Image.Image, longest_side: int) -> Image.Image:
        width, height = image.size
        if max(width, height) <= longest_side:
            return image
        if width >= height:
            new_width = longest_side
            new_height = int(round(height * (longest_side / width)))
        else:
            new_height = longest_side
            new_width = int(round(width * (longest_side / height)))
        return image.resize((new_width, new_height), Image.BICUBIC)

    def _coerce_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict) and "bytes" in image:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _serialize_image_bytes(self, image: Image.Image) -> dict[str, Any]:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return {
            "bytes": buffer.getvalue(),
            "format": "PNG",
            "width": image.width,
            "height": image.height,
        }

    def _normalize_images(self, images: Optional[list[Any]]) -> list[Image.Image]:
        if not images:
            return []
        return [self._coerce_pil(img) for img in images]

    def _build_messages(self, prompt: str, images: list[Image.Image]) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        for image in images:
            resized = self._resize_longest_side(image.convert("RGB"), self.max_image_side)
            content.append({"type": "image", "image": resized})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _extract_query(self, messages: list[dict[str, object]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            text = self._extract_text_from_content(content)
            if text:
                return text.strip()
        return ""

    def _extract_text_from_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return ""

    def _load_font(self, image: Image.Image, scale: float = FONT_SCALE) -> ImageFont.ImageFont:
        width, height = image.size
        size = max(12, int(min(width, height) * scale))
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def _scale_point(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        x_px = int(round(x / 1000.0 * width))
        y_px = int(round(y / 1000.0 * height))
        return max(0, min(width - 1, x_px)), max(0, min(height - 1, y_px))

    def _measure_text(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        if hasattr(font, "getbbox"):
            left, top, right, bottom = font.getbbox(text)
            return right - left, bottom - top
        return font.getsize(text)

    def _overlay_boxes(self, image: Image.Image, entries: list[LocEntry]) -> Image.Image:
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._load_font(base)

        width, height = base.size
        outline_width = max(2, int(min(width, height) * BOX_OUTLINE_SCALE))
        idx = 1
        for entry in entries:
            x1, y1, x2, y2 = entry.coords[:4]
            left, top = self._scale_point(x1, y1, width, height)
            right, bottom = self._scale_point(x2, y2, width, height)
            if right < left:
                left, right = right, left
            if bottom < top:
                top, bottom = bottom, top
            draw.rectangle(
                (left, top, right, bottom),
                outline=BOX_COLOR,
                width=outline_width,
                fill=BOX_FILL_RGBA,
            )
            label = f"{idx}:{entry.label}" if entry.label else str(idx)
            text_w, text_h = self._measure_text(draw, label, font)
            text_x = min(width - text_w - LABEL_PADDING, max(0, left + LABEL_PADDING))
            text_y = min(height - text_h - LABEL_PADDING, max(0, top + LABEL_PADDING))
            draw.rectangle(
                (text_x - LABEL_PADDING, text_y - LABEL_PADDING, text_x + text_w + LABEL_PADDING, text_y + text_h),
                fill=(255, 255, 255, 200),
            )
            draw.text((text_x, text_y), label, fill=BOX_COLOR, font=font)
            idx += 1

        return Image.alpha_composite(base, overlay).convert("RGB")

    def _extract_bbox_2d_payloads(self, text: str) -> list[str]:
        if not text:
            return []
        payloads: list[str] = []
        for raw_payload in re.findall(r"(?is)<bbox_2d>(.*?)</bbox_2d>", text):
            payloads.append(str(raw_payload).strip())
        return payloads

    def _extract_loc_token_spans(
        self,
        chunk: str,
        *,
        chunk_start: int,
        chunk_token_count: int,
    ) -> list[dict[str, Any]]:
        """Extract token spans for <bbox_2d> calls in one assistant chunk.

        Uses char-level span to token-level conversion; falls back to chunk-level
        when an exact match cannot be established.
        """
        spans: list[dict[str, Any]] = []
        if chunk_token_count <= 0:
            return spans

        def _encode_len(text: str) -> int:
            if not text:
                return 0
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

        loc_matches = list(re.finditer(r"(?is)<bbox_2d>.*?</bbox_2d>", chunk))
        if not loc_matches and "<bbox_2d>" in chunk.lower():
            loc_matches = [None]

        if not loc_matches:
            return spans

        # Strict single-block format: if multiple blocks appear in one chunk,
        # treat them as one invalid span for one tool-call record.
        if len(loc_matches) > 1:
            first = loc_matches[0]
            last = loc_matches[-1]
            prefix = chunk[: first.start()]
            loc_text = chunk[first.start() : last.end()]
            prefix_tokens = _encode_len(prefix)
            loc_tokens = _encode_len(loc_text)
            token_start = chunk_start + prefix_tokens
            token_end = token_start + max(1, loc_tokens)
            chunk_end = chunk_start + chunk_token_count
            if token_start >= chunk_end:
                token_start = chunk_start
                token_end = chunk_end
                span_type = "chunk_fallback"
            else:
                token_end = min(token_end, chunk_end)
                span_type = "merged_multi_block"
            spans.append(
                {
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": loc_text,
                    "span_type": span_type,
                }
            )
            return spans

        for match in loc_matches:
            if match is None:
                token_start = chunk_start
                token_end = chunk_start + chunk_token_count
                spans.append(
                    {
                        "token_start": token_start,
                        "token_end": token_end,
                        "text": chunk,
                        "span_type": "chunk_fallback",
                    }
                )
                continue

            prefix = chunk[: match.start()]
            loc_text = chunk[match.start() : match.end()]
            prefix_tokens = _encode_len(prefix)
            loc_tokens = _encode_len(loc_text)
            token_start = chunk_start + prefix_tokens
            token_end = token_start + max(1, loc_tokens)

            chunk_end = chunk_start + chunk_token_count
            if token_start >= chunk_end:
                token_start = chunk_start
                token_end = chunk_end
                span_type = "chunk_fallback"
            else:
                token_end = min(token_end, chunk_end)
                span_type = "exact"

            spans.append(
                {
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": loc_text,
                    "span_type": span_type,
                }
            )

        return spans

    def _parse_bbox_2d_entries(self, payload: str) -> list[LocEntry]:
        entries: list[LocEntry] = []
        nonempty_line_count = 0
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            nonempty_line_count += 1
            match = BBOX_2D_ENTRY_PATTERN.fullmatch(line)
            if match is None:
                return []

            label = match.group("label").strip()
            if not label:
                return []

            numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group("coords"))
            if len(numbers) != 4:
                return []
            coords_tuple = tuple(float(n) for n in numbers[:4])

            entries.append(LocEntry(image_index=1, label=label, coords=coords_tuple))
        if nonempty_line_count == 0:
            return []
        return entries

    def _has_missing_label(self, payload: str) -> bool:
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            has_coords = bool(re.search(r"\[[^\]]*\]", line))
            has_label = bool(re.search(r'(?i)\blabel\s*=\s*"', line))
            if has_coords and not has_label:
                return True
        return False

    def _has_invalid_box(self, entries: list[LocEntry]) -> bool:
        for entry in entries:
            if len(entry.coords) < 4:
                continue
            x1, y1, x2, y2 = entry.coords[:4]
            if x2 < x1 or y2 < y1:
                return True
        return False

    def _build_verifier_prompt(self, entries: list[LocEntry], image_count: int) -> str:
        targets = sorted({entry.label for entry in entries})
        targets_str = ", ".join(targets) if targets else "(none)"

        lines: list[str] = []
        for i, entry in enumerate(entries, 1):
            prefix = f"image_{entry.image_index}, " if image_count > 1 else ""
            x1, y1, x2, y2 = entry.coords[:4]
            lines.append(f'{i}) {prefix}label="{entry.label}", [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]')

        preds_str = "\n".join(lines) if lines else "(none)"
        return self.verifier_template.format(targets=targets_str, preds=preds_str)

    def _parse_verifier_feedback(self, text: str) -> tuple[float, str, str]:
        cleaned = text.replace("<|im_end|>", "").strip()
        pq_score = 0.0
        corrections = ""
        payload_obj: Any = None

        candidates: list[str] = []
        fenced_match = re.search(r"(?is)```(?:json)?\s*(\{.*?\})\s*```", cleaned)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())
        brace_match = re.search(r"(?is)\{.*\}", cleaned)
        if brace_match:
            candidates.append(brace_match.group(0).strip())
        if cleaned:
            candidates.append(cleaned)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                payload_obj = parsed
                break

        if isinstance(payload_obj, dict):
            raw_pq = payload_obj.get("pq_score_estimate", payload_obj.get("pq_score", payload_obj.get("pq", 0.0)))
            try:
                pq_score = float(raw_pq)
            except (TypeError, ValueError):
                pq_score = 0.0
            pq_score = max(0.0, min(1.0, pq_score))

            raw_corrections = payload_obj.get("corrections", "")
            if isinstance(raw_corrections, list):
                corrections = " ".join(str(item).strip() for item in raw_corrections if str(item).strip())
            else:
                corrections = str(raw_corrections).strip()

        if not corrections:
            corrections = cleaned
        return pq_score, corrections, cleaned

    async def _generate_once(
        self,
        messages: list[dict[str, object]],
        images: list[Image.Image],
        sampling_params: dict[str, Any],
        stop_sequences: Optional[list[str]] = None,
        max_new_tokens: int = 256,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, list[int], list[float]]:
        chunk, token_ids, log_probs, _ = await self._generate_once_with_prompt_ids(
            messages=messages,
            images=images,
            sampling_params=sampling_params,
            stop_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
            metrics=metrics,
        )
        return chunk, token_ids, log_probs

    async def _generate_once_with_prompt_ids(
        self,
        messages: list[dict[str, object]],
        images: list[Image.Image],
        sampling_params: dict[str, Any],
        stop_sequences: Optional[list[str]] = None,
        max_new_tokens: int = 256,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, list[int], list[float], list[int]]:
        prompt_ids = await self.apply_chat_template(messages, images=images)
        if stop_sequences:
            sampling_params = dict(sampling_params)
            sampling_params["stop"] = stop_sequences
        sampling_params = dict(sampling_params)
        sampling_params["max_tokens"] = max_new_tokens
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
        )
        chunk = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        log_probs = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        return chunk, output.token_ids, log_probs, prompt_ids

    async def _append_user_message(
        self,
        messages: list[dict[str, object]],
        text: str,
        train_prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]],
        images: Optional[list[Image.Image]] = None,
    ) -> int:
        add_messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        messages.extend(add_messages)
        extra_prompt_ids = await self.apply_chat_template(
            add_messages, remove_system_prompt=True, images=None
        )
        train_prompt_ids.extend(extra_prompt_ids)
        response_mask.extend([0] * len(extra_prompt_ids))
        if response_logprobs is not None:
            response_logprobs.extend([0.0] * len(extra_prompt_ids))
        return len(extra_prompt_ids)

    def _append_assistant_tokens(
        self,
        output_chunks: list[str],
        chunk: str,
        token_ids: list[int],
        train_prompt_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]],
        log_probs: list[float],
    ) -> int:
        output_chunks.append(chunk)
        train_prompt_ids.extend(token_ids)
        response_mask.extend([1] * len(token_ids))
        if response_logprobs is not None:
            if log_probs:
                response_logprobs.extend(log_probs)
            else:
                response_logprobs.extend([0.0] * len(token_ids))
        return len(token_ids)

    async def _build_verifier_messages(
        self,
        originals: list[Image.Image],
        overlays: list[Image.Image],
        prompt: str,
        metrics: dict[str, Any],
    ) -> Tuple[str, list[int], list[int], list[dict[str, Any]]]:
        resized_originals = [self._resize_longest_side(img, self.verifier_image_side) for img in originals]
        resized_overlays = [self._resize_longest_side(img, self.verifier_image_side) for img in overlays]

        content: list[dict[str, object]] = []
        verifier_images: list[Image.Image] = []
        for original, overlay in zip(resized_originals, resized_overlays, strict=True):
            content.append({"type": "image", "image": original})
            content.append({"type": "image", "image": overlay})
            verifier_images.extend([original, overlay])
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        sampling_params = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "logprobs": False}
        text, output_ids, _, prompt_ids = await self._generate_once_with_prompt_ids(
            messages=messages,
            images=verifier_images,
            sampling_params=sampling_params,
            max_new_tokens=self.verifier_max_new_tokens,
            metrics=metrics,
        )
        serialized_images = [self._serialize_image_bytes(image) for image in verifier_images]
        return text, prompt_ids, output_ids, serialized_images

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_messages = list(kwargs["raw_prompt"])
        query = self._extract_query(raw_messages)
        multi_modal_data = await self.process_vision_info(raw_messages)
        raw_images = self._normalize_images(multi_modal_data.get("images"))
        # Resize images for model input so prompt/image token length reflects STTV_MAX_IMAGE_SIDE.
        images = [self._resize_longest_side(img, self.max_image_side) for img in raw_images]

        prompted_query = self._build_prompted_context(query)
        messages = self._build_messages(prompted_query, images)
        initial_prompt_ids = await self.apply_chat_template(messages, images=images)

        train_prompt_ids = list(initial_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: Optional[list[float]] = [] if sampling_params.get("logprobs") else None
        sttv_answer_mask: list[int] = []
        sttv_loc_mask: list[int] = []
        sttv_loc_calls: list[dict[str, Any]] = []
        sttv_loc_verifier_calls: list[dict[str, Any]] = []
        loc_call_counter = 0

        output_chunks: list[str] = []
        step_count = 0
        max_total_tokens = self.max_steps * self.max_new_tokens_per_chunk
        total_generated_tokens = 0
        max_empty_generation_attempts = 3
        empty_generation_failures = 0

        final_answer_prompt = "I can now predict the final answer which is: "
        final_fail_prompt = (
            "I am unable to locate the objects correctly. Provide a final <reason> step and then the final <answer>."
        )
        bbox_line_format = 'label="object_name", [x_min, y_min, x_max, y_max]'

        metrics: dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": -1}
        user_turns = 1
        assistant_turns = 0

        def _extend_objective_masks(answer_value: int, loc_value: int, count: int) -> None:
            if count <= 0:
                return
            sttv_answer_mask.extend([answer_value] * count)
            sttv_loc_mask.extend([loc_value] * count)

        async def _append_user_turn(text: str) -> None:
            appended = await self._append_user_message(
                messages,
                text,
                train_prompt_ids,
                response_mask,
                response_logprobs,
            )
            _extend_objective_masks(answer_value=0, loc_value=0, count=appended)

        def _append_verifier_injection(text: str) -> None:
            # Verifier injections are tool-like metadata: keep them in trajectory text,
            # but mask them out of response tokens and all objectives.
            output_chunks.append(text)
            token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            train_prompt_ids.extend(token_ids)
            response_mask.extend([0] * len(token_ids))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(token_ids))
            appended = len(token_ids)
            _extend_objective_masks(answer_value=0, loc_value=0, count=appended)

        def _append_assistant_turn(chunk: str, token_ids: list[int], log_probs: list[float]) -> None:
            nonlocal loc_call_counter
            chunk_start = len(response_mask)
            appended = self._append_assistant_tokens(
                output_chunks,
                chunk,
                token_ids,
                train_prompt_ids,
                response_mask,
                response_logprobs,
                log_probs,
            )
            _extend_objective_masks(answer_value=1, loc_value=0, count=appended)
            for loc_span in self._extract_loc_token_spans(
                chunk,
                chunk_start=chunk_start,
                chunk_token_count=appended,
            ):
                span_start = max(0, int(loc_span["token_start"]))
                span_end = min(len(sttv_loc_mask), int(loc_span["token_end"]))
                if span_end <= span_start:
                    continue
                # Decouple objectives: do not optimize answer loss on bbox tokens.
                sttv_answer_mask[span_start:span_end] = [0] * (span_end - span_start)
                sttv_loc_mask[span_start:span_end] = [1] * (span_end - span_start)
                sttv_loc_calls.append(
                    {
                        "call_index": loc_call_counter,
                        "token_start": span_start,
                        "token_end": span_end,
                        "text": loc_span.get("text", chunk),
                        "span_type": loc_span.get("span_type", "chunk_fallback"),
                    }
                )
                loc_call_counter += 1

        async def _inject_final_answer(prompt_prefix: str, max_new_tokens: int) -> str:
            nonlocal total_generated_tokens, assistant_turns, user_turns
            await _append_user_turn(prompt_prefix)
            user_turns += 1
            chunk, token_ids, log_probs = await self._generate_once(
                messages,
                images=images,
                sampling_params=sampling_params,
                stop_sequences=["</answer>"],
                max_new_tokens=max_new_tokens,
                metrics=metrics,
            )
            total_generated_tokens += len(token_ids)
            assistant_turns += 1
            _append_assistant_turn(chunk, token_ids, log_probs)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})
            return "".join(output_chunks)

        def _return_output() -> AgentLoopOutput:
            response_ids = train_prompt_ids[len(initial_prompt_ids) :]
            max_response_len = min(len(response_ids), self.response_length)
            clipped_loc_calls: list[dict[str, Any]] = []
            for call in sttv_loc_calls:
                token_start = max(0, int(call.get("token_start", 0)))
                token_end = min(max_response_len, int(call.get("token_end", 0)))
                if token_end <= token_start:
                    continue
                clipped = dict(call)
                clipped["token_start"] = token_start
                clipped["token_end"] = token_end
                clipped_loc_calls.append(clipped)
            return AgentLoopOutput(
                prompt_ids=initial_prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=(response_logprobs[: self.response_length] if response_logprobs is not None else None),
                multi_modal_data={"images": images} if images else {},
                num_turns=user_turns + assistant_turns,
                metrics=metrics,
                extra_fields={
                    "sttv_answer_mask": sttv_answer_mask[: self.response_length],
                    "sttv_loc_mask": sttv_loc_mask[: self.response_length],
                    "sttv_loc_calls": clipped_loc_calls,
                    "sttv_loc_verifier_calls": sttv_loc_verifier_calls,
                },
            )

        while True:
            if len(response_mask) >= self.response_length:
                return _return_output()

            chunk, token_ids, log_probs = await self._generate_once(
                messages,
                images=images,
                sampling_params=sampling_params,
                stop_sequences=["</bbox_2d>"],
                max_new_tokens=self.max_new_tokens_per_chunk,
                metrics=metrics,
            )
            if not token_ids and not chunk.strip():
                empty_generation_failures += 1
                if empty_generation_failures >= max_empty_generation_attempts:
                    await _inject_final_answer(final_fail_prompt, self.max_final_answer_tokens)
                    return _return_output()
            else:
                empty_generation_failures = 0

            total_generated_tokens += len(token_ids)
            assistant_turns += 1
            _append_assistant_turn(chunk, token_ids, log_probs)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

            if total_generated_tokens >= max_total_tokens:
                await _inject_final_answer(final_answer_prompt, self.max_final_answer_tokens)
                return _return_output()
            if "</answer>" in chunk:
                return _return_output()

            reason_steps = len(re.findall(r"(?is)<reason>.*?</reason>", chunk))
            if reason_steps == 0 and "<reason>" in chunk:
                reason_steps = 1
            depth_steps = 0
            if self.depth_enabled:
                depth_steps = len(re.findall(r"(?is)<depth>.*?</depth>", chunk))
                if depth_steps == 0 and "<depth>" in chunk:
                    depth_steps = 1
            step_count += reason_steps + depth_steps
            if step_count >= self.max_steps:
                await _inject_final_answer(final_answer_prompt, self.max_final_answer_tokens)
                return _return_output()

            loc_payloads = self._extract_bbox_2d_payloads(chunk)
            if len(loc_payloads) != 1:
                return _return_output()
            loc_payload = loc_payloads[0]

            if self._has_missing_label(loc_payload):
                return _return_output()

            entries = self._parse_bbox_2d_entries(loc_payload)
            if not entries:
                return _return_output()
            if self._has_invalid_box(entries):
                return _return_output()

            current_entries = entries
            for verifier_round in range(self.loc_verifier_rounds):
                entries_by_image: dict[int, list[LocEntry]] = {}
                for entry in current_entries:
                    entries_by_image.setdefault(entry.image_index, []).append(entry)

                originals: list[Image.Image] = []
                overlays: list[Image.Image] = []
                for image_idx in range(1, len(raw_images) + 1):
                    original = raw_images[image_idx - 1].convert("RGB")
                    image_entries = entries_by_image.get(image_idx, [])
                    overlay = self._overlay_boxes(original, image_entries)
                    originals.append(original)
                    overlays.append(overlay)

                verifier_prompt = self._build_verifier_prompt(current_entries, len(images))
                verifier_output, verifier_prompt_ids, verifier_output_ids, verifier_images = await self._build_verifier_messages(
                    originals,
                    overlays,
                    verifier_prompt,
                    metrics,
                )
                pq_score, corrections, _ = self._parse_verifier_feedback(verifier_output)
                current_loc_call_index = sttv_loc_calls[-1]["call_index"] if sttv_loc_calls else -1
                if current_loc_call_index >= 0:
                    sttv_loc_verifier_calls.append(
                        {
                            "call_index": int(current_loc_call_index),
                            "round_index": int(verifier_round),
                            "pq_score_estimate": float(pq_score),
                            "corrections": str(corrections),
                            "verifier_prompt_text": verifier_prompt,
                            "verifier_output_text": verifier_output,
                            "verifier_prompt_token_ids": verifier_prompt_ids,
                            "verifier_output_token_ids": verifier_output_ids,
                            "verifier_images": verifier_images,
                        }
                    )

                round_message = (
                    f"Verifier round {verifier_round + 1}/{self.loc_verifier_rounds}: "
                    f"estimated PQ={pq_score:.3f}. Corrections: {corrections}"
                )
                _append_verifier_injection(f"<verifier>{round_message}</verifier>")
                await _append_user_turn(
                    (
                        f"{round_message}\nRewrite your <bbox_2d> to improve the estimated PQ. "
                        f"Output ONLY one corrected <bbox_2d> block using lines formatted as {bbox_line_format}."
                    )
                )

                correction_chunk, correction_token_ids, correction_log_probs = await self._generate_once(
                    messages,
                    images=images,
                    sampling_params=sampling_params,
                    stop_sequences=["</bbox_2d>"],
                    max_new_tokens=self.max_new_tokens_per_chunk,
                    metrics=metrics,
                )
                total_generated_tokens += len(correction_token_ids)
                assistant_turns += 1
                _append_assistant_turn(correction_chunk, correction_token_ids, correction_log_probs)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": correction_chunk}]})

                corrected_payloads = self._extract_bbox_2d_payloads(correction_chunk)
                if len(corrected_payloads) != 1:
                    continue
                corrected_payload = corrected_payloads[0]
                if self._has_missing_label(corrected_payload):
                    continue
                corrected_entries = self._parse_bbox_2d_entries(corrected_payload)
                if not corrected_entries:
                    continue
                if self._has_invalid_box(corrected_entries):
                    continue
                current_entries = corrected_entries

            await _append_user_turn(
                "Grounding refinement is complete. Using your latest <bbox_2d>, output exactly one <reason> and then one <answer>. Do not output another <bbox_2d>."
            )
            final_chunk, final_token_ids, final_log_probs = await self._generate_once(
                messages,
                images=images,
                sampling_params=sampling_params,
                stop_sequences=["</answer>"],
                max_new_tokens=self.max_new_tokens_per_chunk,
                metrics=metrics,
            )
            total_generated_tokens += len(final_token_ids)
            assistant_turns += 1
            _append_assistant_turn(final_chunk, final_token_ids, final_log_probs)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": final_chunk}]})
            return _return_output()
