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

from pathlib import Path
import re
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.sttv_agent_loop import SttvAgentLoop

LOGIC_REASON_EDIT_PATTERN = re.compile(r"(?i)^EDIT_REASON\s*:\s*(?P<body>.+)$")
LOGIC_ANSWER_EDIT_PATTERN = re.compile(r"(?i)^EDIT_ANSWER\s*:\s*(?P<body>.+)$")


@register("sttv_all_verifiers_agent")
class SttvAllVerifiersAgentLoop(SttvAgentLoop):
    def __init__(
        self,
        *args: Any,
        logic_verifier_rounds: int = 2,
        logic_verifier_max_new_tokens: int = 96,
        logic_self_verifier_prompt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.logic_verifier_rounds = max(0, int(logic_verifier_rounds))
        self.logic_verifier_max_new_tokens = int(logic_verifier_max_new_tokens)

        if logic_self_verifier_prompt_path:
            prompt_path = Path(logic_self_verifier_prompt_path)
        else:
            prompt_path = self._resolve_training_prompts_dir() / "logic_self_verifier_instructions.txt"
        self.logic_self_verifier_template = self._read_prompt_file(prompt_path)

    def _resolve_training_prompts_dir(self) -> Path:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "training" / "prompts").exists():
                return parent / "training" / "prompts"
            if (parent / "STTV" / "training" / "prompts").exists():
                return parent / "STTV" / "training" / "prompts"
        raise FileNotFoundError("Could not locate training/prompts relative to the repo.")

    def _build_logic_self_verifier_prompt(self, query: str, latest_answer_output: str) -> str:
        return self.logic_self_verifier_template.format(
            query=query.strip(),
            answer=str(latest_answer_output or "").strip(),
        )

    def _build_clean_answer_prompt(self, query: str, latest_bbox_block: str) -> str:
        query_text = query.strip()
        return (
            f"Original query:\n{query_text}\n\n"
            "Detected objects (in [x_min, y_min, x_max, y_max] format with coordinates in [0,1000]):\n"
            f"{latest_bbox_block}\n\n"
            f"Here is the query again:\n{query_text}\n\n"
            "Please now answer the query by first reasoning inside <reason> tags and then putting ONLY your final "
            "answer inside <answer>. Do not round answers, express all ratios as unrounded decimals. "
            "Do not output another <bbox_2d>."
        )

    def _parse_logic_self_verifier_output(self, text: str) -> tuple[str, bool, dict[str, Any]]:
        cleaned = str(text or "").replace("<|im_end|>", "").strip()
        normalized_lines: list[tuple[str, int]] = []
        line_order = 0
        has_reason_edit = False
        has_answer_edit = False
        seen: set[str] = set()

        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if not line:
                continue
            if line.lower().startswith("feedback:"):
                line = line.split(":", 1)[1].strip()
                if not line:
                    continue

            reason_match = LOGIC_REASON_EDIT_PATTERN.match(line)
            if reason_match is not None:
                body = reason_match.group("body").strip()
                if body:
                    normalized = f"EDIT_REASON: {body}"
                    if normalized not in seen:
                        normalized_lines.append((normalized, line_order))
                        seen.add(normalized)
                        has_reason_edit = True
                        line_order += 1
                continue

            answer_match = LOGIC_ANSWER_EDIT_PATTERN.match(line)
            if answer_match is not None:
                body = answer_match.group("body").strip()
                if body:
                    normalized = f"EDIT_ANSWER: {body}"
                    if normalized not in seen:
                        normalized_lines.append((normalized, line_order))
                        seen.add(normalized)
                        has_answer_edit = True
                        line_order += 1
                continue

        normalized_lines.sort(key=lambda item: item[1])
        has_effect = len(normalized_lines) > 0
        feedback_info: dict[str, Any] = {
            "logic_feedback_has_effect": bool(has_effect),
            "logic_feedback_valid_for_reward": bool(has_effect),
            "logic_feedback_has_reason_edit": bool(has_reason_edit),
            "logic_feedback_has_answer_edit": bool(has_answer_edit),
        }
        if not has_effect:
            return "NO_VALID_REFINEMENTS. Re-emit the current <reason>/<answer> unchanged.", False, feedback_info
        return "\n".join(line for line, _ in normalized_lines), True, feedback_info

    async def _build_logic_verifier_messages(
        self,
        images: list[Image.Image],
        prompt: str,
        generate_logprobs: bool,
        metrics: dict[str, Any],
    ) -> tuple[str, list[int], list[int], list[float], list[dict[str, Any]], dict[str, Any]]:
        messages = self._build_messages(prompt, images)
        prompt_ids, logic_multi_modal_inputs = await self._build_prompt_ids_and_mm_inputs(messages, images)

        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": bool(generate_logprobs),
            "max_tokens": int(self.logic_verifier_max_new_tokens),
        }
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=images,
        )
        text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        output_ids = output.token_ids
        output_log_probs = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        if output_log_probs:
            logic_output_log_probs = [float(x) for x in output_log_probs]
        else:
            logic_output_log_probs = [0.0] * len(output_ids)
        serialized_images = [self._serialize_image_bytes(image) for image in images]
        return (
            text,
            prompt_ids,
            output_ids,
            logic_output_log_probs,
            serialized_images,
            logic_multi_modal_inputs,
        )

    async def _build_answer_rewrite_aux_messages(
        self,
        images: list[Image.Image],
        query: str,
        latest_bbox_block: str,
        current_answer_output: str,
        logic_feedback: str,
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[str, str, list[int], list[int], list[float], list[dict[str, Any]], dict[str, Any]]:
        clean_prompt = self._build_clean_answer_prompt(query, latest_bbox_block)
        rewrite_prompt = (
            f"{clean_prompt}\n\n"
            f"Current answer draft:\n{str(current_answer_output or '').strip()}\n\n"
            "I have some feedback for you to incorporate. "
            "Please output exactly one full <reason> block and then one full <answer> block that incorporates the feedback.\n"
            f"Feedback:\n{logic_feedback}\n"
            "You MUST re-predict BOTH the reasoning and the answer, including unchanged parts. "
            "You MUST incorporate the feedback and MUST NOT make unrelated changes. "
            "Please output exactly one full <reason> block and then one full <answer> block. "
            "Ensure that the answer is either yes/no, one word, or one number. "
            "Do not round answers, express all ratios as unrounded decimals. Nothing else. "
            "Do not output any <bbox_2d>."
        )
        messages = self._build_messages(rewrite_prompt, images)
        prompt_ids, answer_multi_modal_inputs = await self._build_prompt_ids_and_mm_inputs(messages, images)
        answer_sampling_params = dict(sampling_params)
        answer_sampling_params["max_tokens"] = self.max_new_tokens_per_chunk
        answer_sampling_params["stop"] = ["</answer>"]
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=answer_sampling_params,
            image_data=images,
        )
        output_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        output_ids = output.token_ids
        output_log_probs_raw = output.log_probs or []
        if metrics is not None and metrics.get("num_preempted", -1) == -1:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        if output_log_probs_raw:
            output_log_probs = [float(x) for x in output_log_probs_raw]
        else:
            output_log_probs = [0.0] * len(output_ids)
        serialized_images = [self._serialize_image_bytes(image) for image in images]
        return (
            rewrite_prompt,
            output_text,
            prompt_ids,
            output_ids,
            output_log_probs,
            serialized_images,
            answer_multi_modal_inputs,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        validate_mode = bool(kwargs.get("validate", False))
        raw_messages = list(kwargs["raw_prompt"])
        query = self._extract_query(raw_messages)
        multi_modal_data = await self.process_vision_info(raw_messages)
        raw_images = self._normalize_images(multi_modal_data.get("images"))
        raw_images_rgb = [img.convert("RGB") for img in raw_images]
        images = [self._resize_longest_side(img, self.max_image_side) for img in raw_images_rgb]

        prompted_query = self._build_prompted_context(query)
        messages = self._build_messages(prompted_query, images)
        initial_prompt_ids = await self.apply_chat_template(messages, images=images)
        model_prompt_ids = list(initial_prompt_ids)

        train_prompt_ids = list(initial_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: Optional[list[float]] = [] if sampling_params.get("logprobs") else None
        sttv_answer_mask: list[int] = []
        sttv_loc_mask: list[int] = []
        sttv_loc_calls: list[dict[str, Any]] = []
        sttv_loc_verifier_calls: list[dict[str, Any]] = []
        sttv_answer_calls: list[dict[str, Any]] = []
        sttv_answer_logic_verifier_calls: list[dict[str, Any]] = []
        sttv_answer_aux_call: Optional[dict[str, Any]] = None
        loc_call_counter = 0

        output_chunks: list[str] = []
        max_total_tokens = self.response_length
        total_generated_tokens = 0
        bbox_line_format = '1: label="object_name", [x_min, y_min, x_max, y_max]'

        metrics: dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": -1}
        user_turns = 1
        assistant_turns = 0

        def _extend_objective_masks(answer_value: int, loc_value: int, count: int) -> None:
            if count <= 0:
                return
            sttv_answer_mask.extend([answer_value] * count)
            sttv_loc_mask.extend([loc_value] * count)

        async def _append_user_turn(text: str) -> None:
            extra_prompt_ids = await self._append_user_message(
                messages,
                text,
                train_prompt_ids,
                response_mask,
                response_logprobs,
            )
            if self.agent_loop_cpu_cleanup_enable and extra_prompt_ids:
                model_prompt_ids.extend(extra_prompt_ids)
            appended = len(extra_prompt_ids)
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
                sttv_answer_mask[span_start:span_end] = [0] * (span_end - span_start)
                sttv_loc_mask[span_start:span_end] = [1] * (span_end - span_start)
                loc_text = str(loc_span.get("text", chunk))
                parsed_entries_1000: list[dict[str, Any]] = []
                payloads = self._extract_bbox_2d_payloads(loc_text)
                if len(payloads) == 1:
                    parsed_entries = self._parse_bbox_2d_entries(payloads[0])
                    if len(parsed_entries) > 0:
                        parsed_entries_1000 = [
                            {
                                "image_index": int(entry.image_index),
                                "label": str(entry.label),
                                "box_1000": [
                                    float(entry.coords[0]),
                                    float(entry.coords[1]),
                                    float(entry.coords[2]),
                                    float(entry.coords[3]),
                                ],
                            }
                            for entry in parsed_entries
                        ]
                sttv_loc_calls.append(
                    {
                        "call_index": loc_call_counter,
                        "token_start": span_start,
                        "token_end": span_end,
                        "text": loc_text,
                        "span_type": loc_span.get("span_type", "chunk_fallback"),
                        "parsed_entries_1000": parsed_entries_1000,
                    }
                )
                loc_call_counter += 1

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
                    "sttv_answer_aux_call": sttv_answer_aux_call,
                    "sttv_answer_calls": sttv_answer_calls,
                    "sttv_answer_logic_verifier_calls": sttv_answer_logic_verifier_calls,
                },
            )

        while True:
            if len(response_mask) >= self.response_length:
                return _return_output()

            if self.agent_loop_cpu_cleanup_enable:
                chunk, token_ids, log_probs = await self._generate_once_with_cached_prompt_ids(
                    prompt_ids=model_prompt_ids,
                    images=images,
                    sampling_params=sampling_params,
                    stop_sequences=["</bbox_2d>"],
                    max_new_tokens=self.max_new_tokens_per_chunk,
                    metrics=metrics,
                )
            else:
                chunk, token_ids, log_probs = await self._generate_once(
                    messages,
                    images=images,
                    sampling_params=sampling_params,
                    stop_sequences=["</bbox_2d>"],
                    max_new_tokens=self.max_new_tokens_per_chunk,
                    metrics=metrics,
                )
            if not token_ids and not chunk.strip():
                return _return_output()

            total_generated_tokens += len(token_ids)
            assistant_turns += 1
            _append_assistant_turn(chunk, token_ids, log_probs)
            if self.agent_loop_cpu_cleanup_enable and token_ids:
                model_prompt_ids.extend(token_ids)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

            if total_generated_tokens >= max_total_tokens:
                return _return_output()
            if "</answer>" in chunk:
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
                entries_by_image: dict[int, list[Any]] = {}
                for entry in current_entries:
                    entries_by_image.setdefault(entry.image_index, []).append(entry)

                originals: list[Image.Image] = []
                overlays: list[Image.Image] = []
                for image_idx in range(1, len(raw_images_rgb) + 1):
                    original = raw_images_rgb[image_idx - 1]
                    image_entries = entries_by_image.get(image_idx, [])
                    overlay = self._overlay_boxes(original, image_entries)
                    originals.append(original)
                    overlays.append(overlay)

                verifier_prompt = self._build_verifier_prompt(current_entries, len(images))
                (
                    verifier_output,
                    verifier_prompt_ids,
                    verifier_output_ids,
                    verifier_output_log_probs,
                    verifier_images,
                    verifier_multi_modal_inputs,
                ) = await self._build_verifier_messages(
                    originals,
                    overlays,
                    verifier_prompt,
                    generate_logprobs=bool(sampling_params.get("logprobs", False)),
                    metrics=metrics,
                )
                corrections, _, feedback_info = self._parse_verifier_feedback(verifier_output, current_entries)
                current_loc_call_index = sttv_loc_calls[-1]["call_index"] if sttv_loc_calls else -1
                if current_loc_call_index >= 0:
                    sttv_loc_verifier_calls.append(
                        {
                            "call_index": int(current_loc_call_index),
                            "round_index": int(verifier_round),
                            "corrections": str(corrections),
                            "verifier_prompt_text": verifier_prompt,
                            "verifier_output_text": verifier_output,
                            "verifier_prompt_token_ids": verifier_prompt_ids,
                            "verifier_output_token_ids": verifier_output_ids,
                            "verifier_output_log_probs": verifier_output_log_probs,
                            "verifier_images": verifier_images,
                            "verifier_multi_modal_inputs": verifier_multi_modal_inputs,
                            "sttv_loc_verifier_feedback_has_effect": bool(
                                feedback_info.get("feedback_has_effect", False)
                            ),
                            "sttv_loc_verifier_feedback_valid_for_reward": bool(
                                feedback_info.get("feedback_valid_for_reward", False)
                            ),
                            "sttv_loc_verifier_feedback_has_duplicate_add_existing": bool(
                                feedback_info.get("feedback_has_duplicate_add_existing", False)
                            ),
                            "sttv_loc_verifier_feedback_has_disallowed_remove": bool(
                                feedback_info.get("feedback_has_disallowed_remove", False)
                            ),
                            "sttv_loc_verifier_feedback_duplicate_add_existing_count": int(
                                feedback_info.get("feedback_duplicate_add_existing_count", 0)
                            ),
                            "sttv_loc_verifier_feedback_disallowed_remove_count": int(
                                feedback_info.get("feedback_disallowed_remove_count", 0)
                            ),
                        }
                    )

                await _append_user_turn(
                    (
                        "I have some feedback for you to incorporate. "
                        f"Please output ONLY one <bbox_2d> block using lines formatted as {bbox_line_format} "
                        "that incorporates the feedback.\n"
                        f"Feedback: {corrections}\n"
                        "You MUST re-predict ALL boxes, including unchanged ones, and keep indices sequential "
                        "starting at 1. You MUST incorporate the feedback and MUST NOT make unrelated changes."
                    )
                )

                if self.agent_loop_cpu_cleanup_enable:
                    correction_chunk, correction_token_ids, correction_log_probs = (
                        await self._generate_once_with_cached_prompt_ids(
                            prompt_ids=model_prompt_ids,
                            images=images,
                            sampling_params=sampling_params,
                            stop_sequences=["</bbox_2d>"],
                            max_new_tokens=self.max_new_tokens_per_chunk,
                            metrics=metrics,
                        )
                    )
                else:
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
                if self.agent_loop_cpu_cleanup_enable and correction_token_ids:
                    model_prompt_ids.extend(correction_token_ids)
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

            latest_bbox_block = self._format_bbox_block(current_entries)

            # Initial compacted answer call.
            (
                answer_prompt_text,
                answer_output_text,
                answer_prompt_token_ids,
                answer_output_token_ids,
                answer_output_log_probs,
                answer_images,
                answer_multi_modal_inputs,
            ) = await self._build_answer_aux_messages(
                images=images,
                query=query,
                latest_bbox_block=latest_bbox_block,
                sampling_params=sampling_params,
                metrics=metrics,
            )

            current_answer_output = str(answer_output_text or "")
            answer_call_index = 0
            current_answer_aux_record: dict[str, Any] = {
                "call_index": int(answer_call_index),
                "answer_prompt_text": answer_prompt_text,
                "answer_output_text": current_answer_output,
                "answer_prompt_token_ids": answer_prompt_token_ids,
                "answer_output_token_ids": answer_output_token_ids,
                "answer_output_log_probs": answer_output_log_probs,
                "answer_images": answer_images,
                "answer_multi_modal_inputs": answer_multi_modal_inputs,
                "answer_latest_bbox_block": latest_bbox_block,
                "answer_solution_str": f"{latest_bbox_block}\n{current_answer_output}",
            }
            sttv_answer_calls.append(
                {
                    "call_index": int(answer_call_index),
                    "answer_prompt_text": answer_prompt_text,
                    "answer_output_text": current_answer_output,
                    "answer_solution_str": current_answer_aux_record["answer_solution_str"],
                }
            )

            # Logic self-verifier rounds on reason/answer.
            for logic_round_index in range(self.logic_verifier_rounds):
                logic_prompt_text = self._build_logic_self_verifier_prompt(query, current_answer_output)
                (
                    logic_output_text,
                    logic_prompt_token_ids,
                    logic_output_token_ids,
                    logic_output_log_probs,
                    logic_images,
                    logic_multi_modal_inputs,
                ) = await self._build_logic_verifier_messages(
                    images=images,
                    prompt=logic_prompt_text,
                    generate_logprobs=bool(sampling_params.get("logprobs", False)),
                    metrics=metrics,
                )
                logic_feedback, logic_parse_valid, logic_feedback_info = self._parse_logic_self_verifier_output(
                    logic_output_text
                )
                if not logic_parse_valid:
                    logic_feedback = "No valid self-verifier feedback was produced. Re-emit the current answer unchanged."

                sttv_answer_logic_verifier_calls.append(
                    {
                        "round_index": int(logic_round_index),
                        "answer_call_index": int(answer_call_index),
                        "logic_feedback": str(logic_feedback),
                        "logic_feedback_parse_valid": bool(logic_parse_valid),
                        "logic_feedback_valid_for_reward": bool(
                            logic_feedback_info.get("logic_feedback_valid_for_reward", False)
                        ),
                        "logic_feedback_has_reason_edit": bool(
                            logic_feedback_info.get("logic_feedback_has_reason_edit", False)
                        ),
                        "logic_feedback_has_answer_edit": bool(
                            logic_feedback_info.get("logic_feedback_has_answer_edit", False)
                        ),
                        "logic_verifier_prompt_text": logic_prompt_text,
                        "logic_verifier_output_text": logic_output_text,
                        "logic_verifier_prompt_token_ids": logic_prompt_token_ids,
                        "logic_verifier_output_token_ids": logic_output_token_ids,
                        "logic_verifier_output_log_probs": logic_output_log_probs,
                        "logic_verifier_images": logic_images,
                        "logic_verifier_multi_modal_inputs": logic_multi_modal_inputs,
                    }
                )

                (
                    rewrite_prompt_text,
                    rewrite_output_text,
                    rewrite_prompt_token_ids,
                    rewrite_output_token_ids,
                    rewrite_output_log_probs,
                    rewrite_images,
                    rewrite_multi_modal_inputs,
                ) = await self._build_answer_rewrite_aux_messages(
                    images=images,
                    query=query,
                    latest_bbox_block=latest_bbox_block,
                    current_answer_output=current_answer_output,
                    logic_feedback=logic_feedback,
                    sampling_params=sampling_params,
                    metrics=metrics,
                )

                answer_call_index += 1
                current_answer_output = str(rewrite_output_text or "")
                current_answer_aux_record = {
                    "call_index": int(answer_call_index),
                    "answer_prompt_text": rewrite_prompt_text,
                    "answer_output_text": current_answer_output,
                    "answer_prompt_token_ids": rewrite_prompt_token_ids,
                    "answer_output_token_ids": rewrite_output_token_ids,
                    "answer_output_log_probs": rewrite_output_log_probs,
                    "answer_images": rewrite_images,
                    "answer_multi_modal_inputs": rewrite_multi_modal_inputs,
                    "answer_latest_bbox_block": latest_bbox_block,
                    "answer_solution_str": f"{latest_bbox_block}\n{current_answer_output}",
                }
                sttv_answer_calls.append(
                    {
                        "call_index": int(answer_call_index),
                        "answer_prompt_text": rewrite_prompt_text,
                        "answer_output_text": current_answer_output,
                        "answer_solution_str": current_answer_aux_record["answer_solution_str"],
                    }
                )

            final_answer_call = current_answer_aux_record
            sttv_answer_aux_call = dict(final_answer_call) if isinstance(final_answer_call, dict) else None

            if validate_mode and isinstance(final_answer_call, dict):
                final_output_token_ids = list(final_answer_call.get("answer_output_token_ids", []) or [])
                final_output_log_probs_raw = list(final_answer_call.get("answer_output_log_probs", []) or [])
                final_output_log_probs = [
                    float(x) if isinstance(x, (int, float)) else 0.0 for x in final_output_log_probs_raw
                ]
                final_output_text = str(final_answer_call.get("answer_output_text", "") or "")
                total_generated_tokens += len(final_output_token_ids)
                assistant_turns += 1
                _append_assistant_turn(final_output_text, final_output_token_ids, final_output_log_probs)
                if self.agent_loop_cpu_cleanup_enable and final_output_token_ids:
                    model_prompt_ids.extend(final_output_token_ids)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": final_output_text}]})
            return _return_output()
