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

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.sttv_agent_loop import SttvAgentLoop


@register("sttv_no_verifier_agent")
class SttvNoVerifierAgentLoop(SttvAgentLoop):
    """Single-pass STTV loop: one generation with no verifier calls."""

    def __init__(
        self,
        *args: Any,
        prompt_template_name: str = "sttv_no_verifier_single_turn.txt",
        no_verifier_max_new_tokens: int = 768,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        prompts_dir = self._resolve_prompts_dir()
        self.prompt_template = self._read_prompt_file(prompts_dir / prompt_template_name)
        self.no_verifier_max_new_tokens = int(no_verifier_max_new_tokens)

    def _build_prompted_context(self, query: str) -> str:
        # This loop uses a dedicated single-turn prompt template with 2 placeholders:
        # 1) bbox_2d-format instruction text, 2) task query.
        return self.prompt_template.format(self.instruction_text, query.strip())

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        raw_messages = list(kwargs["raw_prompt"])
        query = self._extract_query(raw_messages)
        multi_modal_data = await self.process_vision_info(raw_messages)
        raw_images = self._normalize_images(multi_modal_data.get("images"))
        images = [self._resize_longest_side(img, self.max_image_side) for img in raw_images]

        prompted_query = self._build_prompted_context(query)
        messages = self._build_messages(prompted_query, images)
        prompt_ids = await self.apply_chat_template(messages, images=images)

        metrics: dict[str, Any] = {}
        chunk, token_ids, log_probs = await self._generate_once(
            messages=messages,
            images=images,
            sampling_params=sampling_params,
            stop_sequences=["</answer>"],
            max_new_tokens=self.no_verifier_max_new_tokens,
            metrics=metrics,
        )

        response_ids = token_ids[: self.response_length]
        response_mask = [1] * len(response_ids)
        response_logprobs = None
        if sampling_params.get("logprobs"):
            if log_probs:
                response_logprobs = log_probs[: self.response_length]
            else:
                response_logprobs = [0.0] * len(response_ids)

        sttv_answer_mask = [1] * len(response_ids)
        sttv_loc_mask = [0] * len(response_ids)
        sttv_loc_calls: list[dict[str, Any]] = []

        for call_index, loc_span in enumerate(
            self._extract_loc_token_spans(
                chunk,
                chunk_start=0,
                chunk_token_count=len(token_ids),
            )
        ):
            span_start = max(0, int(loc_span.get("token_start", 0)))
            span_end = min(len(response_ids), int(loc_span.get("token_end", 0)))
            if span_end <= span_start:
                continue
            # Decouple objectives: do not optimize answer loss on bbox tokens.
            sttv_answer_mask[span_start:span_end] = [0] * (span_end - span_start)
            sttv_loc_mask[span_start:span_end] = [1] * (span_end - span_start)
            sttv_loc_calls.append(
                {
                    "call_index": call_index,
                    "token_start": span_start,
                    "token_end": span_end,
                    "text": loc_span.get("text", ""),
                    "span_type": loc_span.get("span_type", "chunk_fallback"),
                }
            )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={"images": images} if images else {},
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "sttv_answer_mask": sttv_answer_mask,
                "sttv_loc_mask": sttv_loc_mask,
                "sttv_loc_calls": sttv_loc_calls,
                "sttv_loc_verifier_calls": [],
            },
        )
