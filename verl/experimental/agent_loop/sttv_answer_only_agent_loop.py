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

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.sttv_no_verifier_agent_loop import (
    SttvNoVerifierAgentLoop,
)


@register("sttv_answer_only_agent")
class SttvAnswerOnlyAgentLoop(SttvNoVerifierAgentLoop):
    """Answer-only STTV loop: single call that outputs <reason> and <answer>."""

    def _build_prompted_context(self, query: str) -> str:
        query_text = query.strip()
        return (
            f"{query_text}\n\n"
            "Please answer the query by first reasoning inside <reason> tags and then putting ONLY your final answer "
            "inside <answer>. Ensure that the answer is either yes/no, one word, or one number. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Nothing else."
        )
