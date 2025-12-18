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
"""Registry helpers for multimodal preprocessors."""

from __future__ import annotations

import re
from typing import Optional, Type

from .internvl import InternVLPreprocessor  # noqa: F401
from .registry import PREPROCESSOR_REGISTER


def map_processor_to_preprocessor(processor) -> Optional[Type]:
    """Return matching preprocessor class for a HF processor instance."""

    processor_name = processor.__class__.__name__
    if not processor_name.lower().endswith("processor"):
        raise ValueError(f"Source object '{processor_name}' is not a 'Processor'.")

    if re.match(r"internvl.*processor", processor_name, re.IGNORECASE):
        target_name = "InternVLPreprocessor".lower()
    else:
        return None

    return PREPROCESSOR_REGISTER.get(target_name)
