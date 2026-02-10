# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
from pprint import pprint
from typing import Any, Callable, Optional, Sequence

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn, load_extern_object
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
STTV_AUX_MM_CHUNK_SIZE = 8
STTV_PAD_RATIO_WARN_THRESHOLD = 0.1


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _extract_final_answer(self, text: str) -> str:
        if not text:
            return ""
        open_matches = list(re.finditer(r"(?is)<answer>", text))
        if not open_matches:
            return ""
        match = open_matches[-1]
        content_start = match.end()
        close_match = re.search(r"(?is)</answer>", text[content_start:])
        if close_match:
            content_end = content_start + close_match.start()
            return text[content_start:content_end].strip()
        next_tag = re.search(r"(?is)<(reason|depth|loc|verifier|answer)>", text[content_start:])
        if next_tag:
            content_end = content_start + next_tag.start()
            return text[content_start:content_end].strip()
        return text[content_start:].strip()

    def _extract_query_and_image(self, raw_prompt: object) -> tuple[str, object]:
        query_text = ""
        image_entry: object = None
        if isinstance(raw_prompt, list):
            for message in raw_prompt:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    query_text = content.replace("<image>", "").strip()
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                        elif item.get("type") == "image" and image_entry is None:
                            image_entry = item
                    query_text = "".join(text_parts).strip()
                break
        return query_text, image_entry

    def _format_image_for_wandb(self, image_entry: object):
        if not isinstance(image_entry, dict):
            return ""
        if "path" in image_entry:
            return image_entry.get("path", "")
        image_obj = image_entry.get("image")
        if image_obj is None and "bytes" in image_entry:
            try:
                from io import BytesIO
                from PIL import Image

                image_obj = Image.open(BytesIO(image_entry["bytes"]))
            except Exception:
                image_obj = None
        if image_obj is None:
            return ""
        import wandb

        return wandb.Image(image_obj)

    def _select_sample_indices(self, n: int, fraction: float) -> list[int]:
        if n <= 0:
            return []
        if fraction >= 1:
            return list(range(n))
        if fraction <= 0:
            return []
        stride = max(1, int(round(1 / fraction)))
        indices = list(range(0, n, stride))
        if not indices:
            return [0]
        return indices

    def _maybe_log_sample_table(
        self,
        split: str,
        raw_prompts: list[object],
        outputs: list[str],
        gts: list[object],
        scores: list[float],
    ) -> None:
        loggers = self.config.trainer.logger
        if isinstance(loggers, str):
            loggers = [loggers]

        if "wandb" not in loggers:
            return

        fraction = float(self.config.trainer.get("log_sample_fraction", 0.0))
        if fraction <= 0:
            return

        if self.config.trainer.get("log_sample_every_n_steps", 1) > 1:
            step_mod = self.config.trainer.get("log_sample_every_n_steps", 1)
            if self.global_steps % step_mod != 0:
                return

        n = min(len(outputs), len(scores), len(gts), len(raw_prompts))
        if n <= 0:
            return

        indices = self._select_sample_indices(n, fraction)

        import wandb

        columns = [
            "step",
            "split",
            "query",
            "image",
            "output",
            "final_answer",
            "ground_truth",
            "reward",
        ]
        rows = []
        for idx in indices:
            raw_prompt = raw_prompts[idx]
            query, image_entry = self._extract_query_and_image(raw_prompt)
            if not query:
                query = ""
            image_value = self._format_image_for_wandb(image_entry)
            output_text = outputs[idx]
            final_answer = self._extract_final_answer(output_text)
            rows.append(
                [
                    self.global_steps,
                    split,
                    query,
                    image_value,
                    output_text,
                    final_answer,
                    gts[idx],
                    scores[idx],
                ]
            )

        table = wandb.Table(columns=columns, data=rows)
        wandb.log({f"{split}/samples": table}, step=self.global_steps)

    def _is_sttv_multi_objective_enabled(self) -> bool:
        cfg = self.config.algorithm.get("sttv_multi_objective", {})
        return bool(cfg.get("enable", False))

    def _get_sttv_multi_objective_weights(self) -> dict[str, float]:
        cfg = self.config.algorithm.get("sttv_multi_objective", {})
        return {
            "answer": float(cfg.get("answer_weight", 1.0)),
            "loc": float(cfg.get("loc_weight", 1.0)),
            "loc_verifier": float(cfg.get("loc_verifier_weight", 1.0)),
        }

    def _load_sttv_reward_functions(self) -> dict[str, Optional[Callable[..., Any]]]:
        if hasattr(self, "_sttv_reward_fn_cache"):
            return self._sttv_reward_fn_cache

        reward_cfg = self.config.get("custom_reward_function") or {}
        module_path = reward_cfg.get("path")
        loc_fn_name = reward_cfg.get("loc_call_name", "compute_loc_call_rewards_batched")
        loc_verifier_fn_name = reward_cfg.get("loc_verifier_name", "compute_loc_verifier_rewards_batched")
        reward_fns: dict[str, Optional[Callable[..., Any]]] = {
            "loc": None,
            "loc_verifier": None,
        }
        if module_path:
            for key, fn_name in (("loc", loc_fn_name), ("loc_verifier", loc_verifier_fn_name)):
                try:
                    reward_fns[key] = load_extern_object(module_path=module_path, object_name=fn_name)
                except Exception as exc:
                    print(f"[sttv] Optional reward fn '{fn_name}' not loaded from {module_path}: {exc}")

        self._sttv_reward_fn_cache = reward_fns
        return reward_fns

    def _extract_sttv_mask_tensor(
        self,
        batch: DataProto,
        field_name: str,
        *,
        fallback_to_response_mask: bool = False,
    ) -> torch.Tensor:
        response_mask = batch.batch["response_mask"]
        bsz, response_len = response_mask.shape
        mask = torch.zeros_like(response_mask)

        raw_masks = batch.non_tensor_batch.get(field_name)
        if raw_masks is None:
            return response_mask.clone() if fallback_to_response_mask else mask

        for row_idx in range(min(len(raw_masks), bsz)):
            raw_row = raw_masks[row_idx]
            if raw_row is None:
                continue
            if isinstance(raw_row, torch.Tensor):
                row_values = raw_row.detach().cpu().tolist()
            elif isinstance(raw_row, np.ndarray):
                row_values = raw_row.tolist()
            else:
                row_values = list(raw_row)
            if not row_values:
                continue
            row_tensor = torch.tensor(row_values, dtype=mask.dtype, device=mask.device)
            take = min(response_len, row_tensor.numel())
            if take > 0:
                mask[row_idx, :take] = row_tensor[:take]

        return mask * response_mask.to(mask.dtype)

    def _extract_reward_context(
        self,
        batch: DataProto,
        *,
        include_solution_strs: bool = True,
    ) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
        bsz = len(batch)
        data_sources_raw = batch.non_tensor_batch.get("data_source")
        if isinstance(data_sources_raw, np.ndarray):
            data_sources = [str(x) for x in data_sources_raw.tolist()]
        elif data_sources_raw is None:
            data_sources = ["unknown"] * bsz
        else:
            data_sources = [str(x) for x in list(data_sources_raw)]
        if len(data_sources) < bsz:
            data_sources.extend(["unknown"] * (bsz - len(data_sources)))

        if include_solution_strs:
            solution_strs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch.batch["responses"]]
        else:
            solution_strs = [""] * bsz

        ground_truths: list[str] = []
        extra_infos: list[dict[str, Any]] = []
        for item in batch:
            reward_model = item.non_tensor_batch.get("reward_model", {})
            ground_truth = ""
            if isinstance(reward_model, dict):
                raw_gt = reward_model.get("ground_truth", "")
                if raw_gt is not None:
                    ground_truth = str(raw_gt)
            ground_truths.append(ground_truth)

            extra_info = item.non_tensor_batch.get("extra_info", {})
            if not isinstance(extra_info, dict):
                extra_info = {}
            merged_extra = dict(extra_info)
            if isinstance(reward_model, dict):
                for key, value in reward_model.items():
                    if key == "ground_truth":
                        continue
                    merged_extra.setdefault(key, value)
            extra_infos.append(merged_extra)

        return data_sources, solution_strs, ground_truths, extra_infos

    def _normalize_per_sample_call_rewards(
        self,
        raw_rewards: Any,
        call_records_per_sample: Sequence[list[dict[str, Any]]],
    ) -> list[list[float]]:
        expected_per_sample = [len(records) for records in call_records_per_sample]
        default_rewards = [[0.0] * n for n in expected_per_sample]

        if raw_rewards is None:
            return default_rewards
        if isinstance(raw_rewards, np.ndarray):
            raw_rewards = raw_rewards.tolist()
        if not isinstance(raw_rewards, (list, tuple)):
            return default_rewards

        # Preferred shape: List[List[float]] with batch-aligned outer dimension.
        if len(raw_rewards) == len(call_records_per_sample):
            normalized: list[list[float]] = []
            for sample_rewards, expected_n in zip(raw_rewards, expected_per_sample, strict=True):
                if expected_n == 0:
                    normalized.append([])
                    continue
                if isinstance(sample_rewards, np.ndarray):
                    sample_rewards = sample_rewards.tolist()
                if isinstance(sample_rewards, (list, tuple)):
                    row = [float(x) for x in sample_rewards[:expected_n]]
                    if len(row) < expected_n:
                        row.extend([0.0] * (expected_n - len(row)))
                    normalized.append(row)
                    continue
                if sample_rewards is None:
                    normalized.append([0.0] * expected_n)
                    continue
                normalized.append([float(sample_rewards)] * expected_n)
            return normalized

        # Fallback shape: flattened rewards over all calls.
        if all(not isinstance(x, (list, tuple, np.ndarray, dict)) for x in raw_rewards):
            total_calls = sum(expected_per_sample)
            flat = [float(x) for x in raw_rewards[:total_calls]]
            if len(flat) < total_calls:
                flat.extend([0.0] * (total_calls - len(flat)))
            cursor = 0
            normalized = []
            for expected_n in expected_per_sample:
                normalized.append(flat[cursor : cursor + expected_n])
                cursor += expected_n
            return normalized

        return default_rewards

    def _normalize_flat_rewards(self, raw_rewards: Any, expected_len: int) -> list[float]:
        if expected_len <= 0:
            return []
        if raw_rewards is None:
            return [0.0] * expected_len
        if isinstance(raw_rewards, np.ndarray):
            raw_rewards = raw_rewards.tolist()
        if isinstance(raw_rewards, (list, tuple)):
            flat: list[float] = []
            for value in raw_rewards:
                if isinstance(value, (list, tuple, np.ndarray)):
                    if len(value) == 0:
                        flat.append(0.0)
                    else:
                        flat.append(float(list(value)[0]))
                elif value is None:
                    flat.append(0.0)
                else:
                    flat.append(float(value))
            flat = flat[:expected_len]
            if len(flat) < expected_len:
                flat.extend([0.0] * (expected_len - len(flat)))
            return flat
        try:
            scalar = float(raw_rewards)
        except Exception:
            scalar = 0.0
        return [scalar] * expected_len

    def _relocate_scalar_scores_to_mask(
        self,
        token_level_scores: torch.Tensor,
        objective_mask: torch.Tensor,
        fallback_mask: torch.Tensor,
    ) -> torch.Tensor:
        relocated = torch.zeros_like(token_level_scores)
        sample_scores = token_level_scores.sum(dim=-1)
        _, response_len = token_level_scores.shape
        for row_idx in range(token_level_scores.shape[0]):
            active = torch.nonzero(objective_mask[row_idx] > 0, as_tuple=False)
            if active.numel() > 0:
                endpoint = int(active[-1, 0].item())
            else:
                fallback = torch.nonzero(fallback_mask[row_idx] > 0, as_tuple=False)
                endpoint = int(fallback[-1, 0].item()) if fallback.numel() > 0 else response_len - 1
            relocated[row_idx, endpoint] = sample_scores[row_idx]
        return relocated

    def _compute_sttv_loc_call_reward_tensor(
        self,
        batch: DataProto,
        reward_fn: Optional[Callable[..., Any]],
    ) -> tuple[torch.Tensor, int]:
        response_mask = batch.batch["response_mask"]
        response_len = response_mask.shape[1]
        loc_rewards = torch.zeros_like(response_mask, dtype=torch.float32)

        raw_calls = batch.non_tensor_batch.get("sttv_loc_calls")
        if raw_calls is None:
            return loc_rewards, 0

        loc_call_records: list[list[dict[str, Any]]] = []
        for row in raw_calls:
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if not isinstance(row, (list, tuple)):
                loc_call_records.append([])
                continue
            loc_call_records.append([entry for entry in row if isinstance(entry, dict)])

        total_calls = sum(len(records) for records in loc_call_records)
        if total_calls == 0:
            return loc_rewards, 0

        reward_values = [[0.0] * len(records) for records in loc_call_records]
        if reward_fn is not None:
            data_sources, _, ground_truths, extra_infos = self._extract_reward_context(
                batch, include_solution_strs=False
            )
            solution_strs = [""] * len(batch)
            uids_raw = batch.non_tensor_batch.get("uid", np.array(["unknown"] * len(batch), dtype=object))
            if isinstance(uids_raw, np.ndarray):
                uids = [str(x) for x in uids_raw.tolist()]
            else:
                uids = [str(x) for x in list(uids_raw)]
            if len(uids) < len(batch):
                uids.extend(["unknown"] * (len(batch) - len(uids)))

            verifier_calls_raw = batch.non_tensor_batch.get("sttv_loc_verifier_calls")
            loc_verifier_call_records: list[list[dict[str, Any]]] = []
            if verifier_calls_raw is None:
                loc_verifier_call_records = [[] for _ in range(len(batch))]
            else:
                for row in verifier_calls_raw:
                    if isinstance(row, np.ndarray):
                        row = row.tolist()
                    if not isinstance(row, (list, tuple)):
                        loc_verifier_call_records.append([])
                        continue
                    loc_verifier_call_records.append([entry for entry in row if isinstance(entry, dict)])
                if len(loc_verifier_call_records) < len(batch):
                    loc_verifier_call_records.extend([[] for _ in range(len(batch) - len(loc_verifier_call_records))])
            try:
                raw_reward_values = reward_fn(
                    data_sources=data_sources,
                    loc_call_records=loc_call_records,
                    solution_strs=solution_strs,
                    ground_truths=ground_truths,
                    extra_infos=extra_infos,
                    uids=uids,
                    loc_verifier_call_records=loc_verifier_call_records,
                )
            except TypeError:
                raw_reward_values = reward_fn(data_sources, loc_call_records, solution_strs, ground_truths, extra_infos)
            reward_values = self._normalize_per_sample_call_rewards(raw_reward_values, loc_call_records)

        for row_idx, (call_records, call_rewards) in enumerate(zip(loc_call_records, reward_values, strict=True)):
            for call_record, reward in zip(call_records, call_rewards, strict=True):
                start = int(call_record.get("token_start", 0))
                end = int(call_record.get("token_end", 0))
                if end <= start:
                    continue
                start = max(0, min(response_len - 1, start))
                end = max(start + 1, min(response_len, end))
                endpoint = end - 1
                loc_rewards[row_idx, endpoint] += float(reward)

        return loc_rewards, total_calls

    def _deserialize_sttv_images(self, serialized_images: Any) -> list[Image.Image]:
        if not isinstance(serialized_images, (list, tuple)):
            return []
        images: list[Image.Image] = []
        for image_blob in serialized_images:
            if not isinstance(image_blob, dict):
                continue
            image_bytes = image_blob.get("bytes")
            if image_bytes is None:
                continue
            try:
                images.append(Image.open(BytesIO(image_bytes)).convert("RGB"))
            except Exception:
                continue
        return images

    def _compute_aux_multi_modal_inputs(
        self,
        text: str,
        images: list[Image.Image],
    ) -> dict[str, torch.Tensor]:
        if self.processor is None or not images:
            return {}
        multi_modal_inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _split_aux_batched_processor_output(
        self,
        batched_outputs: dict[str, torch.Tensor],
        image_counts: list[int],
    ) -> list[dict[str, torch.Tensor]]:
        batch_size = len(image_counts)
        total_images = sum(image_counts)
        per_item_outputs: list[dict[str, torch.Tensor]] = [{} for _ in range(batch_size)]

        image_grid_thw_all = batched_outputs.get("image_grid_thw")
        per_item_patch_counts: list[int] | None = None
        if isinstance(image_grid_thw_all, torch.Tensor):
            per_item_patch_counts = []
            grid_cursor = 0
            for image_count in image_counts:
                if image_count <= 0:
                    per_item_patch_counts.append(0)
                    continue
                item_grid = image_grid_thw_all[grid_cursor : grid_cursor + image_count]
                patch_count = int((item_grid[:, 0] * item_grid[:, 1] * item_grid[:, 2]).sum().item())
                per_item_patch_counts.append(patch_count)
                grid_cursor += image_count

        for key, value in batched_outputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() == 0:
                for item in per_item_outputs:
                    item[key] = value.clone()
                continue
            if value.shape[0] == batch_size:
                for idx in range(batch_size):
                    per_item_outputs[idx][key] = value[idx : idx + 1].contiguous()
                continue
            if total_images > 0 and value.shape[0] == total_images:
                cursor = 0
                for idx, image_count in enumerate(image_counts):
                    per_item_outputs[idx][key] = value[cursor : cursor + image_count].contiguous()
                    cursor += image_count
                continue
            if (
                key in {"pixel_values", "pixel_values_videos"}
                and per_item_patch_counts is not None
                and value.shape[0] == sum(per_item_patch_counts)
            ):
                cursor = 0
                for idx, patch_count in enumerate(per_item_patch_counts):
                    per_item_outputs[idx][key] = value[cursor : cursor + patch_count].contiguous()
                    cursor += patch_count
                continue
            raise ValueError(
                f"Unsupported batched multimodal tensor shape for key '{key}': "
                f"{tuple(value.shape)} with batch_size={batch_size}, total_images={total_images}"
            )

        for item in per_item_outputs:
            image_grid_thw = item.get("image_grid_thw")
            if image_grid_thw is not None:
                images_seqlens = torch.repeat_interleave(
                    image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
                )
                item["images_seqlens"] = images_seqlens
        return per_item_outputs

    def _compute_aux_multi_modal_inputs_batched(
        self,
        texts: list[str],
        images_per_row: list[list[Image.Image]],
        chunk_size: int = STTV_AUX_MM_CHUNK_SIZE,
    ) -> tuple[list[dict[str, torch.Tensor]], int]:
        if self.processor is None:
            return ([{} for _ in texts], 0)

        outputs: list[dict[str, torch.Tensor]] = [{} for _ in texts]
        fallback_rows = 0
        for start in range(0, len(texts), max(1, int(chunk_size))):
            end = min(len(texts), start + max(1, int(chunk_size)))
            chunk_texts = texts[start:end]
            chunk_images = images_per_row[start:end]
            image_counts = [len(images) for images in chunk_images]
            try:
                batched_multi_modal_inputs = self.processor(
                    text=chunk_texts,
                    images=chunk_images,
                    return_tensors="pt",
                    do_sample_frames=False,
                )
                batched_multi_modal_inputs.pop("input_ids", None)
                batched_multi_modal_inputs.pop("attention_mask", None)
                batched_multi_modal_inputs = dict(batched_multi_modal_inputs.convert_to_tensors("pt"))
                chunk_outputs = self._split_aux_batched_processor_output(
                    batched_outputs=batched_multi_modal_inputs,
                    image_counts=image_counts,
                )
                for idx, chunk_output in enumerate(chunk_outputs):
                    outputs[start + idx] = chunk_output
            except Exception:
                fallback_rows += end - start
                for local_idx, (text, images) in enumerate(zip(chunk_texts, chunk_images, strict=True)):
                    outputs[start + local_idx] = self._compute_aux_multi_modal_inputs(text=text, images=images)

        return outputs, fallback_rows

    def _compute_aux_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.processor is None or ("image_grid_thw" not in multi_modal_inputs and "video_grid_thw" not in multi_modal_inputs):
            return compute_position_id_with_mask(attention_mask)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) -> (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, input_ids.shape[1]), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        return torch.cat((text_position_ids, vision_position_ids), dim=1)

    def _build_sttv_loc_verifier_aux_batch(
        self, batch: DataProto
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        verifier_calls_raw = batch.non_tensor_batch.get("sttv_loc_verifier_calls")
        if verifier_calls_raw is None:
            return None, {
                "sttv/aux_prompt_len": 0.0,
                "sttv/aux_response_len": 0.0,
                "sttv/aux_rows_dropped_no_images": 0.0,
                "sttv/aux_multimodal_fallback_rows": 0.0,
            }

        max_prompt_len = int(self.config.actor_rollout_ref.rollout.prompt_length)
        max_response_len = int(self.config.actor_rollout_ref.rollout.response_length)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        data_sources, _, ground_truths, extra_infos = self._extract_reward_context(
            batch, include_solution_strs=False
        )

        aux_rows: list[dict[str, Any]] = []
        dropped_no_images = 0

        for row_idx in range(len(batch)):
            uid = str(batch.non_tensor_batch["uid"][row_idx])
            call_list = verifier_calls_raw[row_idx]
            if isinstance(call_list, np.ndarray):
                call_list = call_list.tolist()
            if not isinstance(call_list, (list, tuple)):
                continue

            for call_record in call_list:
                if not isinstance(call_record, dict):
                    continue

                prompt_text = str(call_record.get("verifier_prompt_text", "") or "")
                output_text = str(call_record.get("verifier_output_text", "") or "")

                prompt_token_ids = list(call_record.get("verifier_prompt_token_ids", []) or [])
                if len(prompt_token_ids) == 0:
                    if prompt_text:
                        prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

                output_token_ids = list(call_record.get("verifier_output_token_ids", []) or [])
                if len(output_token_ids) == 0:
                    if output_text:
                        output_token_ids = self.tokenizer(output_text, add_special_tokens=False)["input_ids"]
                if len(output_token_ids) == 0:
                    continue

                if len(prompt_token_ids) > max_prompt_len:
                    prompt_token_ids = prompt_token_ids[-max_prompt_len:]
                if len(output_token_ids) > max_response_len:
                    output_token_ids = output_token_ids[:max_response_len]

                images = self._deserialize_sttv_images(call_record.get("verifier_images", []))
                if self.processor is not None and len(images) == 0:
                    # Verifier calls in VLM mode require image tensors for image placeholder tokens.
                    dropped_no_images += 1
                    continue

                call_index = int(call_record.get("call_index", -1))
                if not prompt_text and prompt_token_ids:
                    prompt_text = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
                if not output_text and output_token_ids:
                    output_text = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
                aux_rows.append(
                    {
                        "uid": uid,
                        "call_index": call_index,
                        "call_record": call_record,
                        "data_source": str(data_sources[row_idx]),
                        "ground_truth": str(ground_truths[row_idx]),
                        "extra_info": extra_infos[row_idx],
                        "prompt_token_ids": prompt_token_ids,
                        "output_token_ids": output_token_ids,
                        "prompt_text": prompt_text,
                        "output_text": output_text,
                        "images": images,
                    }
                )

        if len(aux_rows) == 0:
            return None, {
                "sttv/aux_prompt_len": 0.0,
                "sttv/aux_response_len": 0.0,
                "sttv/aux_rows_dropped_no_images": float(dropped_no_images),
                "sttv/aux_multimodal_fallback_rows": 0.0,
            }

        aux_prompt_len = min(max_prompt_len, max(1, max(len(row["prompt_token_ids"]) for row in aux_rows)))
        aux_response_len = min(max_response_len, max(1, max(len(row["output_token_ids"]) for row in aux_rows)))

        prompts: list[torch.Tensor] = []
        responses: list[torch.Tensor] = []
        response_masks: list[torch.Tensor] = []
        input_ids_all: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        aux_texts: list[str] = []
        aux_images: list[list[Image.Image]] = []

        aux_uids: list[str] = []
        parent_uids: list[str] = []
        call_indices: list[int] = []
        call_records: list[dict[str, Any]] = []
        aux_data_sources: list[str] = []
        aux_ground_truths: list[str] = []
        aux_extra_infos: list[dict[str, Any]] = []

        for row in aux_rows:
            prompt_token_ids = row["prompt_token_ids"]
            output_token_ids = row["output_token_ids"]

            prompt_tensor = torch.full((aux_prompt_len,), pad_token_id, dtype=torch.long)
            prompt_attention = torch.zeros((aux_prompt_len,), dtype=torch.long)
            if prompt_token_ids:
                prompt_tensor[-len(prompt_token_ids) :] = torch.tensor(prompt_token_ids, dtype=torch.long)
                prompt_attention[-len(prompt_token_ids) :] = 1

            response_tensor = torch.full((aux_response_len,), pad_token_id, dtype=torch.long)
            response_attention = torch.zeros((aux_response_len,), dtype=torch.long)
            response_tensor[: len(output_token_ids)] = torch.tensor(output_token_ids, dtype=torch.long)
            response_attention[: len(output_token_ids)] = 1

            input_ids = torch.cat([prompt_tensor, response_tensor], dim=0)
            attention_mask = torch.cat([prompt_attention, response_attention], dim=0)
            response_mask = response_attention.clone()

            prompts.append(prompt_tensor)
            responses.append(response_tensor)
            response_masks.append(response_mask)
            input_ids_all.append(input_ids)
            attention_masks.append(attention_mask)
            aux_texts.append(self.tokenizer.decode(input_ids, skip_special_tokens=True))
            aux_images.append(row["images"])

            uid = row["uid"]
            call_index = int(row["call_index"])
            aux_uids.append(f"{uid}::locv::{call_index}")
            parent_uids.append(uid)
            call_indices.append(call_index)
            call_records.append(row["call_record"])
            aux_data_sources.append(row["data_source"])
            aux_ground_truths.append(row["ground_truth"])
            aux_extra_infos.append(row["extra_info"])

        if self.processor is not None:
            multi_modal_inputs_list, fallback_rows = self._compute_aux_multi_modal_inputs_batched(
                texts=aux_texts,
                images_per_row=aux_images,
            )
        else:
            multi_modal_inputs_list = [{} for _ in aux_rows]
            fallback_rows = 0

        position_ids_all: list[torch.Tensor] = []
        for input_ids, attention_mask, multi_modal_inputs in zip(
            input_ids_all, attention_masks, multi_modal_inputs_list, strict=True
        ):
            position_ids = self._compute_aux_position_ids(
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
            ).squeeze(0)
            position_ids_all.append(position_ids)

        tensors = {
            "prompts": torch.stack(prompts, dim=0),
            "responses": torch.stack(responses, dim=0),
            "response_mask": torch.stack(response_masks, dim=0),
            "input_ids": torch.stack(input_ids_all, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "position_ids": torch.stack(position_ids_all, dim=0),
        }
        non_tensors: dict[str, np.ndarray] = {
            "uid": np.array(aux_uids, dtype=object),
            "sttv_parent_uid": np.array(parent_uids, dtype=object),
            "sttv_loc_call_index": np.array(call_indices, dtype=np.int32),
            "sttv_loc_verifier_call_record": np.array(call_records, dtype=object),
            "data_source": np.array(aux_data_sources, dtype=object),
            "sttv_ground_truth": np.array(aux_ground_truths, dtype=object),
            "sttv_extra_info": np.array(aux_extra_infos, dtype=object),
            "multi_modal_inputs": np.array(multi_modal_inputs_list, dtype=object),
        }
        aux_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=deepcopy(batch.meta_info))
        aux_metrics = {
            "sttv/aux_prompt_len": float(aux_prompt_len),
            "sttv/aux_response_len": float(aux_response_len),
            "sttv/aux_rows_dropped_no_images": float(dropped_no_images),
            "sttv/aux_multimodal_fallback_rows": float(fallback_rows),
        }
        return aux_batch, aux_metrics

    def _compute_sttv_loc_verifier_reward_tensor(
        self,
        aux_batch: Optional[DataProto],
        reward_fn: Optional[Callable[..., Any]],
    ) -> tuple[Optional[torch.Tensor], int]:
        if aux_batch is None or len(aux_batch) == 0:
            return None, 0

        response_mask = aux_batch.batch["response_mask"]
        verifier_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
        total_calls = len(aux_batch)

        reward_values = [0.0] * total_calls
        if reward_fn is not None:
            call_records_raw = aux_batch.non_tensor_batch.get("sttv_loc_verifier_call_record", np.array([], dtype=object))
            call_records = call_records_raw.tolist() if isinstance(call_records_raw, np.ndarray) else list(call_records_raw)
            parent_uids_raw = aux_batch.non_tensor_batch.get(
                "sttv_parent_uid",
                np.array(["unknown"] * len(aux_batch), dtype=object),
            )
            parent_uids = (
                [str(x) for x in parent_uids_raw.tolist()]
                if isinstance(parent_uids_raw, np.ndarray)
                else [str(x) for x in list(parent_uids_raw)]
            )
            if len(parent_uids) < len(aux_batch):
                parent_uids.extend(["unknown"] * (len(aux_batch) - len(parent_uids)))
            data_sources_raw = aux_batch.non_tensor_batch.get("data_source", np.array(["unknown"] * len(aux_batch), dtype=object))
            data_sources = data_sources_raw.tolist() if isinstance(data_sources_raw, np.ndarray) else list(data_sources_raw)
            ground_truths_raw = aux_batch.non_tensor_batch.get(
                "sttv_ground_truth",
                np.array([""] * len(aux_batch), dtype=object),
            )
            ground_truths = ground_truths_raw.tolist() if isinstance(ground_truths_raw, np.ndarray) else list(ground_truths_raw)
            extra_infos_raw = aux_batch.non_tensor_batch.get(
                "sttv_extra_info",
                np.array([{}] * len(aux_batch), dtype=object),
            )
            extra_infos = extra_infos_raw.tolist() if isinstance(extra_infos_raw, np.ndarray) else list(extra_infos_raw)
            solution_strs = [""] * len(call_records)
            try:
                raw_rewards = reward_fn(
                    data_sources=data_sources,
                    loc_verifier_call_records=call_records,
                    solution_strs=solution_strs,
                    ground_truths=ground_truths,
                    extra_infos=extra_infos,
                    parent_uids=parent_uids,
                )
            except TypeError:
                raw_rewards = reward_fn(data_sources, call_records, solution_strs, ground_truths, extra_infos)
            reward_values = self._normalize_flat_rewards(raw_rewards, expected_len=total_calls)

        for row_idx, reward in enumerate(reward_values):
            response_len = int(response_mask[row_idx].sum().item())
            if response_len <= 0:
                continue
            endpoint = response_len - 1
            verifier_rewards[row_idx, endpoint] = float(reward)

        return verifier_rewards, total_calls

    def _compute_sttv_grpo_advantages(
        self,
        token_level_rewards: torch.Tensor,
        objective_mask: torch.Tensor,
        group_index: np.ndarray,
        norm_adv_by_std_in_grpo: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        objective_rewards = token_level_rewards * objective_mask
        return core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=objective_rewards,
            response_mask=objective_mask,
            index=group_index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

    def _compose_sttv_actor_batch(self, main_batch: DataProto, aux_batch: Optional[DataProto]) -> DataProto:
        tensor_keys = [
            "prompts",
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "sttv_adv_answer",
            "sttv_mask_answer",
            "sttv_adv_loc",
            "sttv_mask_loc",
            "sttv_adv_loc_verifier",
            "sttv_mask_loc_verifier",
        ]
        if "ref_log_prob" in main_batch.batch:
            tensor_keys.append("ref_log_prob")

        main_tensors = {key: main_batch.batch[key] for key in tensor_keys if key in main_batch.batch}
        non_tensor_keys = ["uid", "multi_modal_inputs"]
        main_non_tensors = {}
        for key in non_tensor_keys:
            if key in main_batch.non_tensor_batch:
                main_non_tensors[key] = main_batch.non_tensor_batch[key]
            else:
                main_non_tensors[key] = np.array([None] * len(main_batch), dtype=object)
        main_actor_batch = DataProto.from_dict(
            tensors=main_tensors,
            non_tensors=main_non_tensors,
            meta_info=deepcopy(main_batch.meta_info),
        )
        if aux_batch is None or len(aux_batch) == 0:
            return main_actor_batch

        main_prompt_len = int(main_tensors["prompts"].shape[-1]) if "prompts" in main_tensors else 0
        main_response_len = int(main_tensors["responses"].shape[-1]) if "responses" in main_tensors else 0
        aux_prompt_len = int(aux_batch.batch["prompts"].shape[-1]) if "prompts" in aux_batch.batch else 0
        aux_response_len = int(aux_batch.batch["responses"].shape[-1]) if "responses" in aux_batch.batch else 0

        response_tensor_keys = {
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
            "sttv_adv_answer",
            "sttv_mask_answer",
            "sttv_adv_loc",
            "sttv_mask_loc",
            "sttv_adv_loc_verifier",
            "sttv_mask_loc_verifier",
            "ref_log_prob",
        }
        full_seq_tensor_keys = {"input_ids", "attention_mask", "position_ids"}

        def _align_aux_tensor(key: str, aux_tensor: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
            if tuple(aux_tensor.shape[1:]) == tuple(template.shape[1:]):
                return aux_tensor.to(dtype=template.dtype, device=template.device)

            target = torch.zeros(
                (len(aux_batch),) + tuple(template.shape[1:]),
                dtype=template.dtype,
                device=template.device,
            )
            src = aux_tensor.to(dtype=template.dtype, device=template.device)

            if key == "prompts":
                copy_len = min(src.shape[-1], target.shape[-1], aux_prompt_len, main_prompt_len)
                if copy_len > 0:
                    target[..., -copy_len:] = src[..., -copy_len:]
                return target

            if key in response_tensor_keys:
                copy_len = min(src.shape[-1], target.shape[-1], aux_response_len, main_response_len)
                if copy_len > 0:
                    target[..., :copy_len] = src[..., :copy_len]
                return target

            if key in full_seq_tensor_keys:
                prompt_copy_len = min(aux_prompt_len, main_prompt_len)
                if prompt_copy_len > 0:
                    target[..., main_prompt_len - prompt_copy_len : main_prompt_len] = src[
                        ..., aux_prompt_len - prompt_copy_len : aux_prompt_len
                    ]
                response_copy_len = min(aux_response_len, main_response_len)
                if response_copy_len > 0:
                    target[..., main_prompt_len : main_prompt_len + response_copy_len] = src[
                        ..., aux_prompt_len : aux_prompt_len + response_copy_len
                    ]
                return target

            raise RuntimeError(
                f"Unsupported aux/main tensor shape mismatch for key '{key}': "
                f"aux={tuple(aux_tensor.shape[1:])}, main={tuple(template.shape[1:])}"
            )

        aux_tensors: dict[str, torch.Tensor] = {}
        for key, template in main_tensors.items():
            if key in aux_batch.batch:
                aux_tensors[key] = _align_aux_tensor(key, aux_batch.batch[key], template)
            else:
                aux_tensors[key] = torch.zeros(
                    (len(aux_batch),) + tuple(template.shape[1:]),
                    dtype=template.dtype,
                    device=template.device,
                )

        aux_non_tensors: dict[str, np.ndarray] = {}
        for key in non_tensor_keys:
            if key in aux_batch.non_tensor_batch:
                aux_non_tensors[key] = aux_batch.non_tensor_batch[key]
            else:
                aux_non_tensors[key] = np.array([None] * len(aux_batch), dtype=object)

        aux_actor_batch = DataProto.from_dict(
            tensors=aux_tensors,
            non_tensors=aux_non_tensors,
            meta_info=deepcopy(main_batch.meta_info),
        )

        merged = DataProto.concat([main_actor_batch, aux_actor_batch])
        merged.meta_info = deepcopy(main_batch.meta_info)
        return merged

    def _get_actor_dp_size(self) -> int:
        return self._get_dp_size(self.actor_rollout_wg, "actor")

    def _prepare_batch_for_log_prob(self, batch: DataProto) -> DataProto:
        """Keep only fields required by actor/ref compute_log_prob to minimize dispatch payload."""
        tensor_keys = ["responses", "input_ids", "attention_mask", "position_ids", "prompts", "response_mask"]
        selected_tensor_keys = [key for key in tensor_keys if key in batch.batch]
        non_tensor_keys = [key for key in ("uid", "multi_modal_inputs") if key in batch.non_tensor_batch]
        return batch.select(batch_keys=selected_tensor_keys, non_tensor_batch_keys=non_tensor_keys)

    def _compute_old_log_prob_with_padding(self, batch: DataProto) -> tuple[DataProto, float]:
        dp_size = self._get_actor_dp_size()
        log_prob_batch = self._prepare_batch_for_log_prob(batch)
        padded_batch, pad_size = pad_dataproto_to_divisor(log_prob_batch, dp_size)
        old_log_prob, mfu = self._compute_old_log_prob(padded_batch)
        old_log_prob = unpad_dataproto(old_log_prob, pad_size=pad_size)
        return old_log_prob, mfu

    def _can_merge_old_log_prob_batches(self, main_batch: DataProto, aux_batch: Optional[DataProto]) -> bool:
        if aux_batch is None or len(aux_batch) == 0:
            return False
        if "multi_modal_inputs" not in main_batch.non_tensor_batch or "multi_modal_inputs" not in aux_batch.non_tensor_batch:
            return False
        required_keys = ("responses", "input_ids", "attention_mask", "position_ids")
        for key in required_keys:
            if key not in main_batch.batch or key not in aux_batch.batch:
                return False
            main_shape = tuple(main_batch.batch[key].shape[1:])
            aux_shape = tuple(aux_batch.batch[key].shape[1:])
            if main_shape != aux_shape:
                return False
        return True

    def _compute_main_aux_old_log_prob(
        self, main_batch: DataProto, aux_batch: Optional[DataProto]
    ) -> tuple[DataProto, Optional[DataProto], float, bool]:
        if not self._can_merge_old_log_prob_batches(main_batch, aux_batch):
            main_old_log_prob, mfu = self._compute_old_log_prob_with_padding(main_batch)
            aux_old_log_prob = None
            if aux_batch is not None and len(aux_batch) > 0:
                aux_old_log_prob, _ = self._compute_old_log_prob_with_padding(aux_batch)
            return main_old_log_prob, aux_old_log_prob, mfu, False

        merged_log_prob_batch = DataProto.concat(
            [self._prepare_batch_for_log_prob(main_batch), self._prepare_batch_for_log_prob(aux_batch)]
        )
        merged_old_log_prob, mfu = self._compute_old_log_prob_with_padding(merged_log_prob_batch)
        main_rows = len(main_batch)
        main_old_log_prob = merged_old_log_prob[:main_rows]
        aux_old_log_prob = merged_old_log_prob[main_rows:]
        return main_old_log_prob, aux_old_log_prob, mfu, True

    def _compute_ref_log_prob_with_padding(self, batch: DataProto) -> DataProto:
        dp_size = self._get_actor_dp_size()
        log_prob_batch = self._prepare_batch_for_log_prob(batch)
        padded_batch, pad_size = pad_dataproto_to_divisor(log_prob_batch, dp_size)
        ref_log_prob = self._compute_ref_log_prob(padded_batch)
        ref_log_prob = unpad_dataproto(ref_log_prob, pad_size=pad_size)
        return ref_log_prob

    def _pad_actor_batch_for_update(self, actor_batch: DataProto) -> tuple[DataProto, int]:
        dp_size = self._get_actor_dp_size()
        # Keep dispatch and local PPO minibatching aligned:
        # global batch size must be divisible by (dp_size * local_ppo_mini_batch_size).
        local_ppo_mini = int(self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", 1))
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if local_ppo_mini <= 0:
            local_ppo_mini = 1
        global_ppo_divisor = max(1, local_ppo_mini * rollout_n)
        target_divisor = int(np.lcm(dp_size, global_ppo_divisor))

        padded_batch, pad_size = pad_dataproto_to_divisor(actor_batch, target_divisor)
        if pad_size == 0:
            return padded_batch, 0

        # Neutralize padded rows so they do not contribute to policy/entropy/KL losses.
        start = len(actor_batch)
        end = len(padded_batch)
        zero_tensor_keys = [
            "response_mask",
            "advantages",
            "sttv_adv_answer",
            "sttv_mask_answer",
            "sttv_adv_loc",
            "sttv_mask_loc",
            "sttv_adv_loc_verifier",
            "sttv_mask_loc_verifier",
        ]
        for key in zero_tensor_keys:
            if key in padded_batch.batch:
                padded_batch.batch[key][start:end] = 0
        return padded_batch, pad_size

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        reward_for_val: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            reward_for_val: Whether this is for validation
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If reward_for_val=False and sum_reward=True: summed reward_tensor (1D tensor)
            Otherwise: tuple of (reward_tensor, reward_extra_infos_dict)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if not reward_for_val and sum_reward:
                return reward_tensor

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = (
                {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
            )
            return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_raw_prompts = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            raw_prompts = test_batch.non_tensor_batch.get("raw_prompt", None)
            if raw_prompts is not None:
                sample_raw_prompts.extend(raw_prompts.tolist())
            else:
                sample_raw_prompts.extend(input_texts)

            # evaluate using reward_function
            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        self._maybe_log_sample_table(
            split="val",
            raw_prompts=sample_raw_prompts,
            outputs=sample_outputs,
            gts=sample_gts,
            scores=sample_scores,
        )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            rm_resource_pool=rm_resource_pool,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False) or self.config.trainer.get("debug_exit_after_val", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile(global_step=self.global_steps)
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, reward_for_val=False
                            )

                    sttv_multi_objective_enabled = self._is_sttv_multi_objective_enabled()
                    aux_batch: Optional[DataProto] = None
                    aux_old_log_prob: Optional[DataProto] = None
                    if sttv_multi_objective_enabled:
                        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                            raise ValueError("STTV multi-objective currently supports only GRPO adv_estimator.")
                        if self.config.algorithm.use_kl_in_reward:
                            raise ValueError(
                                "STTV multi-objective currently does not support algorithm.use_kl_in_reward=True."
                            )
                        aux_batch, aux_batch_metrics = self._build_sttv_loc_verifier_aux_batch(batch)
                        if aux_batch_metrics:
                            metrics.update(aux_batch_metrics)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob_merged = False
                            if sttv_multi_objective_enabled and aux_batch is not None and len(aux_batch) > 0:
                                old_log_prob, aux_old_log_prob, old_log_prob_mfu, old_log_prob_merged = (
                                    self._compute_main_aux_old_log_prob(batch, aux_batch)
                                )
                                metrics["sttv/old_log_prob_merged"] = float(old_log_prob_merged)
                            else:
                                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if aux_old_log_prob is not None and "entropys" in aux_old_log_prob.batch:
                                aux_old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    actor_train_batch = batch
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        if sttv_multi_objective_enabled:
                            # Answer objective score is scalar outcome score relocated to the answer mask endpoint.
                            answer_mask = self._extract_sttv_mask_tensor(
                                batch,
                                "sttv_answer_mask",
                                fallback_to_response_mask=True,
                            )
                            answer_scores = self._relocate_scalar_scores_to_mask(
                                token_level_scores=reward_tensor,
                                objective_mask=answer_mask,
                                fallback_mask=batch.batch["response_mask"],
                            )
                            batch.batch["token_level_scores"] = answer_scores
                        else:
                            answer_mask = batch.batch["response_mask"]
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Log a sample table from training batches to wandb if enabled.
                        if float(self.config.trainer.get("log_sample_fraction", 0.0)) > 0:
                            outputs = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True
                            )
                            scores = reward_tensor.sum(-1).cpu().tolist()
                            raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
                            if raw_prompts is not None:
                                raw_prompts_list = raw_prompts.tolist()
                            else:
                                raw_prompts_list = ["" for _ in outputs]
                            gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                if hasattr(item, "non_tensor_batch")
                                else None
                                for item in batch
                            ]
                            if not any(gts):
                                gts = [
                                    item.non_tensor_batch.get("answer", None) if hasattr(item, "non_tensor_batch") else None
                                    for item in batch
                                ]
                            self._maybe_log_sample_table(
                                split="train",
                                raw_prompts=raw_prompts_list,
                                outputs=outputs,
                                gts=gts,
                                scores=scores,
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward and not sttv_multi_objective_enabled:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        if sttv_multi_objective_enabled:
                            sttv_reward_fns = self._load_sttv_reward_functions()

                            # Objective 1: answer correctness.
                            answer_advantages, answer_returns = self._compute_sttv_grpo_advantages(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                objective_mask=answer_mask,
                                group_index=batch.non_tensor_batch["uid"],
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                            batch.batch["advantages"] = answer_advantages
                            batch.batch["returns"] = answer_returns
                            batch.batch["sttv_adv_answer"] = answer_advantages
                            batch.batch["sttv_mask_answer"] = answer_mask

                            # Objective 3: loc calls.
                            loc_mask = self._extract_sttv_mask_tensor(batch, "sttv_loc_mask")
                            loc_score_tensor, total_loc_calls = self._compute_sttv_loc_call_reward_tensor(
                                batch,
                                sttv_reward_fns.get("loc"),
                            )
                            loc_advantages, _ = self._compute_sttv_grpo_advantages(
                                token_level_rewards=loc_score_tensor,
                                objective_mask=loc_mask,
                                group_index=batch.non_tensor_batch["uid"],
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                            batch.batch["sttv_adv_loc"] = loc_advantages
                            batch.batch["sttv_mask_loc"] = loc_mask
                            batch.batch["sttv_adv_loc_verifier"] = torch.zeros_like(answer_advantages)
                            batch.batch["sttv_mask_loc_verifier"] = torch.zeros_like(answer_mask)
                            metrics["sttv/loc_call_count"] = float(total_loc_calls)

                            # Objective 2: verifier calls after <loc> (separate aux sub-batch).
                            aux_rows = len(aux_batch) if aux_batch is not None else 0
                            total_loc_verifier_calls = 0
                            if aux_batch is not None and aux_rows > 0:
                                if aux_old_log_prob is None:
                                    aux_old_log_prob, _ = self._compute_old_log_prob_with_padding(aux_batch)
                                if "entropys" in aux_old_log_prob.batch:
                                    aux_old_log_prob.batch.pop("entropys")
                                aux_batch = aux_batch.union(aux_old_log_prob)

                                need_aux_ref_log_prob = self.use_reference_policy and bool(
                                    self.config.actor_rollout_ref.actor.get("use_kl_loss", False)
                                )
                                if need_aux_ref_log_prob:
                                    aux_ref_log_prob = self._compute_ref_log_prob_with_padding(aux_batch)
                                    aux_batch = aux_batch.union(aux_ref_log_prob)

                                aux_verifier_scores, total_loc_verifier_calls = self._compute_sttv_loc_verifier_reward_tensor(
                                    aux_batch,
                                    sttv_reward_fns.get("loc_verifier"),
                                )
                                aux_verifier_mask = aux_batch.batch["response_mask"].to(dtype=answer_mask.dtype)
                                if aux_verifier_scores is None:
                                    aux_verifier_advantages = torch.zeros_like(aux_verifier_mask, dtype=torch.float32)
                                else:
                                    aux_verifier_advantages, _ = self._compute_sttv_grpo_advantages(
                                        token_level_rewards=aux_verifier_scores,
                                        objective_mask=aux_verifier_mask,
                                        group_index=aux_batch.non_tensor_batch["uid"],
                                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    )

                                aux_batch.batch["advantages"] = aux_verifier_advantages
                                aux_batch.batch["sttv_adv_answer"] = torch.zeros_like(aux_verifier_advantages)
                                aux_batch.batch["sttv_mask_answer"] = torch.zeros_like(aux_verifier_mask)
                                aux_batch.batch["sttv_adv_loc"] = torch.zeros_like(aux_verifier_advantages)
                                aux_batch.batch["sttv_mask_loc"] = torch.zeros_like(aux_verifier_mask)
                                aux_batch.batch["sttv_adv_loc_verifier"] = aux_verifier_advantages
                                aux_batch.batch["sttv_mask_loc_verifier"] = aux_verifier_mask

                            metrics["sttv/loc_verifier_aux_rows"] = float(aux_rows)
                            metrics["sttv/loc_verifier_call_count"] = float(total_loc_verifier_calls)

                            actor_train_batch = self._compose_sttv_actor_batch(batch, aux_batch)
                            actor_train_batch.meta_info.update(
                                {
                                    "sttv_multi_objective_enable": True,
                                    "sttv_multi_objective_weights": self._get_sttv_multi_objective_weights(),
                                }
                            )
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                            actor_train_batch = batch

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_train_batch_for_update, actor_pad_size = self._pad_actor_batch_for_update(
                                actor_train_batch
                            )
                            padded_batch_size = max(1, len(actor_train_batch_for_update))
                            pad_ratio = float(actor_pad_size) / float(padded_batch_size)
                            metrics["sttv/actor_update_pad_size"] = float(actor_pad_size)
                            metrics["sttv/actor_update_pad_ratio"] = pad_ratio
                            if pad_ratio > float(
                                self.config.algorithm.get("sttv_multi_objective", {}).get(
                                    "pad_ratio_warn_threshold", STTV_PAD_RATIO_WARN_THRESHOLD
                                )
                            ):
                                print(
                                    "[sttv] Warning: high actor update padding ratio "
                                    f"{pad_ratio:.3f} (pad={actor_pad_size}, batch={padded_batch_size})"
                                )
                            actor_output = self._update_actor(actor_train_batch_for_update)

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        # sleep replicas to avoid OOM during checkpoint saving
                        self.checkpoint_manager.sleep_replicas()
                        self._save_checkpoint()
                        # wake replicas to avoid OOM during checkpoint saving
                        self.checkpoint_manager.update_weights()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
