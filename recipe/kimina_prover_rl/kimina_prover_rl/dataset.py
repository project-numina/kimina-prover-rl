# Copyright 2025 Project-Numina and/or its affiliates
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

import logging
import random
from typing import Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from kimina_prover_rl.reward.format_reward import FormatError
from verl import DataProto
from verl.utils.dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def init_metrics(n_turns=1) -> dict:
    """
    Initialize custom metrics.
    Returns:
        dict: A dictionary with initialized metrics for each turn.
    """
    metrics = {}
    for turn_id in range(1, n_turns + 1):
        for metric in [
            "valid",
            "filtered",
            "feedback_too_long",
            "max_prompt_token_len_exceeded",
            "exceed_max_turn",
        ]:
            metrics[f"multiturn/turn_{turn_id}/{metric}"] = 0
        metrics[f"multiturn/turn_{turn_id}/added_in_database"] = 0

        for error in FormatError:
            metrics[f"format_error/turn_{turn_id}/{error.value}"] = 0
    return metrics


def get_turn_id(prompt: list[dict]) -> int:
    """
    Get the turn ID based on the number of user messages in the prompt.
    Args:
        prompt (list[dict]): The list of messages in the prompt.
    Returns:
        int: The turn ID.
    """
    return sum(1 for message in prompt if message["role"] == "user")


class NuminaRLDataset(RLHFDataset):
    """
    A dataset class for the Numina RL project, extending the RLHFDataset.
    This class is designed to handle multi-turn interactions and custom metrics.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
        )

        self.multi_turn_data = []
        self.multi_turn_enabled = config.get("multiturn", False)
        self.multi_turn_sampling_rate = config.get("multiturn_sampling_rate", 0.5)
        self.multi_turn_n_samples_in_cache = config.get("multiturn_n_samples_in_cache", 5000)
        self.multiturn_max_feedback_length = config.get("multiturn_max_feedback_length", 3000)
        logger.info(
            f"Multi-turn: {'enabled' if self.multi_turn_enabled else 'disabled'}, "
            f"sampling rate: {self.multi_turn_sampling_rate}, "
            f"n_samples_in_cache: {self.multi_turn_n_samples_in_cache}"
        )
        self.n_turns = 1 if not self.multi_turn_enabled else 2

    def _process_row_dict(self, row_dict: dict) -> dict:
        """
        Process a single row dictionary to prepare it for the model.
        Args:
            row_dict (dict): A dictionary containing the prompt and additional information.
        Returns:
            dict: A processed dictionary with input IDs, attention mask, position IDs, and other relevant information.
        """
        messages = self._build_messages(row_dict)
        model_inputs = {}

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning(
                "tools_kwargs is empty for index {}, data source: {}",
                index,
                row_dict["data_source"],
            )
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict

    def __getitem__(self, item: int) -> dict:
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """

        if (
            self.multi_turn_enabled
            and len(self.multi_turn_data) > 0
            and random.random() < self.multi_turn_sampling_rate
        ):
            # take a sample from the multiturn cache
            row_dict = self.multi_turn_data[item % len(self.multi_turn_data)]
        else:
            row_dict: dict = self.dataframe[item]

        # deep copy + select revelant columns
        row_dict = {
            "prompt": row_dict["prompt"],
            "extra_info": {
                "formal_statement": row_dict["extra_info"]["formal_statement"],
                "index": row_dict["extra_info"]["index"],
                # Adding the prompt here so that is can be used in the multiturn reward
                "prompt": row_dict["prompt"],
            },
            "data_source": row_dict["data_source"],
            "reward_model": {
                "style": row_dict["reward_model"]["style"],
                "ground_truth": row_dict["reward_model"]["ground_truth"],
            },
        }

        return self._process_row_dict(row_dict)

    def compute_format_reward_metrics(self, batch: DataProto, metrics: dict) -> dict:
        """
        Compute the format reward metrics for the given batch.
        Args:
            batch (DataProto): The input batch containing the data.
            metrics (dict): A dictionary to store the computed metrics.
        Returns:
            dict: The updated metrics dictionary with computed format rewards.
        """
        for i, format_error in enumerate(batch.non_tensor_batch["format_error"]):
            turn_id = get_turn_id(batch.non_tensor_batch["raw_prompt"][i])
            format_error = f"format_error/turn_{turn_id}/{format_error}"
            if format_error not in metrics:
                metrics[format_error] = 0
            metrics[format_error] += 1
        return metrics

    def create_one_multiturn_prompt(
        self,
        prompt: list[dict],
        response: str,
        feedback: str,
        extra_info: dict,
        turn_id: int,
        metrics: dict,
    ) -> dict:
        """
        Create a new multi-turn prompt based on the lean feedback and response.
        Args:
            prompt (list[dict]): The current prompt containing messages.
            response (str): The response from the model.
            feedback (str): The feedback provided by lean.
            extra_info (dict): Additional information related to the prompt.
            metrics (dict): A dictionary to update the metrics.
        Returns:
            dict: A dictionary containing the new prompt for the next turn.
        """
        if feedback == "valid proof found.":
            metrics[f"multiturn/turn_{turn_id}/valid"] += 1
            return None

        if feedback == "filtered proof.":
            metrics[f"multiturn/turn_{turn_id}/filtered"] += 1
            return None

        if len(feedback) > self.multiturn_max_feedback_length:
            metrics[f"multiturn/turn_{turn_id}/feedback_too_long"] += 1
            return None

        if turn_id >= self.n_turns:
            metrics[f"multiturn/turn_{turn_id}/exceed_max_turn"] += 1
            return None

        prompt.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        prompt.append({"role": "user", "content": feedback})

        token_length = len(self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))
        if token_length > self.max_prompt_length:
            metrics[f"multiturn/turn_{turn_id}/max_prompt_token_len_exceeded"] += 1
            return None

        # Create a new prompt for the next turn
        return {
            "prompt": prompt,
            "extra_info": {
                "formal_statement": extra_info["formal_statement"],
                "index": extra_info["index"],
                "prompt": prompt,
            },
            "data_source": "multiturn",
            "reward_model": {
                "style": "rule",
                "ground_truth": extra_info["formal_statement"],
            },
        }

    def validation_generate_next_turn(
        self,
        batch: DataProto,
        responses: list[str],
        reward_extra_info,
    ) -> tuple[DataProto, list[int]]:
        """
        Generate the next turn prompts based on the responses and feedback.
        This method is used during validation to create new multi-turn prompts.
        Args:
            batch (DataProto): The input batch containing the data.
            responses (List[str]): The responses from the model.
            reward_extra_info (dict): Additional information related to the reward.
        Returns:
            DataProto: A DataProto object containing the new prompts for the next turn.
            List[int]: The indices of the batch to perform a second turn
        """

        next_turn_prompts = []
        idx_to_update = []

        for i in range(len(batch.non_tensor_batch["raw_prompt"])):
            prompt = batch.non_tensor_batch["raw_prompt"][i]
            prompt = [msg for msg in prompt]
            new_prompt = self.create_one_multiturn_prompt(
                prompt=prompt,
                response=responses[i],
                feedback=reward_extra_info["tool_feedback"][i],
                extra_info=batch.non_tensor_batch["extra_info"][i],
                turn_id=get_turn_id(batch.non_tensor_batch["raw_prompt"][i]),
                metrics=init_metrics(self.n_turns),
            )

            if new_prompt is not None:
                new_prompt = self._process_row_dict(new_prompt)
                next_turn_prompts.append(new_prompt)
                idx_to_update.append(i)

        # convert list(dict) to dict(list) for dataproto
        if len(next_turn_prompts) == 0:
            return None, idx_to_update

        # Convert into a dict of list for the DataProto object
        new_prompts_dict = {}
        for key in next_turn_prompts[0]:
            if isinstance((next_turn_prompts[0][key]), torch.Tensor):
                new_prompts_dict[key] = torch.stack([d[key] for d in next_turn_prompts], dim=0)
            else:
                new_prompts_dict[key] = np.array([d[key] for d in next_turn_prompts], dtype=object)

        return DataProto.from_single_dict(new_prompts_dict), idx_to_update

    def create_multiturn_prompts(self, batch: DataProto, metrics: dict) -> tuple[list[dict], dict]:
        """
        Process the DataProto batch and extract relevant information.
        This method is used to generate new multi-turn prompts based on the feedback received.

        Args:
            batch (DataProto): The input batch to process.

        Returns:
            List[dict]: A list of new prompts for multi-turn interactions.
            dict: A dictionary containing metrics for the processed batch.
        """
        next_turn_prompts = []
        for i, feedback in enumerate(batch.non_tensor_batch["tool_feedback"]):
            prompt = batch.non_tensor_batch["raw_prompt"][i]
            prompt = [prompt[idx] for idx in range(len(prompt))]

            turn_id = get_turn_id(prompt)

            response = batch.non_tensor_batch["response"][i]
            extra_info = batch.non_tensor_batch["extra_info"][i]

            new_multiturn_prompt = self.create_one_multiturn_prompt(
                prompt=prompt,
                response=response,
                feedback=feedback,
                extra_info=extra_info,
                turn_id=turn_id,
                metrics=metrics,
            )

            if new_multiturn_prompt is None:
                continue

            # Create a new prompt for the next turn
            next_turn_prompts.append(new_multiturn_prompt)
            metrics[f"multiturn/turn_{turn_id}/added_in_database"] += 1

        return next_turn_prompts, metrics

    def on_batch_end(self, batch: DataProto) -> None:
        """
        Generate data using the provided data generation strategy.
        Note: This method is intended to change the dataset after each training batch.

        You can't use this method in a multi-process setting. You need to disable multi-processing by setting
        data.dataloader_num_workers=0 in the config file.
        """

        metrics = init_metrics(self.n_turns)
        metrics = self.compute_format_reward_metrics(batch, metrics)
        new_multi_turn_prompts, metrics = self.create_multiturn_prompts(batch, metrics)

        wandb.log(metrics)

        self.multi_turn_data += new_multi_turn_prompts

        # Only keep the last n samples in cache
        self.multi_turn_data = self.multi_turn_data[-self.multi_turn_n_samples_in_cache :]
