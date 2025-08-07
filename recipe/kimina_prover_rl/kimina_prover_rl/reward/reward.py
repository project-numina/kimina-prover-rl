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

from kimina_client import (
    CheckResponse,
    KiminaClient,
    ReplResponse,
    Snippet,
    SnippetStatus,
)

from kimina_prover_rl.reward.error_fixing import create_tool_message
from kimina_prover_rl.reward.format_reward import FormatReward
from kimina_prover_rl.reward.proof_utils import FormatError, extract_proof_from_text

# Kimina client uses environment variables `LEAN_SERVER_API_URL` and `LEAN_SERVER_API_KEY`
# by default. Adjust those in `grpo.sh` or pass in URL and API key here directly.
client = KiminaClient()


def format_reward(
    solution_str: str,
    formal_statement: str,
    prompt: list[dict],
    reward_func: FormatReward,
) -> FormatError:
    """
    Check the thinking and the lean format of the solution.
    Args:
        solution_str (str): The solution string to check.
        formal_statement (str): The formal statement to check against.
        prompt (list[dict]): A list of dictionaries representing the conversation history.
            Each dictionary should have keys like 'role' (e.g., 'user' or 'assistant') and 'content' (the message text).
        reward_funct (FormatReward): The format reward function to use.
    Returns:
        FormatError: The format error
    """

    messages = [message for message in prompt]
    messages.append(
        {
            "role": "assistant",
            "content": solution_str,
        }
    )

    format_error, _, _ = reward_func.check_format_error(messages, formal_statement)
    return format_error


def formal_rewards(lean4_proofs, timeout=60) -> tuple[list[float], list[str]]:
    """
    Compute the formal rewards for the given Lean 4 proofs.
    1 if the proof is valid, 0 otherwise.
    Args:
        lean4_proofs (list[str]): The list of Lean 4 proofs to verify.
        timeout (int): The timeout for the verification process.
    Returns:
        tuple[list[float], list[str]]: A tuple containing the rewards and tool feedbacks used for multiturn
    """

    snippets = [Snippet(id=str(idx), code=proof) for idx, proof in enumerate(lean4_proofs)]

    to_skip = [
        "Theorem statement couldn't be parsed from statement.",
        "No proof found in the output.",
    ]
    snippets_to_check = [s for s in snippets if s.code not in to_skip]

    response: CheckResponse = client.check(
        snips=snippets_to_check,
        timeout=timeout,
        max_workers=40,
    )

    results: list[ReplResponse] = response.results
    id_to_result: dict[str, ReplResponse] = {r.id: r for r in results}

    rewards, tool_feedbacks = [], []

    for s in snippets:
        if (result := id_to_result.get(s.id)) is None:
            rewards.append(0.0)
            tool_feedbacks.append("filtered proof.")
            continue

        analysis = result.analyze()

        if analysis.status == SnippetStatus.valid:
            rewards.append(1.0)
            tool_feedbacks.append("valid proof found.")
            continue

        rewards.append(0.0)
        tool_feedback = create_tool_message(formal_code=s.code, lean_feedback=result.model_dump())
        tool_feedbacks.append(tool_feedback)

    assert len(rewards) == len(snippets), "Unexpected number of rewards: got {}, expected {}".format(
        len(rewards), len(snippets)
    )

    return rewards, tool_feedbacks


def reward(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict],
    return_dict: bool = False,
) -> list[float] | list[dict]:
    """
    Compute the rewards for the output of the model.
    Args:
        data_sources (list[str]): The list of data sources. Unused here.
        solution_strs (list[str]): The list of solution strings.
        ground_truths (list[str]): The list of ground truths. Unused here.
        extra_infos (list[dict]): The list of extra information dictionaries.
        return_dict (bool): If True, return a list of dictionaries with the rewards and tool feedbacks.
    Returns:
        list[float] or list[dict]: A list of rewards if return_dict is False,
            or a list of dictionaries with rewards, predictions, accuracies, tool feedbacks, and responses
            if return_dict is True.

    """
    formal_statements = [extra_info["formal_statement"] for extra_info in extra_infos]
    prompts = [extra_info["prompt"] for extra_info in extra_infos]
    lean4_proofs = [
        extract_proof_from_text(solution_str, formal_statement)
        for solution_str, formal_statement in zip(solution_strs, formal_statements, strict=False)
    ]

    format_func = FormatReward()
    format_errors = [
        format_reward(solution_str, formal_statement, prompt, format_func)
        for solution_str, formal_statement, prompt in zip(solution_strs, formal_statements, prompts, strict=False)
    ]
    format_rws = [format_error == FormatError.NONE for format_error in format_errors]

    proof_rws, tool_feedbacks = formal_rewards(lean4_proofs)

    scores = [proof_rw * format_rw for proof_rw, format_rw in zip(proof_rws, format_rws, strict=False)]

    if not return_dict:
        return scores

    rws = []
    for lean4_proof, score, proof_rw, tool_feedback, solution_str, format_error in zip(
        lean4_proofs, scores, proof_rws, tool_feedbacks, solution_strs, format_errors, strict=False
    ):
        rws.append(
            {
                "score": score,  # This is the final reward
                "pred": lean4_proof,  # This is the Lean 4 proof
                "acc": proof_rw,  # This is the proof reward, used to compute pass@k metrics
                "tool_feedback": tool_feedback,  # This is the feedback from the Lean 4 client
                "response": solution_str,  # This is the original response from the model
                "format_error": format_error.value,  # To log format errors
            }
        )

    return rws
