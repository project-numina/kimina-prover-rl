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
import re
from collections import Counter

from kimina_prover_rl.reward.definitions import COMMON_STOPWORDS
from kimina_prover_rl.reward.proof_utils import (
    FormatError,
    compute_code_and_comment_length,
    remove_comments,
    remove_trailing_comments,
    verify_tactics_match,
)

logger = logging.getLogger(__name__)


class FormatReward:
    """
    A class to handle the formatting rewards for proofs and thinking blocks.
    """

    def __init__(
        self,
        min_tactic_blocks_per_2k_chars: int = 1,
        min_lines_per_tactic_block: int = 3,
        tactic_block_comment_character_threshold: float = 0.2,
        lean_code_comment_character_threshold: float = 0.2,
        total_tactic_blocks_lines_threshold: float = 0.95,
        comment_lines_ratio_threshold: float = None,
        comment_length_ratio_threshold: float = None,
        tactics_blocks_threshold: int = 2,
        tactics_lean4_match_threshold: float = 0.7,
        repeated_lines_threshold: int = 6,
    ):
        """
        Initialize the FormatReward class with various thresholds for checking the format of the proof.
        Args:
            min_tactic_blocks_per_2k_chars (int): Minimum number of tactic blocks per 2000 characters.
            min_lines_per_tactic_block (int): Minimum number of lines per tactic block.
            tactic_block_comment_character_threshold (float): Threshold for comment characters in tactic blocks.
            lean_code_comment_character_threshold (float): Threshold for comment characters in Lean code.
            total_tactic_blocks_lines_threshold (float): Threshold for total lines in tactic blocks.
            comment_lines_ratio_threshold (float): Ratio threshold for comment lines in tactic blocks.
            comment_length_ratio_threshold (float): Ratio threshold for comment length in tactic blocks.
            tactics_blocks_threshold (int): Minimum number of tactics blocks required.
            tactics_lean4_match_threshold (float): Minimum match score for tactics and Lean 4 code.
            repeated_lines_threshold (int): Minimum number of repeated lines to trigger a format error.
        """
        self.min_tactic_blocks_per_2k_chars = min_tactic_blocks_per_2k_chars
        self.min_lines_per_tactic_block = min_lines_per_tactic_block
        self.tactic_block_comment_character_threshold = tactic_block_comment_character_threshold
        self.lean_code_comment_character_threshold = lean_code_comment_character_threshold
        self.total_tactic_blocks_lines_threshold = total_tactic_blocks_lines_threshold
        self.comment_lines_ratio_threshold = comment_lines_ratio_threshold
        self.comment_length_ratio_threshold = comment_length_ratio_threshold
        self.tactics_blocks_threshold = tactics_blocks_threshold
        self.tactics_lean4_match_threshold = tactics_lean4_match_threshold
        self.repeated_lines_threshold = repeated_lines_threshold

    def check_tactic_code(
        self,
        lean4_code: str,
        tactic_code: str,
    ) -> tuple[bool, str]:
        """
        Check the format of the tactic code.

        Args:
            lean4_code (str): The Lean 4 code to check.
            tactic_code (str): The tactic code to check.
        Returns:
            tuple[FormatError, str]: A tuple containing the format error and the tactic code.
        """
        code_lines, comment_lines, code_length, comment_length = compute_code_and_comment_length(
            tactic_code,
        )

        lean4_code_lines = len(lean4_code.strip().split("\n"))
        max_tactic_lines = max(5, int(lean4_code_lines * self.total_tactic_blocks_lines_threshold))

        if self.total_tactic_blocks_lines_threshold and comment_lines + code_lines > max_tactic_lines:
            return FormatError.TACTIC_CODE_TOO_MANY_LINES, tactic_code

        if (
            self.comment_lines_ratio_threshold
            and comment_lines / (code_lines + 1e-5) > self.comment_lines_ratio_threshold
        ):
            return FormatError.TACTIC_CODE_TOO_MANY_COMMENTS_LINES, tactic_code
        if (
            self.comment_length_ratio_threshold
            and comment_length / (code_length + 1e-5) > self.comment_length_ratio_threshold
        ):
            return FormatError.TACTIC_CODE_TOO_MANY_COMMENTS_CHAR, tactic_code

        return FormatError.NONE, tactic_code

    def _is_significant_thinking_line(self, line: str, repeat_significant_min_tokens: int = 10) -> bool:
        """
        Check if a line of thinking text is significant enough to be included
        in the final output.
        A line is considered significant if it contains enough non-stopword tokens
        or is long enough to convey meaningful information.
        Args:
            line (str): The line of text to check.
            repeat_significant_min_tokens (int): Minimum number of non-stopword tokens
                required for a line to be considered significant.
        Returns:
            bool: True if the line is significant, False otherwise.
        """

        tokens = re.findall(r"\b\w+\b", line.lower())
        content_tokens = [t for t in tokens if t not in COMMON_STOPWORDS]
        return len(content_tokens) >= repeat_significant_min_tokens

    def extract_thinking_text_outside_tactic_blocks(self, think_block: str) -> list[str]:
        """
        Return a list of *significant* natural-language lines inside <think> …,
        stripped of ` ```tactics` fences and trivial whitespace-only lines.
        """
        # 1. Strip every fenced tactics block from the think-section
        clean_think = re.sub(r"```tactics\n.*?\n```", "", think_block, flags=re.DOTALL)

        # 2. Split, trim, lower-case, keep only “significant” lines
        result = []
        for raw in clean_think.splitlines():
            ln = raw.strip()
            if not ln or not re.search(r"\w", ln):  # skip blanks / punctuation rows
                continue
            if self._is_significant_thinking_line(ln):
                result.append(ln.lower())  # normalised
        return result

    def check_thinking_block(
        self,
        text: str,
        num_turns: int = 1,
        previous_formal_code: str | None = None,
    ):
        """
        Check the format of the thinking block in the given text.
        Args:
            text (str): The text to check.
            num_turns (int): The number of turns in the conversation.
            previous_formal_code (str | None): The previous formal code, if any.
        Returns:
            tuple[FormatError, str]: A tuple containing the format error and the text.
        """

        # Find the boundaries of the lean4 code block.
        lean4_match = re.search(r"```lean4\s*([\s\S]+?)\s*```", text)
        if not lean4_match:
            return FormatError.NO_VALID_LEAN4_CODE_BLOCK, text
        lean4_end_index = lean4_match.end()

        # Ensure no extra text exists after the lean4 code block.
        if text[lean4_end_index:].strip():
            return FormatError.NO_VALID_LEAN4_CODE_BLOCK, text

        if "<think>" not in text:
            return FormatError.NO_VALID_THINK_BLOCK, text

        text_after_think = text[text.index("<think>") :]

        # Check for exactly one <think>...</think> block.
        think_blocks = re.findall(r"<think>(.*?)</think>", text_after_think, re.DOTALL)

        if len(think_blocks) != 1:
            return FormatError.INVALID_THINK_BLOCK_COUNT, text

        think_block = think_blocks[0]

        if text_after_think.count("<think>") != 1:
            return FormatError.INVALID_THINK_BLOCK_COUNT, text
        if text_after_think.count("</think>") != 1:
            return FormatError.INVALID_THINK_BLOCK_COUNT, text

        # Check for exactly one Lean 4 code block.
        lean4_blocks = re.findall(r"```lean4\n(.*?)\n```", text_after_think, re.DOTALL)
        if len(lean4_blocks) != 1:
            return FormatError.INVALID_LEAN4_CODE_BLOCK_COUNT, text

        # Check generation repeats in the thinking block
        thinking_lines = self.extract_thinking_text_outside_tactic_blocks(think_block)
        thinking_line_counts = Counter(thinking_lines)
        repeated_lines = [
            line for line, count in thinking_line_counts.items() if count >= self.repeated_lines_threshold
        ]

        if repeated_lines:
            return FormatError.GENERATION_REPEATS, text

        # Check for number of tactics blocks
        tactics_blocks = re.findall(r"```tactics\n(.*?)\n```", text_after_think, re.DOTALL)

        if self.tactics_blocks_threshold and len(tactics_blocks) < self.tactics_blocks_threshold:
            return FormatError.INSUFFICIENT_TACTICS_BLOCKS, text

        for tactic_block in tactics_blocks:
            format_error, _ = self.check_tactic_code(
                lean4_blocks[0],
                tactic_block,
            )
            if format_error != FormatError.NONE:
                return format_error, text

        if num_turns == 1:
            assert previous_formal_code is None, "previous_formal_code should be None for num_turns == 1"
            if self.tactics_lean4_match_threshold is not None:
                format_error, match_score = verify_tactics_match(text, method="iof")
                if format_error != FormatError.NONE:
                    return format_error, text

                if match_score < self.tactics_lean4_match_threshold:
                    return FormatError.TACTICS_LEAN4_NOT_MATCH, text
        else:
            assert previous_formal_code is not None, "previous_formal_code should not be None for num_turns > 1"
            if self.tactics_lean4_match_threshold is not None:
                format_error, match_score = verify_tactics_match(
                    text,
                    method="iof",
                )
                if format_error != FormatError.NONE:
                    return format_error, text

                if match_score < self.tactics_lean4_match_threshold:
                    return FormatError.TACTICS_LEAN4_NOT_MATCH, text

        return FormatError.NONE, text

    def check_lean4_code(
        self,
        lean4_code: str,
        formal_statement: str,
    ) -> tuple[bool, str]:
        """
        Checks if the given Lean 4 code is valid.
        Args:
            lean4_code (str): The Lean 4 code to check.
            formal_statement (str): The formal statement to check against.
        Returns:
            tuple[FormatError, str]: A tuple containing the format error and the Lean 4 code.

        """

        formal_statements = formal_statement.split("\n\n")

        import re

        def clean_and_normalize(s: str) -> str:
            s = remove_comments(s)
            s = s.replace("sorry", "")
            s = re.sub(r"\s+", " ", s)  # normalize whitespace
            return s.strip()

        # Step 1: Clean the entire formal_statement block first
        cleaned_text = remove_comments(formal_statement)

        # Step 2: Then split into blocks (now guaranteed to be comment-free)
        formal_statements = [
            clean_and_normalize(stmt)
            for stmt in cleaned_text.split("\n\n")
            if (clean_stmt := clean_and_normalize(stmt)) and not clean_stmt.startswith(("import", "set_option", "open"))
        ]
        # Clean and normalize lean4 code
        lean4_code_no_comments = clean_and_normalize(lean4_code)

        # Check disallowed keywords
        if "axiom" in lean4_code_no_comments or "local_instance" in lean4_code_no_comments:
            return FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT, lean4_code

        # Final inclusion check
        if not all(fs in lean4_code_no_comments for fs in formal_statements):
            return FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT, lean4_code

        # Check if the code contains the keyword "sorry".
        # retrieve code after the formal statement
        proof_context = lean4_code[len(formal_statement) :]
        code_lines, comment_lines, code_length, comment_length = compute_code_and_comment_length(
            proof_context,
        )

        if (
            self.comment_lines_ratio_threshold
            and comment_lines / (code_lines + 1e-5) > self.comment_lines_ratio_threshold
        ):
            return FormatError.LEAN4_CODE_TOO_MANY_COMMENTS_LINES, lean4_code

        if (
            self.comment_length_ratio_threshold
            and comment_length / (code_length + 1e-5) > self.comment_length_ratio_threshold
        ):
            return FormatError.LEAN4_CODE_TOO_MANY_COMMENTS_CHAR, lean4_code

        # remove the trailing comments
        lean4_code = remove_trailing_comments(lean4_code).strip()

        return FormatError.NONE, lean4_code

    def check_one_turn_format_error(
        self,
        model_output: list | str | None,
        num_turns: int,
        previous_formal_code: str | None,
        formal_statement: str,
    ) -> tuple[bool, str | None, str | None]:
        """
        Extracts a proof from a Lean 4 code block, ensuring it follows the formal statement.

        ground_truths: if provided, Check if the lean4_code in the output contains the solution
        for each answer_tag named in ``abbrev <theorem_name>_solution``, and add
        ```
        example: {answer_tag} = {answer_model_predict} := by
            try rfl
            try norm_num
        ```
        to the end of the lean4_code. Otherwise, the format error is NO_VALID_ANSWER.

        Args:
            model_output (str): The model output containing the Lean 4 code.
            num_turns (int): The number of turns in the conversation.
            previous_formal_code (str | None): The previous formal code, if any.
            formal_statement (str): The formal statement to check against.
        Returns:
            tuple[FormatError, str | None, str | None]: A tuple containing the format error,
                the new output, and the new Lean 4
        """
        format_error: FormatError = FormatError.NONE

        format_error, new_output = self.check_thinking_block(
            model_output,
            num_turns=num_turns,
            previous_formal_code=previous_formal_code,
        )

        if format_error != FormatError.NONE:
            return format_error, None, None

        # Extract all Lean 4 code blocks
        lean4_codes = re.findall(r"```lean4\n(.*?)\n```", new_output, re.DOTALL)
        assert len(lean4_codes) == 1
        lean4_code = lean4_codes[-1].strip()

        # Check if the Lean 4 code is valid
        format_error, new_lean_code = self.check_lean4_code(
            lean4_code,
            formal_statement,
        )
        if format_error != FormatError.NONE:
            return format_error, None, None

        # switch the new lean code into the new output in the lean code block
        assert lean4_code in new_output, "lean4 code not in output"
        new_output = new_output.replace(lean4_code, new_lean_code)

        assert format_error == FormatError.NONE, "format error should be None"

        return format_error, new_output, new_lean_code

    def check_format_error(
        self,
        messages: list | str | None,
        formal_statement: str,
    ) -> tuple[bool, str | None, str | None]:
        """
        Check the format of the messages in the conversation.
        Args:
            messages (list | str | None): The messages to check.
            formal_statement (str): The formal statement to check against.
        Returns:
            tuple[FormatError, list | None, str | None]: A tuple containing the format
                error, the modified messages, and the new Lean 4 code.
        """

        if messages is None or len(messages) == 0:
            return FormatError.UNKNOWN, None, None

        # if it's a single turn message
        if isinstance(messages, str):
            return self.check_one_turn_format_error(
                messages,
                num_turns=1,
                previous_formal_code=None,
                formal_statement=formal_statement,
            )

        if isinstance(messages, list):
            previous_formal_code = None
            num_turns = 1
            for i, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue

                model_output = message["content"]

                format_error, new_output, new_lean_code = self.check_one_turn_format_error(
                    model_output,
                    num_turns=num_turns,
                    previous_formal_code=previous_formal_code,
                    formal_statement=formal_statement,
                )

                if format_error != FormatError.NONE:
                    return format_error, None, None

                previous_formal_code = new_lean_code
                num_turns += 1
                messages[i]["content"] = new_output

            return FormatError.NONE, messages, new_lean_code

        logger.warning(f"Invalid messages: {messages}")
        return FormatError.UNKNOWN, None, None
