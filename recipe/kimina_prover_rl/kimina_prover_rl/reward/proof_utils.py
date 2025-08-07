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

import copy
import re
from enum import Enum

import numpy as np

BEGIN_TOKEN = ["theorem", "lemma", "example"]
DELAYED_TOKEN = ["let", "have"]


class FormatError(Enum):
    UNKNOWN = "Unknown error."
    NONE = "No error."
    GENERATION_OVER_LENGTH_LIMIT = "Generation over length limit."
    GENERATION_REPEATS = "Generation repeats."
    NO_VALID_LEAN4_CODE_BLOCK = "Could not find a valid lean4 code block."
    NO_VALID_THINK_BLOCK = "Could not find a valid think block."
    INVALID_LEAN4_CODE_BLOCK_COUNT = "There must be exactly one lean4 code block."
    INVALID_THINK_BLOCK_COUNT = "There must be exactly one think block."
    INSUFFICIENT_TACTICS_BLOCKS = "There must be more than one tactics block."
    MISSING_CONCLUSION_BLOCK = "Conclusion block is missing between the think block and lean4 code block."
    LEAN4_CODE_NOT_START_WITH_STATEMENT = "Lean 4 code does not start with the formal statement."
    LEAN4_CODE_TOO_MANY_COMMENTS_CHAR = "Lean 4 code contains too many comments (in characters)."
    LEAN4_CODE_TOO_MANY_COMMENTS_LINES = "Lean 4 code contains too many comments (in lines)."
    TACTIC_CODE_TOO_MANY_COMMENTS_CHAR = "Tactic code contains too many comments (in characters)."
    TACTIC_CODE_TOO_MANY_COMMENTS_LINES = "Tactic code contains too many comments (in lines)."
    TACTIC_CODE_TOO_MANY_LINES = "Tactic code contains too many lines."
    TACTIC_BLOCK_FORMATTING_ERROR = "Tactic block formatting error."
    TACTICS_LEAN4_NOT_MATCH = "Tactics and Lean4 code do not match."


class StatsNames(Enum):
    LEAN4_CHARS = "lean4_chars"
    INFORMAL_CHARS = "informal_chars"
    TACTICS_CHARS = "tactics_chars"
    TOTAL_CHARS = "total_chars"
    THINK_CHARS = "think_chars"
    NUM_TACTICS_BLOCKS = "num_tactics_blocks"
    CODE_CHARS = "code_chars"
    COMMENT_CHARS = "comment_chars"
    CODE_LINES = "code_lines"
    COMMENT_LINES = "comment_lines"
    TOTAL_LINES = "total_lines"
    TACTICS_LEAN4_MATCH_SCORE = "tactics_lean4_match_score"


def remove_comments(code: str, remove_empty_lines: bool = False) -> str:
    """
    Remove comments from the Lean 4 code.
    Args:
        code (str): The Lean 4 code to process.
        remove_empty_lines (bool): If True, remove empty lines after removing comments.
    Returns:
        str: The Lean 4 code without comments.
    """
    code = re.sub(r"/-.*?(--/|-/)", "", code, flags=re.DOTALL)
    code = re.sub(r"--.*", "", code)
    if remove_empty_lines:
        lines = code.split("\n")
        code = "\n".join(line for line in lines if line.strip())
        filtered_lines = []
        empty_count = 0
        for line in lines:
            if not line.strip():
                empty_count += 1
                if empty_count < 2:
                    filtered_lines.append(line)
            else:
                empty_count = 0
                filtered_lines.append(line)
        code = "\n".join(filtered_lines)
    return code


def get_statement_split_indices(code: str) -> list[int]:
    """
    This function will return the indexes of the code where the proof statement ends, like after := or := by

    One cases that this function will not work is with match statement, which don't ends with := or := by
    ```lean4
    import Mathlib

    theorem example : "...."
    | casea := sorry
    | caseb := sorry
    ```

    Args:
        code (str): The code that need to be split

    Returns:
        List[int]: The indexes where the code should be split, each for one statements in the proof

    """
    ori_code = copy.copy(code)
    split_indexs = []

    # make sure the := is a separate token
    code = remove_comments(code.replace(":=", " := "))

    # split code into tokens
    tokens = code.split()

    index = 0
    in_statement = False
    in_delayed = False
    for i, token in enumerate(tokens):
        if token in BEGIN_TOKEN:
            in_statement = True
        if token in DELAYED_TOKEN and in_statement:
            in_delayed = True
        if token in [":="]:
            while ori_code[index - len(token) : index] != token:
                index += 1
            if in_statement and not in_delayed:
                if tokens[i + 1] == "by":
                    while ori_code[index - len("by") : index] != "by":
                        index += 1
                split_indexs.append(index)
                in_statement = False
                continue
            if in_statement and in_delayed:
                in_delayed = False
                continue
        while ori_code[index - len(token) : index] != token:
            index += 1
        if index >= len(ori_code):
            raise ValueError(f"Token {token!r} not found")
    return split_indexs


def is_proof_splitable(proof):
    """
    This function will check if the proof is splitable or not

    Args:
        proof (str): The proof that need to be checked

    Returns:
        bool: True if the proof can be split, False otherwise
    """
    return len(get_statement_split_indices(proof)) == 1


def split_proof(proof: str) -> tuple[str, str]:
    """
    This function will split the proof into input and output parts.
    It will split the proof at the last statement, which is the one that ends with := or := by.
    Args:
        proof (str): The proof that need to be split
    Returns:
        tuple[str, str]: The input and output parts of the proof
    """
    split_indexs = get_statement_split_indices(proof)
    split_index = split_indexs[-1]
    proof_input = proof[:split_index]
    proof_output = proof[split_index:]
    # remove leading and trailing whitespaces
    if proof_output.startswith(" \n"):
        proof_input += "\n"
        proof_output = proof_output[2:]
    elif proof_output.startswith("\n"):
        proof_input += "\n"
        proof_output = proof_output[1:]
    return proof_input, proof_output


def parse_proof_from_output(output: str) -> str:
    """
    Parse the proof from the model output for thinking models.
    Takes the last code inside ```lean4 and ``` that has the formal statement inside

    Args:
        output: The model output string.

    Returns:
        The parsed proof string.
    """
    lean4_codes = re.findall(r"```lean4\n(.*?)\n```", output, re.DOTALL)
    words = ["theorem", "by", ":=", "import"]

    for i in range(len(lean4_codes)):
        lean4_code = lean4_codes[-i - 1]
        if all(word in lean4_code for word in words):
            return lean4_code

    return "No proof found in the output."  # this will not pass verification


def extract_last_theorem_name(text: str, key: str) -> str:
    """
    Extract the last theorem or lemma name from the text.
    Args:
        text (str): The input text containing theorems or lemmas.
        key (str): The keyword to search for ("theorem" or "lemma").
    Returns:
        str: The last theorem or lemma name found in the text.
    """
    # Regular expression to match 'theorem' followed by the name and ':=', accounting for multi-line
    if key == "theorem":
        pattern = r"theorem\s+([^\n]+(?:\n[^\n]*)*)\s+:="
    elif key == "lemma":
        pattern = r"lemma\s+([^\n]+(?:\n[^\n]*)*)\s+:="

    # Find all matches (multi-line handling)
    matches = re.findall(pattern, text, flags=re.DOTALL)

    # Return the last matched theorem name if any matches exist
    return matches[-1].strip() if matches else None


def extract_first_theorem_statement(text: str) -> str:
    """
    Extract the first theorem statement from the text.
    Args:
        text (str): The input text containing theorems or lemmas.
    Returns:
        str: The first theorem statement found in the text, or None if not found.
    """
    # Regular expression to match 'theorem' followed by the name and ':=', accounting for multi-line
    pattern = r"theorem\s*(.*?)\s*:="

    # Find all matches (multi-line handling)
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        # Try matching 'lemma' instead
        pattern = r"lemma\s*(.*?)\s*:="
        matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        # Try matching 'example' instead
        pattern = r"example\s*(.*?)\s*:="
        matches = re.findall(pattern, text, flags=re.DOTALL)

    # Return the last matched theorem name if any matches exist
    return matches[0].strip() if matches else None


def compute_code_and_comment_length(proof_code: str) -> tuple[int, int, int, int]:
    """
    Compute the number of lines and characters in the code and comments.
    Args:
        proof_code (str): The Lean 4 code to analyze.
    Returns:
        Tuple[int, int, int, int]: A tuple containing:
            - Number of code lines
            - Number of comment lines
            - Total number of characters in code
            - Total number of characters in comments
    """
    code = remove_comments(proof_code, remove_empty_lines=True)
    comment_lines = len(proof_code.split("\n")) - len(code.split("\n"))
    code_lines = len(code.split("\n"))
    code_length = len(code)
    comment_length = len(proof_code) - code_length
    return code_lines, comment_lines, code_length, comment_length


def contains_trailing_comments(proof_code: str) -> bool:
    """
    Check if the proof code contains trailing comments.
    Args:
        proof_code (str): The Lean 4 code to check.
    Returns:
        bool: True if the code contains trailing comments, False otherwise.
    """
    code = remove_comments(proof_code, remove_empty_lines=True)
    last_code_line = code.strip().split("\n")[-1]
    proof_code_lines = proof_code.strip().split("\n")[-1]
    return last_code_line != proof_code_lines


def remove_trailing_comments(proof_code: str) -> str:
    """
    Remove trailing comments from the proof code.
    Args:
        proof_code (str): The Lean 4 code to modify.
    Returns:
        str: The modified code without trailing comments.
    """
    lines = proof_code.splitlines()

    # Remove any trailing blank lines
    while lines and lines[-1].strip() == "":
        lines.pop()

    # Remove trailing lines that are only comments
    while lines and lines[-1].lstrip().startswith("--"):
        lines.pop()

    return "\n".join(lines)


def parse_tactics_code_blocks(text: str) -> list[str]:
    """
    Extract code blocks that start with ```tactics and end with ``` from a text.
    Args:
        text (str): The input text containing code blocks
    Returns:
        list: A list of extracted code blocks
    Raises:
        ValueError: If the format of any code block is invalid
    """
    result = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        # Look for the start marker
        if lines[i].strip() == "```tactics":
            start_idx = i + 1
            found_end = False

            # Search for the end marker
            for j in range(start_idx, len(lines)):
                if lines[j].strip() == "```":
                    # Extract code between markers
                    code_block = "\n".join(lines[start_idx:j])
                    result.append(code_block)
                    i = j  # Move the index to the end marker
                    found_end = True
                    break

            if not found_end:
                return FormatError.TACTIC_BLOCK_FORMATTING_ERROR, None

        i += 1

    # Additional validation to ensure the format is correct
    for block in result:
        if "```" in block:
            return FormatError.TACTIC_BLOCK_FORMATTING_ERROR, None

    return FormatError.NONE, result


def extract_lean4_block(text: str) -> str:
    """
    Extract the Lean 4 code block from the text.
    Args:
        text (str): The input text containing the Lean 4 code block.
    Returns:
        str: The extracted Lean 4 code block.
    """

    # Check for exactly one Lean 4 code block.
    lean4_blocks = re.findall(r"```lean4\n(.*?)\n```", text, re.DOTALL)
    assert len(lean4_blocks) == 1, FormatError.INVALID_LEAN4_CODE_BLOCK_COUNT
    return lean4_blocks[0].strip()


def check_subcode(subcode: str, final_code: str) -> bool:
    """
    Check if the subcode is a valid subcode of the final code.
    Args:
        subcode (str): The subcode to check.
        final_code (str): The final code to check against.
    Returns:
        bool: True if the subcode is a valid subcode of the final code, False otherwise.
    """
    subcode = remove_comments(subcode, remove_empty_lines=True)
    lines = subcode.split("\n")
    # remove comments
    lines = [x.strip() for x in lines if x.strip() and not x.strip().startswith("--")]
    # check if the 80% of the lines are in final_code
    n_lines = len(lines)
    if n_lines == 0:
        return False
    n_lines_in_final = sum([x in final_code for x in lines])
    return n_lines_in_final / n_lines > 0.6


def intersection_over_union(code_blocks: list[str], final_code: str, use_final_only: bool = False) -> float:
    """
    Compute the intersection over union score between the code blocks and the final code.
    Args:
        code_blocks (list): A list of code blocks to compare.
        final_code (str): The final code to compare against.
        use_final_only (bool): If True, only use the final code for the comparison.
    Returns:
        float: The intersection over union score.
    """
    code_lines = []
    for block in code_blocks:
        block_no_comments = remove_comments(block, remove_empty_lines=True)
        code_lines.extend(
            [x.strip() for x in block_no_comments.split("\n") if x.strip()],
        )

    final_code_no_comments = remove_comments(final_code, remove_empty_lines=True)
    final_lines = [x.strip() for x in final_code_no_comments.split("\n") if x.strip()]

    # turn into sets, compute score
    code_lines = set(code_lines)
    final_lines = set(final_lines)
    if not final_lines:
        return 0
    intersection = code_lines.intersection(final_lines)
    union = code_lines.union(final_lines)
    if use_final_only:
        return float(len(intersection)) / len(final_lines)
    return float(len(intersection)) / len(union)


def verify_tactics_match(text: str, method: str = "iof") -> tuple[FormatError, float]:
    """
    Return the matching score and stop reason.
    Args:
        text (str): The input text to analyze.
        method (str): The method to use for matching. Options are "iof", "iou", or "subcode".
    Returns:
        tuple: A tuple containing the format error and the matching score.
    """
    format_error, code_blocks = parse_tactics_code_blocks(text)

    if format_error != FormatError.NONE:
        return format_error, 0.0

    lean4_block = extract_lean4_block(text)

    # intersection over union, but base on the final code
    if method == "iof":
        return (
            FormatError.NONE,
            intersection_over_union(code_blocks, lean4_block, use_final_only=True),
        )
    if method == "iou":
        return FormatError.NONE, intersection_over_union(code_blocks, lean4_block)
    if method == "subcode":
        check_list = [check_subcode(x, lean4_block) for x in code_blocks]

        # return the average
        return FormatError.NONE, np.mean(check_list)
    raise ValueError(f"Invalid method {method}")


def normalize_formal_statement(formal_statement: str) -> str:
    """
    Normalize the formal statement by removing trailing 'sorry' and ensuring it ends with 'by'.
    Args:
        formal_statement (str): The formal statement to normalize.
    Returns:
        str: The normalized formal statement.
    """

    # Remove trailing "sorry" if present
    formal_statement = formal_statement.rstrip()
    if formal_statement.endswith("sorry"):
        formal_statement = formal_statement[:-5].strip()

    if not formal_statement.endswith("by"):
        formal_statement += " by"

    return formal_statement


def is_index_commented(lean4_code: str, index: int) -> bool:
    """
    Check if the given index in the Lean 4 code is inside a comment.
    Args:
        lean4_code (str): The Lean 4 code to check.
        index (int): The character index to check.
    Returns:
        bool: True if the index is inside a comment, False otherwise.
    """

    if index < 0 or index >= len(lean4_code):
        return False

    # Check if inside a block comment
    block_comments = [(m.start(), m.end()) for m in re.finditer(r"/-.*?-/", lean4_code, re.DOTALL)]
    for start, end in block_comments:
        if start <= index < end:
            return True  # Inside a block comment

    # Check if inside a single-line comment
    lines = lean4_code.splitlines()
    char_count = 0  # Track character index in full string

    for line in lines:
        comment_start = line.find("--")
        if comment_start != -1:
            if char_count <= index < char_count + len(line):  # Index is in this line
                if index >= char_count + comment_start:  # Index is after `--`
                    return True  # Inside a single-line comment
        char_count += len(line) + 1  # +1 for newline character

    return False  # Not inside a comment


def extract_proof_from_text(output: str, formal_statement: str) -> str:
    """
    Extract the proof from the output text.
    Args:
        output (str): The output text containing the proof.
        formal_statement (str): The formal statement to look for in the proof.
    Returns:
        str: The extracted proof if found, otherwise an error message.
    """

    formal_statement = normalize_formal_statement(formal_statement)

    theorem_statement = extract_first_theorem_statement(formal_statement)

    if theorem_statement is None:
        return "Theorem statement couldn't be parsed from statement."

    # Extract all Lean 4 code blocks
    lean4_codes = re.findall(r"```lean4\n(.*?)\n```", output, re.DOTALL)

    for lean4_code in reversed(lean4_codes):
        if theorem_statement in lean4_code:
            # Find the starting position of theorem_statement in the original lean4_code
            theorem_start = lean4_code.find(theorem_statement)
            if theorem_start == -1:
                continue  # This should not happen due to previous check, but added for safety

            # Check if the theorem statement is inside a comment
            if is_index_commented(lean4_code, theorem_start):
                continue

            # Search for `:= by` only after the theorem statement
            match = re.search(r":=\s*by", lean4_code[theorem_start:])
            if match:
                proof_start = theorem_start + match.end()  # Adjust index relative to original string

                return formal_statement + lean4_code[proof_start:]

    return "No proof found in the output."
