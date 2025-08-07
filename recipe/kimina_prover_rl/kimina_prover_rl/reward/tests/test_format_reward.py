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

import pytest
from kimina_prover_rl.reward.format_reward import FormatError, FormatReward


@pytest.fixture
def reward_checker():
    return FormatReward(
        total_tactic_blocks_lines_threshold=0.5,
        comment_lines_ratio_threshold=0.5,
        comment_length_ratio_threshold=0.5,
        tactics_blocks_threshold=1,
        tactics_lean4_match_threshold=0.5,
        min_lines_per_tactic_block=3,
        tactic_block_comment_character_threshold=0.2,
        min_tactic_blocks_per_2k_chars=1,
        lean_code_comment_character_threshold=0.2,
        repeated_lines_threshold=6,  # Example threshold for repeated lines
    )


def test_sorry_dropped(reward_checker):
    statement = "theorem T : A := by sorry"
    output = "theorem T : A := by\n  sorry"
    fmt_err, _, _ = reward_checker.check_one_turn_format_error(output, 1, None, statement)
    assert fmt_err != FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT


def test_import_and_open_ignored(reward_checker):
    statement = "import Mathlib\n\nopen Real\n\ntheorem T : A := by sorry"
    output = "import Mathlib\n\nopen Real\n\ntheorem T : A := by\n  sorry"
    fmt_err, _, _ = reward_checker.check_one_turn_format_error(output, 1, None, statement)
    assert fmt_err != FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT


def test_comment_block_removed(reward_checker):
    statement = "/- comment -/\n\ntheorem T : A := by sorry"
    output = "theorem T : A := by\n  sorry"
    fmt_err, _, _ = reward_checker.check_one_turn_format_error(output, 1, None, statement)
    assert fmt_err != FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT


def test_multi_line_formatting(reward_checker):
    statement = "theorem T :\n  A := by sorry"
    output = "theorem T :\n  A := by\n  sorry"
    fmt_err, _, _ = reward_checker.check_one_turn_format_error(output, 1, None, statement)
    assert fmt_err != FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT


def test_type_annotation(reward_checker):
    statement = "theorem T : IsGreatest (image g (Icc 0 (4/3))) (Real.sqrt 3 / 2) :="
    output = "theorem T : IsGreatest (image g (Icc 0 (4/3 : ℝ))) (Real.sqrt 3 / 2) := by"
    assert (
        reward_checker.check_one_turn_format_error(output, 1, None, statement)
        != FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT
    )


def test_axiom_not_allowed(reward_checker):
    statement = "theorem T : A := by"
    output = "axiom foo : A"
    fmt_err, _, _ = reward_checker.check_one_turn_format_error(output, 1, None, statement)
    print(f"Format error: {fmt_err}")
    assert fmt_err == FormatError.NO_VALID_LEAN4_CODE_BLOCK


def test_missing_theorem(reward_checker):
    statement = "theorem T : A := by"
    output = "def foo := 42"
    fmt_err, _, _ = reward_checker.check_one_turn_format_error(output, 1, None, statement)
    assert fmt_err == FormatError.NO_VALID_LEAN4_CODE_BLOCK


def test_multiple_statements_all_matched(reward_checker):
    statement = """import Mathlib

open Real

theorem T1 : A := by sorry

theorem T2 : B := by sorry"""
    output = """import Mathlib

open Real

theorem T1 : A := by\n  sorry

theorem T2 : B := by\n  sorry"""
    fmt_err, _, _ = reward_checker.check_one_turn_format_error(output, 1, None, statement)
    assert fmt_err != FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT


def test_theorem_preceded_by_lemma(reward_checker):
    # The canonical line that should start the Lean file.
    statement = "theorem number_theory_25102 : A := by sorry"

    # What the model actually produced: imports + lemmas first, then the theorem.
    output = (
        "import Mathlib\n\n"
        "open Finset\n\n"
        "lemma lemma_1 (n : ℤ) (a b c : ℤ) "
        "(h₀ : n = a^3 + b^3 + c^3) : True := by\n"
        "  trivial\n\n"
        "lemma lemma_2 (x y : ℤ) (h : x ^ 3 = y) : True := by\n"
        "  trivial\n\n"
        "theorem number_theory_25102 : A := by\n"
        "  sorry"
    )

    fmt_err, _, _ = reward_checker.check_one_turn_format_error(
        output,  # model output
        1,  # turn number
        None,  # (unused in this test)
        statement,  # expected statement line
    )

    # Because the file does NOT start with `theorem …`, the checker must flag it.
    assert fmt_err != FormatError.LEAN4_CODE_NOT_START_WITH_STATEMENT


def test_thinking_block_without_lean4_returns_error(reward_checker):
    """If the assistant forgets to add *any* Lean 4 code block, the checker
    must flag the output with `NO_VALID_LEAN4_CODE_BLOCK`."""
    text = """
<think>
Some internal reasoning that never culminates in code.
```tactics
simp
```
</think>
"""
    fmt_err, _ = reward_checker.check_thinking_block(text)
    assert fmt_err == FormatError.NO_VALID_LEAN4_CODE_BLOCK


def test_thinking_block_unterminated_lean4_returns_error(reward_checker):
    """A Lean 4 block that is opened but never closed should also raise the
    same error code – the regex fails to find a complete block."""
    text = """
<think>
Thoughts...
</think>
```lean4
#check Nat
"""  # <-- missing closing back‑ticks
    fmt_err, _ = reward_checker.check_thinking_block(text)
    assert fmt_err == FormatError.NO_VALID_LEAN4_CODE_BLOCK


@pytest.mark.parametrize(
    "repeat_count, unique_extra, expected",
    [
        # 10 repeated lines + 40 unique ones  → ratio = 10/50 = 0.20 (< 0.30)
        #  and 10 < abs-limit 50  → should PASS
        (4, 10, FormatError.NONE),
        # ❷ 60 repeated lines, 0 unique         → hits abs-limit 50  → should FAIL
        (6, 10, FormatError.GENERATION_REPEATS),
    ],
)
def test_repeated_line_thresholds(reward_checker, repeat_count, unique_extra, expected):
    # should contain a minimimum of 10 meaningful tokens to be considered significant
    repeated_line = (
        "The goal is `0 = 0`, which is clearly true by definition. "
        "I will try `trivial` to let Lean solve it automatically.\n"
    )

    # Optional extra lines to keep the repeat-ratio under the 0.30 cut-off
    unique_lines = [f"aux {i}\n" for i in range(unique_extra)]

    text = (
        "<think>\n" + repeated_line * repeat_count + "".join(unique_lines) + "\n" * 10 + "```tactics\n"
        "trivial\n"
        "```\n"
        "</think>\n"
        "```lean4\n"
        "theorem T : True := by\n"
        "  trivial\n"
        "```\n"
    )

    fmt_err, _ = reward_checker.check_thinking_block(text)
    assert fmt_err == expected


# -----------------------------------------------------------------------------
# Positive case: a *minimal* valid message should pass the checker
# -----------------------------------------------------------------------------


def test_valid_minimal_thinking_block_passes(reward_checker):
    """Construct the smallest message that satisfies all structural rules **and**
    keeps the tactics block consistent with the Lean 4 code so that the
    tactics‑vs‑Lean match check succeeds.
    """
    text = (
        "<think>\n"
        "Planning the proof…\n\n"
        "```tactics\n"
        "trivial\n"
        "```\n"
        "</think>\n"
        "```lean4\n"
        "theorem T : True := by\n"
        "  trivial\n"
        "```\n"
    )

    fmt_err, _ = reward_checker.check_thinking_block(text)
    assert fmt_err == FormatError.NONE


if __name__ == "__main__":
    # Usage: pytest lean_reward/format_reward.py
    pytest.main([__file__])
