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


def split_proof_header(proof: str) -> tuple[str, str]:
    """
    Split the proof into header and context parts.
    The header contains import statements, while the context contains the rest of the proof.
    Args:
        proof (str): The proof to be split.
    Returns:
        tuple[str, str]: The header and context parts of the proof.
    """
    proof = proof.strip()
    header_lines = []
    context_lines = []
    toggle = False
    proof_lines = proof.split("\n")
    index = 0
    for line in proof_lines:
        if line.startswith("import"):
            if toggle is False:
                toggle = True
            header_lines.append(line)
            index += 1
        else:
            if toggle is True:
                toggle = False
                break
    context_lines = proof_lines[index:]
    return "\n".join(header_lines).strip(), "\n".join(context_lines)


def position_greater_or_equal(pos1: dict, pos2: dict) -> bool:
    """
    Check if position pos1 is greater than or equal to position pos2.
    Args:
        pos1 (dict): The first position with keys "line" and "column".
        pos2 (dict): The second position with keys "line" and "column".
    Returns:
        bool: True if pos1 is greater than or equal to pos2, False otherwise.
    """
    if not isinstance(pos1, dict) or not isinstance(pos2, dict):
        return False

    line1 = pos1["line"]
    col1 = pos1["column"]
    line2 = pos2["line"]
    col2 = pos2["column"]

    if line1 > line2:
        return True
    elif line1 == line2 and col1 >= col2:
        return True
    else:
        return False


def position_distance(pos1: dict, pos2: dict) -> float:
    """
    Calculate the distance between two positions.
    The distance is calculated as 10 times the line difference plus the column difference.
    Args:
        pos1 (dict): The first position with keys "line" and "column".
        pos2 (dict): The second position with keys "line" and "column".
    Returns:
        float: The calculated distance between the two positions.
    """
    if not isinstance(pos1, dict) or not isinstance(pos2, dict):
        return float("inf")

    line1, col1 = pos1["line"], pos1["column"]
    line2, col2 = pos2["line"], pos2["column"]

    return 10 * abs(line1 - line2) + abs(col1 - col2)


def get_error_node(infotree: list, startPos: dict, endPos: dict) -> dict | None:
    """
    Find the closest node in the infotree that contains the error position.
    This function traverses the infotree to find a node whose syntax range
    encompasses the start and end positions of the error.
    Args:
        infotree (list): The infotree containing nodes with syntax ranges.
        startPos (dict): The start position of the error.
        endPos (dict): The end position of the error.
    Returns:
        dict | None: The closest node that contains the error position, or None if no such node is found.
    """
    stack = list(infotree)  # initialize stack with top-level nodes
    tree_candidates = []
    while stack:
        tree = stack.pop()
        try:
            stxRange = tree["node"]["stx"]["range"]
            if not isinstance(stxRange, dict):
                continue
        except (KeyError, TypeError):
            continue

        start_is_in_range = position_greater_or_equal(stxRange.get("start"), startPos)
        end_is_in_range = position_greater_or_equal(endPos, stxRange.get("finish"))

        if start_is_in_range and end_is_in_range:
            distance = position_distance(stxRange.get("start"), startPos)
            distance += position_distance(stxRange.get("finish"), endPos)
            tree_candidates.append(
                {
                    "node": tree,
                    "distance": distance,
                }
            )

        # Add children to stack if they exist
        children = tree.get("children", [])
        if isinstance(children, list):
            stack.extend(children)

    if not tree_candidates:
        return None

    closest_tree = min(
        tree_candidates,
        key=lambda x: x["distance"],
    )
    return closest_tree["node"]


def find_goals_state(message: dict, feedback: dict) -> list | None:
    """
    Find the goals state before or after the error position in the Lean feedback.
    This function checks the error message for position information and retrieves
    the corresponding goals state from the infotree in the feedback.
    Args:
        message (dict): The error message containing position information.
        feedback (dict): The Lean feedback containing the infotree.
    Returns:
        list | None: A list of goals state before or after the error position,
            or None if no goals state is found.
    """
    if "pos" not in message or "endPos" not in message:
        return None
    error_node = get_error_node(feedback.get("infotree", []), message["pos"], message["endPos"])
    if error_node:
        error_node = error_node.get("node", {})
        goals_state = error_node.get("goalsBefore") or error_node.get("goalsAfter")
        return goals_state
    return None


def filter_error_messages(feedback: dict, max_errors: int = 3) -> list:
    """
    Filter error messages from the Lean feedback.
    This function extracts error messages from the feedback response,
    ensuring that only messages with severity "error" are returned.
    Args:
        feedback (dict): The Lean feedback containing error messages.
    Returns:
        list: A list of error messages, limited to the first three.
    """
    if not (feedback.get("response") and feedback["response"].get("messages")):
        return []

    error_messages = [
        msg for msg in feedback["response"]["messages"] if isinstance(msg, dict) and msg.get("severity") == "error"
    ]

    return error_messages[:max_errors]


def format_error_output(error: dict, error_index: int, proof_lines: list, response: dict) -> str:
    """
    Format the error output for a given error message.
    This function extracts the error message, goals state, and code snippet
    from the error and formats them into a structured output.
    Args:
        error (dict): The error message containing position and data.
        error_index (int): The index of the error in the list of errors.
        proof_lines (list): The lines of the proof code.
        response (dict): The Lean response containing the infotree.
    Returns:
        str: A formatted string containing the error message, goals state, and code snippet.
    """
    pos = error.get("pos")
    error_msg = error.get("data", "Unknown error")

    output = f"# Error {error_index}:\n"
    output_dict = {
        "error_message": error_msg.strip(),
        "goals_state": None,
        "code snippet": None,
    }
    if pos:
        line_num = pos.get("line", 1) - 1
        if "line" in pos:
            goals_state = find_goals_state(error, response)
            if goals_state:
                goals_message = "\n".join(goals_state)
                output_dict["goals_state"] = goals_message

        start_line = max(0, line_num - 2)
        end_line = min(len(proof_lines), line_num + 3)

        snippet = "```lean4\n"
        for ln in range(start_line, end_line):
            proof_line = proof_lines[ln].replace("count_heartbeats in", "")
            snippet += f"{ln + 1:03d}{' >' if ln == line_num else '  '}| {proof_line}\n"
        output_dict["code snippet"] = snippet + "```"

    if output_dict["goals_state"]:
        output += f"Goals state before error position: \n{output_dict['goals_state']}\n\n"
    output += f"Error message: \n{output_dict['error_message']}\n\n"
    if output_dict["code snippet"]:
        output += f"Lean 4 code snippet with error:\n{output_dict['code snippet']}\n"
    return output


def create_tool_message(
    formal_code: str,
    lean_feedback: dict,
) -> str:
    """
    Create a tool message from the Lean feedback.
    This function processes the Lean feedback to extract error messages,
    goals state, and code snippets, and formats them into a structured output.
    Args:
        formal_code (str): The formal code to analyze.
        lean_feedback (dict): The Lean feedback containing error messages and other information.
    Returns:
        str: A formatted string containing the error messages, goals state, and code snippets.
    """
    error_messages = filter_error_messages(lean_feedback)

    _, context = split_proof_header(formal_code)
    proof_lines = context.split("\n")
    response = lean_feedback.get("response")

    formatted_output = ""
    # find error with infotree
    for i, error in enumerate(error_messages, 1):
        formatted_output += format_error_output(error, i, proof_lines, response)

    if formatted_output == "":
        top_level_error = lean_feedback.get("error")
        if top_level_error and top_level_error.strip():
            formatted_output += "# System Error:\n"
            formatted_output += f"Error message: \n{top_level_error.strip()}\n\n"
    return formatted_output
