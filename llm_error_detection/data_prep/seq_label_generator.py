"""
Helper function to substitute specific tokens or group of tokens with more meaningful abstract labels.
"""

import logging
import re
from typing import List

from llm_error_detection.data_prep.math_generator import (
    CORRECT_PAIRS,
    NAMES_WITH_PRONOUNS,
    OBJECTS,
    VERBS,
)

INSTRUCT_TOKENS = [
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "system",
    "user",
    "assistant",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

COMMON_TOKENS = ["Problem", "Reasoning", "Does"]


def strip_list(lst: List[str]) -> List[str]:
    """Strip whitespace from each item in the list.

    Args:
        lst (List[str]): List of strings to strip

    Returns:
        List[str]: List with whitespace stripped from each item
    """
    return [item.strip() for item in lst]


def replace_special_tokens(lst: List[str]) -> List[str]:
    """Replace special tokens with more meaningful labels.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with special tokens replaced by labels
    """
    lst = [item.replace("Ġ", " ").replace("▁", " ") for item in lst]
    lst = [item.replace("ĊĊ", " [newline]").replace("Ċ", " [newline]") for item in lst]
    lst = [item.replace(".", " .") for item in lst]
    lst = [" [space] " if item == " " else item for item in lst]
    return lst


def find_equation_index(lst: list) -> int:
    """Finds the index of the element in the list that contains the '=' sign.

    Args:
        lst (list): List of strings to search.

    Returns:
        int: Index of the element containing the '=' sign.

    Raises:
        ValueError: If no element with '=' is found in the list.
    """
    return next((i for i, item in enumerate(lst) if "=" in item), -1)


def replace_space_after_equation(lst: List[str]) -> List[str]:
    """Replaces the first occurrence of '[space]' after an equation in the string with '[space_after_eq]'.

    Args:
        lst (List[str]): Input string containing an equation and the word '[space]'.

    Returns:
        List[str]: Updated string with '[space_after_eq]' replacing the first '[space]' after the equation.

    Raises:
        ValueError: If no equation or '[space]' is found in the string.
    """
    equation_idx = find_equation_index(lst)

    if lst[equation_idx + 1] == " [space] ":
        lst[equation_idx + 1] = " [space_after_eq] "

    return lst


def replace_dates(lst: List[str]) -> List[str]:
    """Replace date tokens in instruction prompt with a string if the model is 'llama'.

    Args:
        lst (List[str]): List of tokens to process
        model_name (str): Name of the model being used

    Returns:
        List[str]: List with date tokens replaced by a string
    """
    if not lst:
        raise ValueError("The input list is empty.")

    if "system" not in lst:
        logging.info("No system prompt with potential dates present!")
        return lst

    if "user" not in lst:
        raise ValueError("The input does not contain the string 'user'.")

    user_index = lst.index("user")
    integer_regex = re.compile(r"^-?\d+$")

    result = [
        "[date]" if integer_regex.match(item) and idx < user_index else item
        for idx, item in enumerate(lst)
    ]
    return result


def replace_pronouns(lst: List[str]) -> List[str]:
    """Replace pronouns with placeholders.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with pronouns replaced by placeholders
    """
    for pronoun in ["he", "she", "He", "She"]:
        lst = ["[pronoun]" if item == pronoun else item for item in lst]
    return lst


def replace_operation_numbers(lst: List[str], numbers: List[int]) -> List[str]:
    """Replace operand numbers with placeholders, ensuring the last occurrences
    are marked with specific placeholders for equations.

    Args:
        lst (List[str]): List of tokens to process.
        numbers (List[int]): List of numbers to replace.

    Returns:
        List[str]: List with operand numbers replaced by placeholders.

    Raises:
        ValueError: If numbers list has fewer than two elements.
    """
    if len(numbers) < 2:
        raise ValueError("The 'numbers' list must contain at least two numbers.")

    first_digit = str(numbers[0])
    second_digit = str(numbers[1])

    def replace_except_last(
        tokens: List[str], target: str, placeholder: str, last_placeholder: str
    ) -> List[str]:
        count = sum(token == target for token in tokens)
        replaced_count = 0
        result = []

        for token in tokens:
            if token == target:
                replaced_count += 1
                if replaced_count == count:
                    result.append(last_placeholder)
                else:
                    result.append(placeholder)
            else:
                result.append(token)
        return result

    lst = replace_except_last(lst, first_digit, "[op1]", "[op1-in-eq]")
    lst = replace_except_last(lst, second_digit, "[op2]", "[op2-in-eq]")

    return lst


def replace_z1_and_z2_numbers(lst: List[str], numbers: List[int]) -> List[str]:
    """Handle perurbation by replacing result numbers with [mistake] or [correct] labels.

    Args:
        lst (List[str]): List of tokens to process
        numbers (List[int]): List of numbers to process

    Returns:
        List[str]: List with perturbed and correct numbers replaced by appropriate labels
    """
    if len(numbers) < 3:
        return lst

    num_as_str = str(numbers[2])
    first_occurence = True
    if num_as_str in lst:
        for pos in range(len(lst) - 1):
            if lst[pos] == num_as_str:
                if first_occurence:
                    lst[pos] = "[z1]"
                    first_occurence = False
                else:
                    lst[pos] = "[z2]"
    else:
        for pos in range(len(lst) - 1):
            if str(lst[pos]) == num_as_str[0] and str(lst[pos + 1]) == num_as_str[1]:
                if first_occurence:
                    lst[pos] = "[z1-first]"
                    lst[pos + 1] = "[z1-second]"
                    first_occurence = False
                else:
                    lst[pos] = "[z2-first]"
                    lst[pos + 1] = "[z2-second]"

    if len(numbers) == 4:
        num_as_str = str(numbers[3])
        if num_as_str in lst:
            lst = [item.replace(num_as_str, "[z2]") for item in lst]
        else:
            for pos in range(len(lst) - 1):
                if (
                    str(lst[pos]) == num_as_str[0]
                    and str(lst[pos + 1]) == num_as_str[1]
                ):
                    lst[pos] = "[z2-first]"
                    lst[pos + 1] = "[z2-second]"
    return lst


def replace_numbers(lst: List[str], numbers: List[int]) -> List[str]:
    """Replace operands and result numbers.

    Args:
        lst (List[str]): List of tokens to process
        numbers (List[int]): List of numbers to replace

    Returns:
        List[str]: List with numbers replaced by appropriate labels
    """
    lst = replace_z1_and_z2_numbers(lst, numbers)
    lst = replace_operation_numbers(lst, numbers)
    return lst


def replace_pairs(lst: List[str]) -> List[str]:
    """Replace correct pairs (e.g., correct - incorrect) with [valid] and [invalid] labels.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with pairs replaced by [valid] and [invalid] labels
    """
    for pair in CORRECT_PAIRS:
        lst = [
            item.replace(" " + pair[1], "[invalid]").replace(" " + pair[0], "[valid]")
            for item in lst
        ]
    return lst


def replace_operators(lst: List[str]) -> List[str]:
    """Replace + and = signs with them inside square brackets.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with + and = signs replaced by [plus] and [equals]
    """
    lst = [item.replace("+", "[plus]").replace("=", "[equals]") for item in lst]
    return lst


def replace_variables(lst: List[str]) -> List[str]:
    """Replace specific elements (proper names, objects, verbs) with placeholders.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with elements replaced by placeholders
    """
    variables = [names for names, _ in NAMES_WITH_PRONOUNS], OBJECTS, VERBS
    special_tokens = ["name", "object", "verb"]

    for pos, item in enumerate(lst):
        for var, token in zip(variables, special_tokens):
            if item in var:
                lst[pos] = f"[{token}]"
            elif pos + 1 < len(lst) and item + lst[pos + 1] in var:
                lst[pos] = f"[{token}-first]"
                lst[pos + 1] = f"[{token}-second]"
            elif pos + 2 < len(lst) and item + lst[pos + 1] + lst[pos + 2] in var:
                lst[pos] = f"[{token}-first]"
                lst[pos + 1] = f"[{token}-second]"
                lst[pos + 2] = f"[{token}-third]"
    return lst


def replace_all_non_variables(lst: List[str]) -> List[str]:
    """Replace all non-variable elements with token_pos_X.
    X is the position of the token in the list.
    This does not modify tokens that represent the model answer

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with non-variable elements replaced by placeholders
    """
    if "Answer" in lst:
        end_idx = lst.index("Answer")
    else:
        end_idx = len(lst)

    for pos, item in enumerate(lst[:end_idx]):
        if not re.match(r"\[.*\]", item):
            lst[pos] = f"token_pos_{pos}"
    return lst


def replace_punctuation(lst: List[str]) -> List[str]:
    """Replace relevant punctuation with a placeholder.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with punctuation replaced by a placeholder
    """
    punctuation = [".", ",", ":"]
    for pos, item in enumerate(lst):
        if item in punctuation:
            lst[pos] = f"[{item}]"
    return lst


def enumerate_special_tokens(lst: List[str]) -> List[str]:
    """Enumerate special tokens based on their number of occurences.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with special tokens enumerated
    """
    special_tokens = set([item for item in lst if re.match(r"\[.*\]", item)])
    for token in special_tokens:
        token_occ = 1
        for pos, item in enumerate(lst):
            if item == token:
                lst[pos] = f"{token}_occ_{token_occ}"
                token_occ += 1
    return lst


def replace_instruct_tokens(lst: List[str]) -> List[str]:
    """Replace instruction tokens with placeholders.

    Args:
        lst (List[str]): List of tokens to process.

    Returns:
        List[str]: List with instruction tokens replaced by placeholders.
    """
    for pos, item in enumerate(lst):
        if item in INSTRUCT_TOKENS:
            lst[pos] = f"[{item}]"

    return lst


def replace_common_tokens(lst: List[str]) -> List[str]:
    """Replace common tokens with placeholders.

    Args:
        lst (List[str]): List of tokens to process

    Returns:
        List[str]: List with common tokens replaced by placeholders
    """
    for pos, item in enumerate(lst):
        if item in COMMON_TOKENS:
            lst[pos] = f"[{item}]"

    return lst


def gen_seq_labels(
    clean_prompt: List[str],
    model_name: str,
) -> List[str]:
    """
    Generate sequence labels with abstract labels instead of specific tokens, all labels
    are unique.

    Args:
        clean_prompt (List[str]): The input list to be processed.
        model_name (str): The name of the model.

    Returns:
        List[str]: A list of tokens derived from the input string, where each token
        is replaced with a more abstract unique label.
    """
    seq_labels = replace_special_tokens(clean_prompt)
    seq_labels = replace_space_after_equation(seq_labels)

    if "llama-3.2" in model_name.lower():
        seq_labels = replace_dates(seq_labels)
    seq_labels = replace_instruct_tokens(seq_labels)

    space_tokenized_string = "".join(seq_labels).split()
    numbers = [int(token) for token in space_tokenized_string if token.isdigit()]
    numbers = list(dict.fromkeys(numbers))

    seq_labels = replace_numbers(seq_labels, numbers)
    seq_labels = replace_pairs(seq_labels)
    seq_labels = strip_list(seq_labels)
    seq_labels = replace_pronouns(seq_labels)
    seq_labels = replace_variables(seq_labels)
    seq_labels = replace_operators(seq_labels)
    seq_labels = replace_punctuation(seq_labels)
    seq_labels = replace_common_tokens(seq_labels)
    seq_labels = enumerate_special_tokens(seq_labels)
    seq_labels = replace_all_non_variables(seq_labels)

    seq_labels = [item.lower() for item in seq_labels]

    return seq_labels
