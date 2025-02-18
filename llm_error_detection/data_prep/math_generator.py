"""
Functions to generate data for arithmetic error detection. Clean inputs are mathematical reasoning traces that contain an arithmetic error.
Corrupted prompts are respective reasoning traces without the error present.
"""

# Based on:
# https://github.com/callummcdougall/ARENA_2.0/blob/main/chapter1_transformers/exercises/part3_indirect_object_identification/ioi_dataset.py # noqa: E501

# JSON data format:
# ```
# {
#     // Optional: used to label circuit visualization
#     "seq_labels": [str, ...],

#     // Optional: used by official circuit functions
#     "word_idxs": {
#         str: int,
#         ...
#     },

#     // Required: the prompt pairs
#     "prompts": [
#         {
#             "clean": str | [[int, ...], ...],
#             "corrupt": str | [[int, ...], ...],
#             "answers": [str, ...] | [int, ...],
#             "wrong_answers": [str, ...] | [int, ...],
#         },
#         ...
#     ]
# }
# ```

import logging
import random
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm import tqdm
from transformers import AutoTokenizer

NAMES_WITH_PRONOUNS = [
    ("Aaron", "he"),
    ("Adam", "he"),
    ("Alan", "he"),
    ("Alex", "she"),
    ("Alice", "she"),
    ("Amy", "she"),
    ("Anderson", "he"),
    ("Andre", "he"),
    ("Andrew", "he"),
    ("Andy", "he"),
    ("Anna", "she"),
    ("Anthony", "he"),
    ("Arthur", "he"),
    ("Austin", "he"),
    ("Blake", "she"),
    ("Brandon", "he"),
    ("Brian", "he"),
    ("Carter", "he"),
    ("Charles", "he"),
    ("Charlie", "he"),
    ("Christian", "he"),
    ("Christopher", "he"),
    ("Clark", "he"),
    ("Cole", "he"),
    ("Collins", "he"),
    ("Connor", "he"),
    ("Crew", "he"),
    ("Crystal", "she"),
    ("Daniel", "he"),
    ("David", "he"),
    ("Dean", "he"),
    ("Edward", "he"),
    ("Elizabeth", "she"),
    ("Emily", "she"),
    ("Eric", "he"),
    ("Eva", "she"),
    ("Ford", "he"),
    ("Frank", "he"),
    ("George", "he"),
    ("Georgia", "she"),
    ("Graham", "he"),
    ("Grant", "he"),
    ("Henry", "he"),
    ("Ian", "he"),
    ("Jack", "he"),
    ("Jacob", "he"),
    ("Jake", "he"),
    ("James", "he"),
    ("Jamie", "he"),
    ("Jane", "she"),
    ("Jason", "he"),
    ("Jay", "he"),
    ("Jennifer", "she"),
    ("Jeremy", "he"),
    ("Jessica", "she"),
    ("John", "he"),
    ("Jonathan", "he"),
    ("Jordan", "he"),
    ("Joseph", "he"),
    ("Joshua", "he"),
    ("Justin", "he"),
    ("Kate", "she"),
    ("Kelly", "she"),
    ("Kevin", "he"),
    ("Kyle", "he"),
    ("Laura", "she"),
    ("Leon", "he"),
    ("Lewis", "he"),
    ("Lisa", "she"),
    ("Louis", "he"),
    ("Luke", "he"),
    ("Madison", "she"),
    ("Marco", "he"),
    ("Marcus", "he"),
    ("Maria", "she"),
    ("Mark", "he"),
    ("Martin", "he"),
    ("Mary", "she"),
    ("Matthew", "he"),
    ("Max", "he"),
    ("Michael", "he"),
    ("Michelle", "she"),
    ("Morgan", "he"),
    ("Patrick", "he"),
    ("Paul", "he"),
    ("Peter", "he"),
    ("Prince", "he"),
    ("Rachel", "she"),
    ("Richard", "he"),
    ("River", "she"),
    ("Robert", "he"),
    ("Roman", "he"),
    ("Rose", "she"),
    ("Ruby", "she"),
    ("Russell", "he"),
    ("Ryan", "he"),
    ("Sarah", "she"),
    ("Scott", "he"),
    ("Sean", "he"),
    ("Simon", "he"),
    ("Stephen", "he"),
    ("Steven", "he"),
    ("Sullivan", "he"),
    ("Taylor", "she"),
    ("Thomas", "he"),
    ("Tyler", "he"),
    ("Victoria", "she"),
    ("Warren", "he"),
    ("William", "he"),
]

OBJECTS = [
    "apples",
    "bananas",
    "oranges",
    "grapes",
    "pears",
    "mangoes",
    "strawberries",
    "blueberries",
    "cherries",
    "pineapples",
    "lemons",
    "watermelons",
    "kiwis",
    "peaches",
    "plums",
    "books",
    "pens",
    "notebooks",
    "flowers",
    "candies",
    "gifts",
    "toys",
    "bottles",
    "tickets",
    "clothes",
    "shoes",
    "hats",
    "gloves",
    "keys",
    "wallets",
    "phones",
    "laptops",
    "tablets",
    "cameras",
    "headphones",
    "glasses",
    "watches",
    "rings",
    "necklaces",
    "bracelets",
    "purses",
    "backpacks",
    "umbrellas",
    "mugs",
    "plates",
    "bowls",
    "forks",
    "spoons",
    "knives",
    "chairs",
    "tables",
    "lamps",
    "blankets",
    "pillows",
    "towels",
    "socks",
    "scarves",
    "jackets",
    "belts",
    "bookshelves",
    "paintings",
    "mirrors",
    "candles",
    "frames",
]


VERBS = [
    "won",
    "bought",
    "received",
    "gained",
    "obtained",
    "earned",
    "acquired",
    "collected",
    "accumulated",
    "gathered",
    "got",
]


CORRECT_PAIRS = [
    ("correct", "incorrect"),
    ("right", "wrong"),
    ("valid", "invalid"),
]


TEMPLATES = [
    "Problem: [person] has [num1] [object]. [pronoun] [verb] [num2] more [object]. How many [object] does [pronoun] have now? Reasoning: [person] has [num1] + [num2] = [num3] [object]. So, [pronoun] has [num3] [object] in total. Answer: The above reasoning is ",
    "Problem: [person] starts with [num1] [object]. After [pronoun] [verb] [num2] more, how many [object] does [pronoun] have in total? Reasoning: To solve this, we add [num1] and [num2]: [num1] + [num2] = [num3]. Therefore, [person] now has [num3] [object]. Answer: The above reasoning is ",
    "Problem: Initially, [person] possesses [num1] [object]. [pronoun] then [verb] [num2] additional [object]. What's the new total amount of [object] that [pronoun] has? Reasoning: We calculate: [num1] (original) + [num2] (added) = [num3] (total). So, [person] now has [num3] [object]. Answer: The above reasoning is ",
    "Problem: [person]'s collection of [object] grows from [num1] to an unknown amount after [pronoun] [verb] [num2] more. Reasoning: To find the new total, we add: [num1] + [num2] = [num3] (final amount). Thus, [person] ends up with [num3] [object]. Answer: The above reasoning is ",
    "Problem: [person] originally owns [num1] [object]. After [pronoun] [verb] [num2] additional [object], how many does [pronoun] have altogether? Reasoning: a simple addition gives us [num1] + [num2] = [num3]. Therefore, [person] has [num3] [object] now. Answer: The above reasoning is ",
    "Problem: [person] possesses [num1] [object] at first. If [pronoun] [verb] [num2] more [object], what is the total count? Reasoning: Adding them gives: [num1] + [num2] = [num3]. Consequently, [person] has a total of [num3] [object]. Answer: The above reasoning is ",
    "Problem: [num1] [object] belong to [person]. [pronoun] [verb] [num2] additional ones. What’s the total? Reasoning: By addition, we get [num1] + [num2] = [num3]. Thus, [person] has [num3] [object] in total. Answer: The above reasoning is ",
    "Problem: [person] begins with [num1] [object] and then [verb] [num2] more. How many [object] does [pronoun] have now? Reasoning: Let's add them up: [num1] + [num2] = [num3]. Therefore, [person] has a total of [num3] [object]. Answer: The above reasoning is ",
]


TASK_INSTRUCTION = [
    "Does the following reasoning chain contain any mistakes? Determine whether it is [correct_pair]. ",
    "Does the reasoning chain provided have any errors? Decide whether it is [correct_pair]. ",
    "Does the given reasoning chain contain any flaws? Evaluate whether it is [correct_pair]. ",
    "Does the reasoning chain shown have any errors? Verify whether it is [correct_pair]. ",
    "Does the reasoning chain below have any mistakes? Check if it is [correct_pair]. ",
    "Does the following reasoning chain  have any errors? Specify whether it is [correct_pair]. ",
    "Does the provided reasoning chain contain any flaws? Assess if it is [correct_pair]. ",
    "Does the reasoning chain presented have any issues? Judge whether it is [correct_pair]. ",
    "Does the reasoning chain contain any mistakes? Examine if it is [correct_pair]. ",
    "Does the reasoning chain have any errors? Inspect it and determine if it is [correct_pair]. ",
    "Does the reasoning chain have any flaws? Review it and confirm if it is [correct_pair]. ",
    "Does the given reasoning chain contain any issues? Analyze it and decide if it is [correct_pair]. ",
]


# Generate all pairs of numbers that sum to 10 or more
NUMBER_PAIRS = [
    j for j in list(combinations([i for i in range(2, 10)], 2)) if sum(j) >= 10
]


# Generate perturbation numbers from 10 to 19
PERTURBATION_NUMBERS = list(range(10, 20))


def remove_last_word(text: str) -> str:
    """
    Remove the last word from text based on whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Text without last word.
    """
    return " ".join(text.split()[:-1])


def filter_last_word(text: str) -> str:
    """
     Filter the last word from the text based on whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: The last word of the input text.
    """
    return text.split()[-1]


def filter_last_sentence(text: str) -> str:
    """
    Retain only the last sentence in the text based on punctuation.

    Args:
        text (str): Input text.

    Returns:
        str: The last sentence of the input text.
    """
    return text.split(".")[-1].strip()


def filter_first_sentence(text: str) -> str:
    """
    Retain only the first sentence from the text based on punctuation.

    Args:
        text (str): Input text.

    Returns:
        str: The first sentence of the input text.
    """
    return text.split(".")[0] + "."


def remove_first_and_last_sentence(text: str) -> str:
    """
    Remove the first and last sentence from the text based on punctuation.

    Args:
        text (str): Input text.

    Returns:
        str: The text without first and last sentence.
    """
    return ".".join(text.split(".")[1:-1]).strip() + "."


def split_first_sentence(input_string: str) -> Tuple[str, str]:
    """
    Splits a given string into the first sentence and the remaining text.

    A sentence is determined by the first punctuation mark that indicates
    the end of a sentence (e.g., '.', '!', or '?'). The remaining text excludes
    this punctuation mark and any leading whitespace.

    Args:
        input_string (str): The input string to be split.

    Returns:
        Tuple[str, str]: A tuple where:
            - The first element is the first sentence (including its ending punctuation).
            - The second element is the remaining text (excluding the first sentence and leading whitespace).

    Raises:
        ValueError: If the input string is empty or does not contain any punctuation.
    """
    if not input_string:
        raise ValueError("Input string cannot be empty.")

    # Find the position of the first sentence-ending punctuation
    sentence_end = None
    pos = input_string.find(".")
    if pos != -1 and (sentence_end is None or pos < sentence_end):
        sentence_end = pos

    if sentence_end is None:
        raise ValueError(
            "Input string does not contain any sentence-ending punctuation."
        )

    # Split the string into the first sentence and the rest
    first_sentence = input_string[: sentence_end + 1]  # Include the punctuation
    remaining_text = input_string[
        sentence_end + 1 :
    ].strip()  # Exclude punctuation and trim whitespace

    return first_sentence, remaining_text


def create_chat_formatted_prompt(
    instruction: str, reasoning: str, assistant_answer: Optional[str] = None
) -> list[dict[str, str]]:
    """
    Create a chat-formatted prompt based on the provided instruction, reasoning, and answer.

    Args:
        instruction (str): The instruction for the prompt.
        reasoning (str): The reasoning for the prompt.
        assistant_answer (Optional[str]): The answer for the prompt. Defaults to None.

    Returns:
        list: A list of dictionaries containing the role and content of the chat-formatted prompt.
    """

    chat_format_prompt = [
        {"role": "user", "content": instruction + " " + reasoning},
    ]

    if assistant_answer is not None:
        chat_format_prompt.append({"role": "assistant", "content": assistant_answer})

    return chat_format_prompt


def load_tokenizers(models: List[str], cache_dir: str) -> Dict[str, AutoTokenizer]:
    """
    Load tokenizers for given models from the cache directory.
    """
    tokenizers = {}
    for model in models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                cache_dir=cache_dir,
            )
            tokenizers[model] = tokenizer
        except Exception as e:
            logging.error(f"Error loading tokenizer for {model}: {str(e)}")
    return tokenizers


def get_all_strings_lengths(strings: List[str], tokenizers: AutoTokenizer) -> List[int]:
    """
    Get token lengths for all strings using the specified tokenizer.

    Args:
        strings (List[str]): A list of strings.
        tokenizers (AutoTokenizer): The model's tokenizer.

    Returns:
        List[int]: The length of each string.
    """
    lengths = []
    for text in strings:
        tokens = tokenizers.tokenize(text, add_special_tokens=False)
        lengths.append(len(tokens))
    return lengths


def same_length_strings_ids(
    strings: List[str], tokenizers: Dict[str, AutoTokenizer]
) -> List[int]:
    """
    Find ids of strings that have the same token length for each tokenizer.

    Args:
        strings (List[str]): A list of strings.
        tokenizers (Dict[str, AutoTokenizer]): The model's tokenizer.

    Raises:
        ValueError: No common length has been found.

    Returns:
        List[int]: The ids of strings that have the same length.
    """
    # First, for each tokenizer, get lengths for all strings and filter those with the most common length
    valid_strings = {}
    for name, tokenizer in tokenizers.items():
        lengths = get_all_strings_lengths(strings, tokenizer)
        most_common_length = Counter(lengths).most_common(1)[0][0]
        valid_strings[name] = [
            idx for idx, length in enumerate(lengths) if length == most_common_length
        ]

    # Get intersection of valid indices across all tokenizers
    ids = list(set.intersection(*[set(indices) for indices in valid_strings.values()]))
    if not ids:
        raise ValueError(
            "No strings found with consistent token length for each tokenizer!"
        )

    return ids


def filter_variables(
    tokenizers: Dict[str, AutoTokenizer], template: str
) -> Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]]:
    """
    Filter the lists of names, objects, verbs, and task instructions that have the same
    token length for each tokenizer. If template is "full", all variables are returned.

    Args:
        tokenizers (Dict[str, AutoTokenizer]): Different tokenizers based on the model considered.
        template (str): The task template to consider.

    Returns:
        Dict[str, Union[List[str], List[Tuple[str, str]]]]: A dictionary containing the shared
            names, objects, verbs, and task instructions that have the same token length for each tokenizer.
    """
    if template == "full":
        return {
            "names_with_pronouns": NAMES_WITH_PRONOUNS,
            "objects": OBJECTS,
            "verbs": VERBS,
            "task_instructions": TASK_INSTRUCTION,
        }

    else:
        # Filter names with pronouns
        names_with_pronouns_ids = same_length_strings_ids(
            [name for name, _ in NAMES_WITH_PRONOUNS], tokenizers
        )
        filtered_names_with_pronouns = [
            NAMES_WITH_PRONOUNS[idx] for idx in names_with_pronouns_ids
        ]

        # Filter objects
        objects_ids = same_length_strings_ids(
            [f" {obj}" for obj in OBJECTS], tokenizers
        )
        filtered_objects = [OBJECTS[idx] for idx in objects_ids]

        # Filter verbs
        verbs_ids = same_length_strings_ids([f" {verb}" for verb in VERBS], tokenizers)
        filtered_verbs = [VERBS[idx] for idx in verbs_ids]

        # Filter task instructions
        task_instruction_ids = same_length_strings_ids(
            [inst.replace(" [correct_pair]", "") for inst in TASK_INSTRUCTION],
            tokenizers,
        )
        filtered_task_instructions = [
            TASK_INSTRUCTION[idx] for idx in task_instruction_ids
        ]

        return {
            "names_with_pronouns": filtered_names_with_pronouns,
            "objects": filtered_objects,
            "verbs": filtered_verbs,
            "task_instructions": filtered_task_instructions,
        }


def randomly_sample_variables(
    variables: Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]],
    num_subsample: int,
) -> Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]]:
    """
    Randomly sample a certain number of elements from the given variables dictionaries.

    Args:
        variables (Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]]):
            The dictionary of variables to sample from.
        num_subsample (int):
            The number of elements to sample from each variable.

    Returns:
        Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]]:
            The dictionary of variables with a randomly sampled subset of elements.
    """
    variables["names_with_pronouns"] = (
        random.sample(variables["names_with_pronouns"], num_subsample)
        if num_subsample < len(variables["names_with_pronouns"])
        else variables["names_with_pronouns"]
    )
    variables["objects"] = (
        random.sample(variables["objects"], num_subsample)
        if num_subsample < len(variables["objects"])
        else variables["objects"]
    )
    variables["verbs"] = (
        random.sample(variables["verbs"], num_subsample)
        if num_subsample < len(variables["verbs"])
        else variables["verbs"]
    )

    return variables


def calculate_total_len(
    filtered_variables: Dict[
        str, Union[List[Sequence], List[str], List[Tuple[str, str]]]
    ],
    template: str,
) -> int:
    """
    Log the total number of samples generated.

    Args:
        filtered_variables (Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]]): The filtered variables to use.
        template (str): The template to consider.

    Returns:
        int: The total length of samples generated.
    """
    if template == "full":
        total_len = (
            len(TEMPLATES)
            * len(filtered_variables["task_instructions"])
            * len(CORRECT_PAIRS)
            * len(filtered_variables["names_with_pronouns"])
            * len(filtered_variables["objects"])
            * len(filtered_variables["verbs"])
            * len(NUMBER_PAIRS)
        )
        logging.info(f"Total number of samples: {total_len}")
    else:
        total_len = (
            len(filtered_variables["task_instructions"])
            * len(CORRECT_PAIRS)
            * len(filtered_variables["names_with_pronouns"])
            * len(filtered_variables["objects"])
            * len(filtered_variables["verbs"])
            * len(NUMBER_PAIRS)
        )
        logging.info(
            f"Total number of samples for each template using filtered variables: {total_len}"
        )

    return total_len


def get_chat_components(input_str: str) -> Tuple[str, str, str]:
    """
    Split the input string into three parts: instruction, reasoning string, and answer.

    Args:
        input_str (str): The input string to split.

    Returns:
        Tuple[str, str, str]: A tuple containing the instruction, reasoning string, and answer.
    """
    instruction = filter_first_sentence(input_str)
    reasoning_str = remove_first_and_last_sentence(input_str)
    answer = filter_last_sentence(input_str)

    return instruction, reasoning_str, answer


def get_chat_components_computation(
    input_str: str, two_digit_tokenization: bool = False
) -> Tuple[str, str]:
    """
    Extracts the instruction and reasoning components from an input string containing a computation.

    Args:
        input_str (str): The input string to process.
        two_digit_tokenization (bool): Whether two-digit tokenization is applied. Defaults to False.

    Returns:
        Tuple[str, str]: A tuple with the instruction and the reasoning string extracted from the input.
    """
    instruction, remaining_str = split_first_sentence(input_str)
    reasoning_str, _ = extract_computation_parts(remaining_str, two_digit_tokenization)

    return instruction, reasoning_str


def remove_solution_phrase(input_string: str, full_prompt: bool = True) -> str:
    """
    Removes the undesired text from the input string, leaving only the first sentence
    and the 'Answer: <some-text>' part.

    Args:
        input_string (str): The input string to process.

    Returns:
        str: The cleaned string with undesired text removed.
    """
    if full_prompt:
        cleaned_string = re.sub(r"\. [^\.]*\. Answer:", ". Answer:", input_string)
    else:
        cleaned_string = re.sub(r"[\w\s,]+[.!?]\s*$", "", input_string)
    return cleaned_string


def obtain_number_pair(input_string: str) -> Tuple[int, int]:
    """
    Extracts a pair of numbers from an input string formatted as a simple arithmetic operation.

    Args:
        input_string (str): The string containing the arithmetic operation in the format 'number1 + number2 = ...'.

    Returns:
        Tuple[int, int]: A tuple containing the two numbers extracted from the operation.

    Raises:
        ValueError: If the input string does not contain two numbers in the expected format.
    """

    pattern = r"(\d+)[^\d+\-*/]*(?:[+\-*/])[^\d+\-*/]*(\d+)"
    match = re.search(pattern, input_string)

    if not match:
        raise ValueError(
            "Input string does not contain two numbers in the expected format: 'number1 + number2 = ...'."
        )

    number1, number2 = int(match.group(1)), int(match.group(2))
    return number1, number2


def sample_new_number_pair(
    n1: int, n2: int, number_pairs: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Samples a new pair of numbers from a list, ensuring it differs from the given pair.

    Args:
        n1 (int): The first number of the current pair.
        n2 (int): The second number of the current pair.
        number_pairs (List[Tuple[int, int]]): List of available number pairs to sample from.

    Returns:
        Tuple[int, int]: A new pair of numbers different from (n1, n2).

    Raises:
        ValueError: If there are no valid pairs to sample from.
    """
    valid_pairs = [(x, y) for x, y in number_pairs if x + y != n1 + n2]

    if not valid_pairs:
        raise ValueError(
            "No valid pairs to sample from. Ensure the list has pairs other than the current one."
        )

    return random.choice(valid_pairs)


def gen_alternative_correct_sample(input_string: str) -> str:
    """
    Generates an alternative correct sample by replacing a number pair and their sum in the input string.

    Args:
        input_string (str): The input string containing a number computation.

    Returns:
        str: The input string with the number pair and their sum replaced by a new pair and sum.

    Raises:
        ValueError: If obtaining the number pair or replacing values fails.
    """
    # Step 1: Extract the original number pair (n1, n2) from the input string
    n1, n2 = obtain_number_pair(input_string)

    # Step 2: Generate a new number pair (new_n1, new_n2) different from (n1, n2)
    new_n1, new_n2 = sample_new_number_pair(n1, n2, NUMBER_PAIRS)

    # Step 3: Calculate the new sum
    old_sum = n1 + n2
    new_sum = new_n1 + new_n2

    # Step 4: Replace occurrences of n1, n2, and their sum in the input string using regex
    def safe_replace(pattern: str, replacement: str, text: str) -> str:
        return re.sub(rf"\b{pattern}\b", replacement, text)

    input_string = safe_replace(str(old_sum), str(new_sum), input_string)

    if new_n1 == n2:
        input_string = safe_replace(str(n2), str(new_n2), input_string)
        input_string = safe_replace(str(n1), str(new_n1), input_string)
    else:
        input_string = safe_replace(str(n1), str(new_n1), input_string)
        input_string = safe_replace(str(n2), str(new_n2), input_string)

    return input_string


def extract_computation_parts(
    input_string: str, two_digit_tokenization: bool = False
) -> Tuple[str, str]:
    """
    Extracts text before the result of a computation and the result itself from a formatted string,
    adapting tokenization.

    Args:
        input_string (str): The input string containing the computation and additional text.
        two_digit_tokenization (bool): Whether two-digit tokenization is in place.

    Returns:
        Tuple[str, str]:
            - A tuple where the first element is `processed_text_before` (processed according to model behavior)
              and the second element is `processed_number` (processed accordingly).
            - Returns None if the input string is not in the expected format.

    Raises:
        ValueError: If `processed_number` cannot be isolated or the input string format is invalid.
    """
    try:
        # Define the regex pattern to match the computation and split the string
        pattern = r"^(.*?)(\d+[^\d]*[+\-*/][^\d]*\d+[^\d]*=[^\d]*)(\d+)(.*)$"
        match = re.match(pattern, input_string)

        if not match:
            raise ValueError(
                "Input string is not in the expected format: 'text_before + number2 = number3 <text>'."
            )

        text_before_computation = match.group(1).strip()
        computation_with_equals = match.group(2).strip()
        result_number = match.group(3).strip()

        full_text_before_result = (
            text_before_computation + " " + computation_with_equals
        )

        if not result_number.isdigit():
            raise ValueError(
                f"Extracted result is not a valid number: '{result_number}'."
            )

        # Determine tokenization behavior based on the model name
        if two_digit_tokenization:
            # Two-digit tokenization behavior
            processed_text_before = full_text_before_result + " "
            processed_number3 = result_number
        else:
            # One-digit tokenization behavior
            processed_text_before = full_text_before_result + " " + result_number[0]
            processed_number3 = result_number[1:] if len(result_number) > 1 else ""

        return processed_text_before, processed_number3.strip()

    except Exception as e:
        raise ValueError(e)


def gen_samples(
    variables: Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]],
    current_template: str,
) -> tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Generate a set of samples from a given set of variables and template formulations.

    Args:
        variables (Dict[str, Union[List[Sequence], List[str], List[Tuple[str, str]]]]): The variables to use.
        current_template (str): The current template to use.

    Returns:
        tuple[List[str], List[str], List[str], List[str], List[str]]: A tuple containing the correct samples,
            z1 error samples, z2 error samples, both error, and shortend-z1 error samples.
    """
    correct_list: List[str] = []
    z2_error_list: List[str] = []
    z1_error_list: List[str] = []
    shortened_z1_error_list: List[str] = []
    both_error_list: List[str] = []

    # Filter templates based on the provided template argument
    templates = (
        [TEMPLATES[int(current_template)]] if current_template != "full" else TEMPLATES
    )

    total_len = calculate_total_len(variables, current_template)

    with tqdm(total=total_len) as pbar:
        for temp in templates:
            for inst in variables["task_instructions"]:
                for correct, incorrect in CORRECT_PAIRS:
                    for name, pronoun in variables["names_with_pronouns"]:  # type: ignore
                        for obj in variables["objects"]:
                            for verb in variables["verbs"]:
                                for n1, n2 in NUMBER_PAIRS:
                                    input_text = inst + temp  # type: ignore
                                    input_text = input_text.replace(
                                        "[correct_pair]", f"{correct} or {incorrect}"
                                    )
                                    input_text = input_text.replace(
                                        ". [pronoun]", f". {pronoun.capitalize()}"
                                    )
                                    input_text = input_text.replace(
                                        "[pronoun]", pronoun
                                    )
                                    input_text = input_text.replace("[person]", name)
                                    input_text = input_text.replace("[object]", obj)  # type: ignore
                                    input_text = input_text.replace("[verb]", verb)  # type: ignore
                                    input_text = input_text.replace("[num1]", str(n1))
                                    input_text = input_text.replace("[num2]", str(n2))
                                    input_text = input_text.replace(
                                        "[num3]", str(n1 + n2)
                                    )

                                    # Generate correct sample
                                    correct_list.append(input_text + correct)

                                    # Generate perturbation numbers
                                    possible_perturbations = PERTURBATION_NUMBERS
                                    possible_perturbations = [
                                        i
                                        for i in possible_perturbations
                                        if i > n1 and i > n2
                                    ]
                                    possible_perturbations.remove(n1 + n2)
                                    random_num = random.choice(possible_perturbations)

                                    # Generate z2 error sample
                                    flawed_text = input_text[::-1].replace(
                                        str(n1 + n2)[::-1], str(random_num)[::-1], 1
                                    )[::-1]
                                    z2_error_list.append(flawed_text + incorrect)
                                    assert (
                                        flawed_text != input_text
                                    ), "Error in z2 error generation"

                                    # Generate both error sample
                                    flawed_text = flawed_text[::-1].replace(
                                        str(n1 + n2)[::-1], str(random_num)[::-1], 1
                                    )[::-1]
                                    both_error_list.append(flawed_text + incorrect)
                                    assert (
                                        flawed_text != input_text
                                    ), "Error in both error generation"

                                    # Generate z1 error sample
                                    flawed_text = flawed_text[::-1].replace(
                                        str(random_num)[::-1], str(n1 + n2)[::-1], 1
                                    )[::-1]
                                    z1_error_list.append(flawed_text + incorrect)
                                    assert (
                                        flawed_text != input_text
                                    ), "Error in z1 error generation"

                                    # Generate shortend z1 error sample
                                    shortened_flawed_text = remove_solution_phrase(
                                        flawed_text, full_prompt=True
                                    )
                                    shortened_z1_error_list.append(
                                        shortened_flawed_text + incorrect
                                    )
                                    assert (
                                        len(shortened_flawed_text.split("."))
                                        == len(flawed_text.split(".")) - 1
                                    ), f"Error in shortened-z1 error generation!\n{shortened_flawed_text}"

                                    # Update progress bar
                                    pbar.update(1)
    pbar.close()

    return (
        correct_list,
        z1_error_list,
        z2_error_list,
        both_error_list,
        shortened_z1_error_list,
    )


def subsample_prompts(
    correct_list: List[str],
    z1_error_list: List[str],
    z2_error_list: List[str],
    both_error_list: List[str],
    shortened_z1_error_list: List[str],
    num_samples: int,
) -> tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Subsample the given prompts.

    Args:
        correct_list (List[str]): List of correct prompts.
        z1_error_list (List[str]): List of prompts with z1 errors.
        z2_error_list (List[str]): List of prompts with z2 errors.
        both_error_list (List[str]): List of prompts with both z1 and z2 errors.
        shortened_z1_error_list (List[str]): List of prompts with shortened z1 errors.
        num_samples (int): Number of samples to return.

    Returns:
        tuple[ List[str],  List[str],  List[str],  List[str], List[str]]: A tuple of four lists each containing the subsampled prompts.
    """
    if num_samples < len(correct_list):
        indexes = random.sample(range(len(correct_list)), num_samples)
    else:
        logging.warning(
            f"Less samples generated that required! (samples asked for/given samples): {num_samples}/{len(correct_list)}"
        )
        indexes = list(range(len(correct_list)))

    # Subsample from correct list, z1 error list, etc. using the indexes
    correct_list = [correct_list[i] for i in indexes]
    z1_error_list = [z1_error_list[i] for i in indexes]
    z2_error_list = [z2_error_list[i] for i in indexes]
    both_error_list = [both_error_list[i] for i in indexes]
    shortened_z1_error_list = [shortened_z1_error_list[i] for i in indexes]

    return (
        correct_list,
        z1_error_list,
        z2_error_list,
        both_error_list,
        shortened_z1_error_list,
    )


def get_chat_dict(
    input_str: str,
) -> List[Dict[str, str]]:
    """
    Convert a string into a list of dictionaries, each containing the role and content of a prompt in chat format.

    Args:
        input_str (str): The string to convert.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the role and content of a prompt in chat format.
    """
    input_without_last_word = remove_last_word(input_str)
    instruction, reasoning, assistant_answer = get_chat_components(
        input_without_last_word
    )

    return create_chat_formatted_prompt(
        instruction=instruction,
        reasoning=reasoning,
        assistant_answer=assistant_answer,
    )


def get_shortened_chat_dict(input_str: str) -> List[Dict[str, str]]:
    """
    Convert a string into a list of dictionaries with shortened reasoning in chat format.

    Args:
        input_str (str): The string to convert.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the role and content with shortened reasoning.
    """
    input_without_last_word = remove_last_word(input_str)
    instruction, reasoning, assistant_answer = get_chat_components(
        input_without_last_word
    )
    shortened_reasoning = remove_solution_phrase(reasoning, full_prompt=False)

    return create_chat_formatted_prompt(
        instruction=instruction,
        reasoning=shortened_reasoning,
        assistant_answer=assistant_answer,
    )


def get_computation_chat_dict(
    input_str: str, two_digit_tokenization: bool
) -> List[Dict[str, str]]:
    """
    Convert an input string containing a computation into a chat-formatted prompt.

    Args:
        input_str (str): The input string containing the computation to be processed.
        two_digit_tokenization (bool): Flag indicating whether two-digit tokenization is applied.

    Returns:
        List[Dict[str, str]]: A list of dictionaries formatted for chat, each containing the role and content
        extracted from the instruction and reasoning components of the input string.
    """
    instruction, reasoning = get_chat_components_computation(
        input_str, two_digit_tokenization
    )

    return create_chat_formatted_prompt(
        instruction=instruction,
        reasoning=reasoning,
        assistant_answer=None,
    )


def build_sample(
    corrupt_chat_dict: List[Dict[str, str]],
    clean_chat_dict: List[Dict[str, str]],
    corrupt_answer: str,
    clean_answer: str,
    whitespace: bool = True,
) -> Dict:
    """
    Build a sample dictionary for a computation example.

    Args:
        corrupt_chat_dict (List[Dict[str, str]]): The corrupt chat dictionary.
        clean_chat_dict (List[Dict[str, str]]): The clean chat dictionary.
        corrupt_answer (str): The wrong answer.
        clean_answer (str): The correct answer.
        whitespace (bool): Whether to add a whitespace as prefix to answer. Defaults to True.

    Returns:
        Dict: A sample dictionary containing the corrupt and clean prompts, wrong answers, and correct answers.
    """
    answer_prefix = " " if whitespace else ""

    return {
        "corrupt": corrupt_chat_dict,
        "clean": clean_chat_dict,
        "wrong_answers": [answer_prefix + filter_last_word(corrupt_answer)],
        "answers": [answer_prefix + filter_last_word(clean_answer)],
    }


def samples_to_chat_format(
    correct_list: List[str],
    z1_error_list: List[str],
    z2_error_list: List[str],
    both_error_list: List[str],
    shortened_z1_error_list: List[str],
) -> tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Convert samples to chat formatted prompts.

    Args:
        correct_list (List[str]): The list of correct prompts.
        z1_error_list (List[str]): The list of z1 error prompts.
        z2_error_list (List[str]): The list of z2 error prompts.
        both_error_list (List[str]): The list of both error prompts.
        shortened_z1_error_list (List[str]): List of prompts with shortened z1 errors.

    Returns:
        tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]: A tuple of four lists of dictionaries,
            each containing the chat formatted prompts for z1, z2, both, shortened z1 errors, and computation samples respectively.
    """
    prompt_z1: List[Dict] = []
    prompt_z2: List[Dict] = []
    prompt_both: List[Dict] = []
    prompt_shortened_z1: List[Dict] = []
    prompt_computation_one_digit: List[Dict] = []
    prompt_computation_two_digits: List[Dict] = []

    for correct, error_z1, error_z2, error_both, shortened_z1_error in tqdm(
        zip(
            correct_list,
            z1_error_list,
            z2_error_list,
            both_error_list,
            shortened_z1_error_list,
        ),
        desc="Convert samples",
        total=len(correct_list),
    ):
        # correct
        correct_chat_dict = get_chat_dict(input_str=correct)

        # z1 error
        z1_chat_dict = get_chat_dict(error_z1)
        z1_sample = build_sample(
            corrupt_chat_dict=correct_chat_dict,
            clean_chat_dict=z1_chat_dict,
            corrupt_answer=filter_last_word(correct),
            clean_answer=filter_last_word(error_z1),
        )
        prompt_z1.append(z1_sample)

        # z2 error
        z2_chat_dict = get_chat_dict(error_z2)
        z2_sample = build_sample(
            corrupt_chat_dict=correct_chat_dict,
            clean_chat_dict=z2_chat_dict,
            corrupt_answer=filter_last_word(correct),
            clean_answer=filter_last_word(error_z2),
        )
        prompt_z2.append(z2_sample)

        # both error
        both_chat_dict = get_chat_dict(error_both)
        both_sample = build_sample(
            corrupt_chat_dict=correct_chat_dict,
            clean_chat_dict=both_chat_dict,
            corrupt_answer=filter_last_word(correct),
            clean_answer=filter_last_word(error_both),
        )
        prompt_both.append(both_sample)

        # shortened z1 error
        shortened_correct_chat_dict = get_shortened_chat_dict(input_str=correct)
        shortened_z1_chat_dict = get_chat_dict(shortened_z1_error)
        shortened_z1_sample = build_sample(
            corrupt_chat_dict=shortened_correct_chat_dict,
            clean_chat_dict=shortened_z1_chat_dict,
            corrupt_answer=filter_last_word(correct),
            clean_answer=filter_last_word(shortened_z1_error),
        )
        prompt_shortened_z1.append(shortened_z1_sample)

        # computation samples
        corrupt_computation_str = gen_alternative_correct_sample(correct)
        corrupt_computation_chat_dict = get_computation_chat_dict(
            corrupt_computation_str, two_digit_tokenization=False
        )
        _, corrupt_computation_answer = extract_computation_parts(
            corrupt_computation_str, two_digit_tokenization=False
        )

        clean_computation_chat_dict = get_computation_chat_dict(
            correct, two_digit_tokenization=False
        )
        _, clean_computation_answer = extract_computation_parts(
            correct, two_digit_tokenization=False
        )

        one_digit_computation_sample = build_sample(
            corrupt_chat_dict=corrupt_computation_chat_dict,
            clean_chat_dict=clean_computation_chat_dict,
            corrupt_answer=corrupt_computation_answer,
            clean_answer=clean_computation_answer,
            whitespace=False,
        )
        prompt_computation_one_digit.append(one_digit_computation_sample)

        # two-digit tokenization computation
        corrupt_computation_two_digits_chat_dict = get_computation_chat_dict(
            corrupt_computation_str, two_digit_tokenization=True
        )
        _, corrupt_computation_two_digits_answer = extract_computation_parts(
            corrupt_computation_str, two_digit_tokenization=True
        )

        clean_computation_two_digits_chat_dict = get_computation_chat_dict(
            correct, two_digit_tokenization=True
        )
        _, clean_computation_two_digits_answer = extract_computation_parts(
            correct, two_digit_tokenization=True
        )

        two_digit_computation_sample = build_sample(
            corrupt_chat_dict=corrupt_computation_two_digits_chat_dict,
            clean_chat_dict=clean_computation_two_digits_chat_dict,
            corrupt_answer=corrupt_computation_two_digits_answer,
            clean_answer=clean_computation_two_digits_answer,
            whitespace=False,
        )
        prompt_computation_two_digits.append(two_digit_computation_sample)

    return (
        prompt_z1,
        prompt_z2,
        prompt_both,
        prompt_shortened_z1,
        prompt_computation_one_digit,
        prompt_computation_two_digits,
    )
