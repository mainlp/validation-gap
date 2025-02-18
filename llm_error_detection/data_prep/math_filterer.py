"""
Functions to filter math data.
"""

import json
import logging
import re
from collections import Counter
from typing import Dict, List, Tuple, Union

import torch as t
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from llm_error_detection.data_prep.math_generator import obtain_number_pair
from llm_error_detection.data_prep.seq_label_generator import gen_seq_labels
from llm_error_detection.utils import save_dict_to_json


def get_token_lengths(
    text: str, tokenizer: Union[HookedTransformer, AutoTokenizer]
) -> int:
    """
    Get token lengths for a text across all tokenizers.

    Args:
        text (str): The input text.
        tokenizer (Union[HookedTransformer, AutoTokenizer]): The tokenizer to use.

    Returns:
        int: The number of tokens in the text.
    """
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def most_common_string_length_ids(
    strings: List[str], tokenizer: Union[HookedTransformer, AutoTokenizer]
) -> Tuple[List[int], int]:
    """
    Find ids of strings that have the most common length within the list.

    Args:
        strings (List[str]): The list of strings.
        tokenizer (Union[HookedTransformer, AutoTokenizer]): The tokenizer to use.

    Returns:
        Tuple[List[int], int]: The ids of strings that have the most common length
        within the list and the most common length.
    """
    # Get lengths for all strings
    paired_str_and_len = []
    length_counts = []
    for text in strings:
        length = get_token_lengths(text, tokenizer)
        length_counts.append(length)
        paired_str_and_len.append((text, length))

    # Find the most common length
    most_common_length = Counter(length_counts).most_common(1)[0][0]

    # Find ids to keep only strings with the most common length
    ids = [
        idx
        for idx, (text, length) in enumerate(paired_str_and_len)
        if length == most_common_length
    ]

    return ids, most_common_length


def filter_common_length_strings(
    clean_list: List[str],
    corrupt_list: List[str],
    answer_strs: List[List[str]],
    wrong_answer_strs: List[List[str]],
    tokenizer: Union[HookedTransformer, AutoTokenizer],
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """
    Filter strings that have the same token length AND share the most common length within the list.

    Args:
        clean_list (List[str]): The list of clean strings.
        corrupt_list (List[str]): The list of corrupt strings.
        answer_strs (List[str]): The list of answer strings.
        wrong_answer_strs (List[str]): The list of wrong answer strings.
        tokenizer (Union[HookedTransformer, AutoTokenizer]): The tokenizer to use.

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: The filtered lists.
    """
    # Find ids of strings that have the same token length AND share the most common length within the list.
    clean_ids, most_common_clean = most_common_string_length_ids(clean_list, tokenizer)
    corrupt_ids, most_common_corrupt = most_common_string_length_ids(
        corrupt_list, tokenizer
    )

    assert (
        most_common_clean == most_common_corrupt
    ), "Most common length is not the same across clean and corrupt"

    # Find the intersection of all ids
    shared_ids = set(clean_ids) & set(corrupt_ids)

    # Filter the lists based on the shared ids
    filtered_correct_list = [clean_list[idx] for idx in shared_ids]
    filtered_corrupt_list = [corrupt_list[idx] for idx in shared_ids]
    filtered_answer_strs = [answer_strs[idx] for idx in shared_ids]
    filtered_wrong_answer_strs = [wrong_answer_strs[idx] for idx in shared_ids]

    return (
        filtered_correct_list,
        filtered_corrupt_list,
        filtered_answer_strs,
        filtered_wrong_answer_strs,
    )


def remove_trailing_eot_id(text: str, eot_marker: str) -> str:
    """
    Removes the last occurrence of an EOT marker if it appears at the end of the string.

    Args:
        text (str): The input string.
        eot_marker (str): The EOT token.

    Returns:
        str: The modified string with the trailing EOT marker removed if it was present,
             otherwise returns the original string.
    """
    if text.endswith(eot_marker):
        return text[: -len(eot_marker)]
    return text


def remove_trailing_spaces_in_answers(
    answer_strs: List[List[str]], wrong_answer_strs: List[List[str]]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Modify the trailing spaces in answers because Phi predicts tokens with no space.
    """
    for i in range(len(answer_strs)):
        answer_strs[i][0] = answer_strs[i][0].strip()
        wrong_answer_strs[i][0] = wrong_answer_strs[i][0].strip()
    return answer_strs, wrong_answer_strs


def convert_chat_format_to_str(
    model_name: str,
    tokenizer: Union[HookedTransformer, AutoTokenizer],
    chat: List[Dict[str, str]],
) -> str:
    """
    Convert a chat format to a string format.

    Args:
        model_name (str): The name of the model.
        tokenizer (Union[HookedTransformer, AutoTokenizer]): The tokenizer to use.
        chat (List[Dict[str, str]]): The chat in dictionary format.

    Returns:
        str: The chat in string format
    """
    if any(keyword in model_name.lower() for keyword in ["it", "chat", "instruct"]):
        output = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )
        if re.match(r"meta-llama/Llama-3.2-.*-Instruct", model_name):
            output = remove_trailing_eot_id(output, "<|eot_id|>")
        if re.match(r"Qwen/Qwen2.*-Instruct", model_name):
            output = remove_trailing_eot_id(output, "<|im_end|>\n")
        if re.match(r"microsoft/Phi-3.*-instruct", model_name):
            output = remove_trailing_eot_id(output, "<|end|>\n<|endoftext|>")

        # preserve initial white space
        if chat and chat[-1]["content"].endswith(" "):
            output = output.rstrip() + " "
    else:
        output = " ".join([interaction["content"] for interaction in chat])

    return output


def load_math_data(file_path: str) -> dict:
    """
    Loads a mathematical reasoning dataset from a json file.

    Args:
        file_path (str): The path to the json file.

    Returns:
        dict: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find the file {file_path}. Please run scripts/gen_math_data.py first."
        )


def get_prompt_strs(
    math_data: dict,
    model_name: str,
    tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
) -> tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """
    Converts a mathematical reasoning dataset from a dictionary format to a list of strings.

    Args:
        math_data (dict): The dataset to convert.
        model_name (str): The model name to determine the chat format conversion.
        tokenizer (Union[PreTrainedTokenizer, AutoTokenizer]): The tokenizer to use for tokenizing the data.

    Returns:
        tuple[List[str], List[str], List[str], List[str]]: A tuple containing the error prompts, correct prompts, answers, and wrong answers as lists of strings.
    """
    error_prompts_strs = [
        convert_chat_format_to_str(model_name, tokenizer, d["clean"])
        for d in math_data["prompts"]
    ]
    correct_prompts_strs = [
        convert_chat_format_to_str(model_name, tokenizer, d["corrupt"])
        for d in math_data["prompts"]
    ]

    answer_strs = [d["answers"] for d in math_data["prompts"]]
    wrong_answer_strs = [d["wrong_answers"] for d in math_data["prompts"]]

    return error_prompts_strs, correct_prompts_strs, answer_strs, wrong_answer_strs


def prep_math_data(
    error_prompts_strs: List[str],
    correct_prompts_strs: List[str],
    tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
    batch_size: int,
    device: t.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """
    Prepare batches of tokenized mathematical prompts for model input.

    Args:
        math_data (dict): The dataset containing prompts in a dictionary format.
        tokenizer (Union[PreTrainedTokenizer, AutoTokenizer]): The tokenizer used
            to convert prompts to token IDs.
        model_name (str): The name of the model, used to determine the chat format conversion.
        batch_size (int): The size of each batch for processing.
        device (t.device): The device on which the tensors will be loaded.

    Returns:
        tuple[tuple[Tensor, ...], tuple[Tensor, ...]]: Two tuples containing batches
        of tokenized prompts for error and correct prompts respectively.
    """
    error_prompts = tokenizer(error_prompts_strs, padding=True, return_tensors="pt")
    correct_prompts = tokenizer(correct_prompts_strs, padding=True, return_tensors="pt")
    error_prompts = error_prompts["input_ids"].to(device)
    correct_prompts = correct_prompts["input_ids"].to(device)

    error_batches = t.split(error_prompts, batch_size, dim=0)
    correct_batches = t.split(correct_prompts, batch_size, dim=0)

    return error_batches, correct_batches


def tokenize_answers(
    answer_strs: List[List[str]],
    tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
) -> List[Dict]:
    """
    Tokenize answer strings and return a list of dictionaries containing the encoded strings.

    Args:
        answer_strs (List[List[str]]): A list of lists of strings, where each inner list contains the strings
            of a single answer.
        tokenizer (Union[PreTrainedTokenizer, AutoTokenizer]): The tokenizer to use for tokenization.

    Returns:
        List[Dict]: A list of dictionaries containing the encoded answers, with keys "input_ids" and "attention_mask".
    """
    ans_dicts: List[Dict] = []

    for a in answer_strs:
        encoded_str = tokenizer(a, add_special_tokens=False, return_tensors="pt")

        if len(encoded_str["input_ids"][0]) > 1:
            logging.warning(
                f"Encoded input_ids consists of more than 1 token! {encoded_str['input_ids'][0]}"
            )
            # Check if the first token is 29871 and the original string doesn't start with a space
            if encoded_str["input_ids"][0][0].item() == 29871 and not a[0].startswith(
                " "
            ):
                encoded_str["input_ids"] = encoded_str["input_ids"][:, 1:]
                encoded_str["attention_mask"] = encoded_str["attention_mask"][:, 1:]
                logging.warning(
                    f"Issue due to prepended whitespace. Removed now! {encoded_str['input_ids'][0]}"
                )

        ans_dicts.append(encoded_str)

    return ans_dicts


def prep_answers(
    answer_strs: List[List[str]],
    wrong_answer_strs: List[List[str]],
    tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
    model_name: str,
    batch_size: int,
    device: t.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """
    Prepare answer and wrong answer prompts for evaluation.

    Args:
        math_data (dict): The dataset containing prompts in a dictionary format.
        tokenizer (Union[PreTrainedTokenizer, AutoTokenizer]): The tokenizer used
            to convert prompts to token IDs.
        model_name (str): The name of the model, used to determine the chat format conversion.
        batch_size (int): The size of each batch for processing.
        device (t.device): The device on which the tensors will be loaded.

    Returns:
        tuple[tuple[Tensor, ...], tuple[Tensor, ...]]: Two tuples containing batches
        of tokenized prompts for correct and wrong answers respectively.
    """
    # Phi handles differently trailing spaces in the answer
    if "Phi" in model_name:
        answer_strs, wrong_answer_strs = remove_trailing_spaces_in_answers(
            answer_strs, wrong_answer_strs
        )

    ans_dicts = tokenize_answers(
        answer_strs=answer_strs,
        tokenizer=tokenizer,
    )
    wrong_ans_dicts = tokenize_answers(
        answer_strs=wrong_answer_strs,
        tokenizer=tokenizer,
    )

    answers = [a["input_ids"].squeeze(-1).to(device) for a in ans_dicts]
    wrong_answers = [a["input_ids"].squeeze(-1).to(device) for a in wrong_ans_dicts]

    ans = t.split(t.stack(answers), batch_size, dim=0)
    wrong = t.split(t.stack(wrong_answers), batch_size, dim=0)

    return ans, wrong


def logits_to_number_pred(
    logits: Tensor,
    tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
    batch: Tensor,
    tokenization_type: str,
) -> Tuple[Tensor, List[str]]:
    """
    Convert logits to predictions for the result prediction task, where the model
    is expected to predict the result of a mathematical operation.

    Args:
        logits (Tensor): The logits output by the model.
        tokenizer (Union[PreTrainedTokenizer, AutoTokenizer]): The tokenizer used to
            decode the predictions.
        batch (Tensor): The batch of tokenized prompts.
        tokenization_type (str): The type of tokenization used for numbers.

    Returns:
        Tuple[Tensor, List[str]]: A tuple containing a tensor of size (batch_size,) where each element is a boolean indicating
            whether the model has made a correct prediction or not, and the decoded predictions.
    """
    preds = t.argmax(logits, dim=-1)
    preds_toks = [tokenizer.decode(i) for i in preds.tolist()]
    inpt_toks = [tokenizer.decode(i) for i in batch.tolist()]

    target_toks = []

    for inpt in inpt_toks:
        num1, num2 = obtain_number_pair(input_string=inpt)
        sum_result = num1 + num2
        second_digit = (
            str(sum_result)[-1] if tokenization_type == "one_digit" else str(sum_result)
        )
        target_toks.append(second_digit)

    valid_preds: List[bool] = []

    for pred, target in zip(preds_toks, target_toks):
        pred = pred.lower().strip()
        if pred == target:
            valid_preds.append(True)
        elif pred.isdigit():
            valid_preds.append(False)
        else:
            logging.warning(f"Unexpted token predicted: {pred}")
            valid_preds.append(False)

    return t.tensor(valid_preds), preds_toks


def logits_to_valid_pred(
    logits: Tensor,
    tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
    valid_sol: list[str],
    invalid_sol: list[str],
) -> Tuple[Tensor, List[str]]:
    """
    Convert logits to predictions for the error detection task, where the model
    is expected to predict whether the solution is valid or not.

    Args:
        logits (Tensor): The logits output by the model.
        tokenizer (Union[PreTrainedTokenizer, AutoTokenizer]): The tokenizer used to
            decode the predictions.
        valid_sol (list[str]): The valid solutions to compare against.
        invalid_sol (list[str]): The invalid solutions to compare against.
        perturbation (str): The type of perturbation applied.

    Returns:
        Tuple[Tensor, List[str]]: A tuple containing a tensor of size (batch_size,) where each element is a boolean indicating
            whether the model has made a correct prediction or not, and the decoded predictions.
    """
    preds = t.argmax(logits, dim=-1)
    preds_toks = [tokenizer.decode(i) for i in preds.tolist()]

    valid_preds: List[bool] = []

    for pred in preds_toks:
        pred = pred.lower().strip()
        if pred in valid_sol:
            valid_preds.append(True)
        elif pred in invalid_sol:
            valid_preds.append(False)
        else:
            logging.warning(f"Unexpted token predicted: {pred}")
            valid_preds.append(False)

    return t.tensor(valid_preds), preds_toks


def extract_seq_label(
    prompt: str,
    model_name: str,
    tokenizer: Union[HookedTransformer, AutoTokenizer],
) -> str:
    """
    Extract the sequence labels from a prompt.

    Args:
        prompt (str): The prompt to process.
        model_name (str): The name of the model.
        tokenizer (Union[HookedTransformer, AutoTokenizer]): The tokenizer to use for tokenizing the data.

    Returns:
        str: The sequence labels as a single string.
    """
    tokens = tokenizer.tokenize(prompt)
    tokens_seq_labels = gen_seq_labels(
        clean_prompt=tokens,
        model_name=model_name,
    )

    return " ".join(tokens_seq_labels)


def ensure_same_seq_labels(
    prompts: List[str],
    model_name: str,
    tokenizer: Union[HookedTransformer, AutoTokenizer],
) -> List[str]:
    """
    Ensure that all prompts have the same sequence labels.

    Args:
        prompts (List[str]): The list of prompts to process.
        model_name (str): The name of the model.
        tokenizer (Union[HookedTransformer, AutoTokenizer]): The tokenizer to use for tokenizing the data.

    Returns:
        List[str]: List of sequence labels.
    """
    seq_labels = extract_seq_label(prompts[0], model_name, tokenizer)

    for prompt in prompts[1:]:
        curr_seq_labels = extract_seq_label(prompt, model_name, tokenizer)
        assert seq_labels == curr_seq_labels, "sequence labels mismatch!"

    return seq_labels.split(" ")


def filter_math_data(
    data_dir: str,
    model_name: str,
    model: Union[HookedTransformer, AutoModelForCausalLM],
    tokenizer: Union[HookedTransformer, AutoTokenizer],
    perturbation: str,
    template: str,
    device: t.device,
    batch_size: int = 32,
    check_answers: bool = True,
) -> None:
    """
    Load a dataset from a json file and filter the data to only
    include the examples of the task that a model can classify correctly.

    JSON data format:
    ```
    {
        "prompts": [
            {
                "clean": str | [[int, ...], ...],
                "corrupt": str | [[int, ...], ...],
                "answers": [str, ...] | [int, ...],
                "wrong_answers": [str, ...] | [int, ...],
            },
            ...
        ]
    }
    ```

    Args:
        model_name: The name of the model to use for saving the filtered data.
        model: The model to use for inference.
        tokenizer: The tokenizer to use for tokenizing the data.
        perturbation: The type of perturbation dataset we want to load.
        template: The type of template to use for the data.
        device: The device to use for inference.
        batch_size: The batch size to use when doing inference on the data to filter it.
        check_answers: Whether to check the answers or not.

    Returns:
        None
    """
    in_data_specifier = perturbation

    if perturbation == "computation":
        if "llama" in model_name.lower():
            in_data_specifier += "_two_digits"
        else:
            in_data_specifier += "_one_digit"

    # get data
    file_path = f"{data_dir}/{template}/math_prompts_{in_data_specifier}.json"
    math_data = load_math_data(file_path)

    clean_prompts_strs, corrupt_prompts_strs, answer_strs, wrong_answer_strs = (
        get_prompt_strs(math_data=math_data, model_name=model_name, tokenizer=tokenizer)
    )

    clean_batches, corrupt_batches = prep_math_data(
        error_prompts_strs=clean_prompts_strs,
        correct_prompts_strs=corrupt_prompts_strs,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    ans, wrong = prep_answers(
        answer_strs=answer_strs,
        wrong_answer_strs=wrong_answer_strs,
        tokenizer=tokenizer,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )

    correct_list: List[Tensor] = []

    if check_answers:
        # Filter the data
        with t.inference_mode():
            for clean_btch, clean_gt, corr_batch, corr_gt in tqdm(
                zip(clean_batches, ans, corrupt_batches, wrong),
                total=len(ans),
                desc=f"Filtering Data for {perturbation}",
            ):
                logits = model(clean_btch)[:, -1, :]
                preds = t.argmax(logits, dim=-1)
                correct_clean = t.eq(preds, clean_gt.squeeze())

                logits = model(corr_batch)[:, -1, :]
                preds = t.argmax(logits, dim=-1)
                correct_corrupt = t.eq(preds, corr_gt.squeeze())

                correct = t.logical_and(correct_clean, correct_corrupt)
                correct_list.extend(correct.tolist())

        correct_tensor = t.tensor(correct_list, dtype=t.bool)
    else:
        # Skip filtering
        correct_tensor = t.tensor([True] * len(clean_prompts_strs), dtype=t.bool)

    filtered_clean_prompts = [
        prompt for prompt, flag in zip(clean_prompts_strs, correct_tensor) if flag
    ]
    filtered_corrupt_prompts = [
        prompt for prompt, flag in zip(corrupt_prompts_strs, correct_tensor) if flag
    ]
    filtered_answers = [
        answer for answer, flag in zip(answer_strs, correct_tensor) if flag
    ]
    filtered_wrong_answers = [
        wrong_answer
        for wrong_answer, flag in zip(wrong_answer_strs, correct_tensor)
        if flag
    ]

    # Filter the data to only include examples that have the same token length
    if template != "full":
        (
            filtered_clean_prompts,
            filtered_corrupt_prompts,
            filtered_answers,
            filtered_wrong_answers,
        ) = filter_common_length_strings(
            filtered_clean_prompts,
            filtered_corrupt_prompts,
            filtered_answers,
            filtered_wrong_answers,
            tokenizer,
        )

        seq_labels = ensure_same_seq_labels(
            filtered_clean_prompts, model_name, tokenizer
        )

    else:
        seq_labels = []

    # Create new data structure
    new_data = {
        "seq_labels": seq_labels,
        "prompts": [
            {
                "clean": clean,
                "corrupt": corrupt,
                "answers": answer,
                "wrong_answers": wrong_answer,
            }
            for clean, corrupt, answer, wrong_answer in zip(
                filtered_clean_prompts,
                filtered_corrupt_prompts,
                filtered_answers,
                filtered_wrong_answers,
            )
        ],
    }

    # Save the new data to a JSON file
    model_save_name = model_name.split("/")[-1].lower()
    file_path = (
        f"{data_dir}/{template}/math_{model_save_name}_{perturbation}.json"
        if check_answers
        else f"{data_dir}/{template}/math_{model_save_name}_{perturbation}_no_check.json"
    )
    save_dict_to_json(data=new_data, file_path=file_path)
