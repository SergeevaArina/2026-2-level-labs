"""
Lab 2.

BPE and machine translation evaluation
"""

# pylint:disable=unused-argument
from json import load
from math import exp, log
from typing import Sequence


def prepare_word(
    raw_word: str, start_of_word: str | None, end_of_word: str | None
) -> tuple[str, ...] | None:
    """
    Tokenize a word into characters and attach optional boundary tokens.

    Args:
        raw_word (str): Original word
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str | None): A token that signifies the end of word

    Returns:
        tuple[str, ...] | None: Preprocessed word

    In case of corrupt input arguments, None is returned
    """
    if not(
        (isinstance(raw_word, str) and raw_word)
        and (isinstance(start_of_word, str) or start_of_word is None)
        and (isinstance(end_of_word, str) or end_of_word is None)
    ):
        return None

    tokens = []

    if start_of_word:
        tokens.append(start_of_word)

    for letter in raw_word:
        tokens.append(letter)

    if end_of_word:
        tokens.append(end_of_word)

    return tuple(tokens)


def collect_frequencies(
    text: str, start_of_word: str | None, end_of_word: str
) -> dict[tuple[str, ...], int] | None:
    """
    Count number of occurrences of each word.

    Args:
        text (str): Original text with no preprocessing
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str): A token that signifies the end of word

    Returns:
        dict[tuple[str, ...], int] | None: Dictionary where key - preprocessed word,
            value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if not(
        (isinstance(text, str) and text)
        and (isinstance(start_of_word, str) or start_of_word is None)
        and (isinstance(end_of_word, str))
    ):
        return None

    freq_dict = {}

    for word in text.split():
        prepared_word = prepare_word(word, start_of_word, end_of_word)
        if prepared_word is None:
            return None
        freq_dict[prepared_word] = freq_dict.get(prepared_word, 0) + 1

    return freq_dict


def count_tokens_pairs(
    word_frequencies: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int] | None:
    """
    Count number of occurrences of each pair of subsequent tokens.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences

    Returns:
        dict[tuple[str, str], int] | None: A dictionary where
            key - token pair, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(word_frequencies, dict):
        return None

    for word, frequency in word_frequencies.items():
        if not (
            isinstance(word, tuple)
            and all(isinstance(token, str) for token in word)
            and isinstance(frequency, int)
        ):
            return None

    freq_token_pairs = {}
    for key, value in word_frequencies.items():
        for i in range(len(key) - 1):
            pair_of_tokens = key[i], key[i + 1]
            freq_token_pairs[pair_of_tokens] = freq_token_pairs.get(pair_of_tokens, 0) + value

    return freq_token_pairs


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Update word frequency dictionary by replacing a pair of tokens with a merged one.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        pair (tuple[str, str]): A pair of tokens to be merged

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(word_frequencies, dict):
        return None

    for word, frequency in word_frequencies.items():
        if not (
            isinstance(word, tuple)
            and all(isinstance(token, str) for token in word)
            and isinstance(frequency, int)
        ):
            return None

    if not(
        isinstance(pair, tuple)
        and len(pair) == 2
        and all(isinstance(token, str) for token in pair)
    ):
        return None

    new_token = "".join(pair)
    freq_token = {}

    for key, value in word_frequencies.items():
        list_of_tokens = list(key)
        i = 0
        merged = []
        while i < len(list_of_tokens):
            if (
                i < len(list_of_tokens) - 1
                and list_of_tokens[i] == pair[0]
                and list_of_tokens[i + 1] == pair[1]
            ):
                merged.append(new_token)
                i += 2
            else:
                merged.append(list_of_tokens[i])
                i += 1

        key = tuple(merged)
        freq_token[key] = freq_token.get(key, 0) + value

    return freq_token


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Create required number of new tokens by merging existing ones.

    Args:
        word_frequencies (dict[tuple[str, ...], int] | None): A dictionary where
            key - preprocessed word, value - number of occurrences
        num_merges (int): Required number of new tokens

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if word_frequencies is None:
        return None

    if not (
        isinstance(word_frequencies, dict)
        and isinstance(num_merges, int)
        and num_merges > 0
    ):
        return None

    for word, frequency in word_frequencies.items():
        if not (
            isinstance(word, tuple)
            and all(isinstance(token, str) for token in word)
            and isinstance(frequency, int)
        ):
            return None

    for _ in range(num_merges):
        pairs = count_tokens_pairs(word_frequencies)

        if pairs is None:
            return None

        if not pairs:
            break

        best_pair = min(
        pairs.items(),
        key=lambda item: (
            -item[1],
            -len(item[0][0] + item[0][1]),
            item[0][0] + item[0][1]
            ),
        )[0]

        word_frequencies = merge_tokens(word_frequencies, best_pair)

        if word_frequencies is None:
            return None

    return word_frequencies


def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establish correspondence between tokens and its integer identifier.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        unknown_token (str): A token to signify an unknown token

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if not(
        isinstance(word_frequencies, dict)
    ):
        return None

    for word, frequency in word_frequencies.items():
        if not (
            isinstance(word, tuple)
            and all(isinstance(token, str) for token in word)
            and isinstance(frequency, int)
        ):
            return None

    if not(
        isinstance(unknown_token, str) and unknown_token
    ):
        return None

    uniq_voc = {unknown_token}

    for key in word_frequencies:
        for token in key:
            for symbol in token:
                uniq_voc.add(symbol)
            uniq_voc.add(token)

    sorted_uniq_voc = sorted(
        uniq_voc,
        key=lambda token: (-len(token), token)
        )

    new_uniq_voc = {}
    for i, token in enumerate(sorted_uniq_voc):
        new_uniq_voc[token] = i

    return new_uniq_voc


def decode(
    encoded_text: Sequence[int] | None,
    vocabulary: dict[str, int] | None,
    end_of_word_token: str | None,
) -> str | None:
    """
    Translate encoded sequence into decoded one.

    Args:
        encoded_text (Sequence[int] | None): A sequence of token identifiers
        vocabulary (dict[str, int] | None): A dictionary where
            key - token, value - identifier
        end_of_word_token (str | None): An end-of-word token

    Returns:
        str | None: Decoded sequence

    In case of corrupt input arguments, None is returned
    """
    if (
        encoded_text is None
        or vocabulary is None
    ):
        return None

    if not(
        (isinstance(encoded_text, Sequence) and encoded_text)
        and all(
            isinstance(symbol, int)
            for symbol in encoded_text
            )
    ):
        return None

    if not(
        (isinstance(vocabulary, dict) and vocabulary)
        and all(
            isinstance(token, str) and isinstance(index, int)
            for token, index in vocabulary.items()
        )
    ):
        return None

    if not(
        end_of_word_token is None
        or (isinstance(end_of_word_token, str) and end_of_word_token)
    ):
        return None

    inverse = {index: token for token, index in vocabulary.items()}
    decoded_token_list = []
    for index in encoded_text:
        token = inverse.get(index)
        if token is None:
            return None
        decoded_token_list.append(token)

    decoded_text = "".join(decoded_token_list)

    if end_of_word_token is not None:
        decoded_text = decoded_text.replace(end_of_word_token, " ")

    return decoded_text


def tokenize_word(
    word: tuple[str, ...], vocabulary: dict[str, int], end_of_word: str | None, unknown_token: str
) -> list[int] | None:
    """
    Split word into tokens.

    Args:
        word (tuple[str, ...]): Preprocessed word
        vocabulary (dict[str, int]): A dictionary where key - token, value - identifier
        end_of_word (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments, None is returned
    """
    if not(
        isinstance(word, tuple) and word
        and all(
            isinstance(token, str) for token in word
        )
        and isinstance(vocabulary, dict) and vocabulary
        and all(
            isinstance(key, str) and isinstance(value, int)
            for key, value in vocabulary.items()
        )
        and isinstance(unknown_token, str) and unknown_token
    ):
        return None

    if not(
        end_of_word is None
        or (isinstance(end_of_word, str) and end_of_word)
    ):
        return None

    str_word = "".join(word)
    if (
        end_of_word is not None
        and not str_word.endswith(end_of_word)
    ):
        str_word += end_of_word

    parts = ["0"] * len(str_word)

    sorted_tokens = sorted(
        vocabulary.keys(), key=lambda token: (-len(token), token)
    )

    for token in sorted_tokens:
        if token and token != unknown_token:
            token_len = len(token)
            ind = 0
            while ind <= len(parts) - token_len:
                if (
                    not any(parts[ind:ind + token_len])
                    and str_word[ind:ind + token_len] == token
                ):
                    parts[ind:ind + token_len] = [token] * token_len
                    ind += token_len
                else:
                    ind += 1

    unknown_id = vocabulary.get(unknown_token)
    if unknown_id is None:
        return None

    encoded_ids = []
    ind = 0
    while ind < len(parts):
        part = parts[ind]
        if part is None:
            encoded_ids.append(unknown_id)
            ind += 1
        else:
            encoded_ids.append(vocabulary[part])
            ind += len(part)

    return encoded_ids


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Read and retrieve dictionary of type <token: identifier>.

    Args:
        vocab_path (str): A path to the saved vocabulary

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(vocab_path, str):
        return None

    with open(vocab_path, "r", encoding="utf-8") as file:
        data = load(file)

    if not isinstance(data, dict):
        return None

    return data


def encode(
    original_text: str,
    vocabulary: dict[str, int] | None,
    start_of_word_token: str | None,
    end_of_word_token: str | None,
    unknown_token: str,
) -> list[int] | None:
    """
    Translate original text into a sequence of token identifiers.

    Args:
        original_text (str): Original text
        vocabulary (dict[str, int] | None): A dictionary where key - token, value - identifier
        start_of_word_token (str | None): A start-of-word token
        end_of_word_token (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if not (
        isinstance(original_text, str)
        and original_text
        and (
            start_of_word_token is None
            or isinstance(start_of_word_token, str)
            and start_of_word_token
            )
        and (
            end_of_word_token is None
            or isinstance(end_of_word_token, str)
            and end_of_word_token
            )
        and isinstance(unknown_token, str)
        and unknown_token
    ):
        return None

    if not(
        isinstance(vocabulary, dict)
        and vocabulary
        and all(
            isinstance(key, str) and isinstance(value, int)
            for key, value in vocabulary.items()
            )
    ):
        return None

    encoded = []

    for raw_word in original_text.split():
        prepared_word = prepare_word(raw_word, start_of_word_token, end_of_word_token)

        if prepared_word is None:
            return None

        tokenized_word = tokenize_word(prepared_word, vocabulary, end_of_word_token, unknown_token)

        if tokenized_word is None:
            return None

        encoded.extend(tokenized_word)

    return encoded


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extract n-grams from the given sequence.

    Args:
        text (str): Original text
        order (int): Required number of elements in a single n-gram

    Returns:
        list[tuple[str, ...]] | None: A sequence of n-grams

    In case of corrupt input arguments, None is returned
    """
    if not(
        isinstance(text, str) and text
        and isinstance(order, int) and order > 0
    ):
        return None

    n_grams = []

    for i in range(len(text) - order + 1):
        n_grams.append(tuple(text[i:i + order]))

    return n_grams


def calculate_precision(
    actual: Sequence[tuple[str, ...]], reference: Sequence[tuple[str, ...]]
) -> float | None:
    """
    Compare two sequences by virtue of Precision metric.

    Args:
        actual (Sequence[tuple[str, ...]]): Predicted sequence of n-grams
        reference (Sequence[tuple[str, ...]]): Expected sequence of n-grams

    Returns:
        float | None: Value of Precision metric

    In case of corrupt input arguments, None is returned.
    """
    if not (
        isinstance(actual, Sequence)
        and isinstance(reference, Sequence)
    ):
        return None

    for n_gram in actual:
        if not(
            isinstance(n_gram, tuple)
            and all(
                isinstance(token, str) for token in n_gram
            )
        ):
            return None

    for n_gram in reference:
        if not(
            isinstance(n_gram, tuple)
            and all(
                isinstance(token, str)
                for token in n_gram
            )
        ):
            return None

    if not actual:
        return 0.0

    matched = set(actual) & set(reference)

    return len(matched) / len(set(actual))


def calculate_geo_mean(precisions: Sequence[float], max_order: int) -> float | None:
    """
    Compute geometric mean of sequence of values.

    Args:
        precisions (Sequence[float]): A sequence of Precision values
        max_order (int): Maximum length of n-gram considered

    Returns:
        float | None: A value of geometric mean of Precision metric

    In case of corrupt input arguments, None is returned
    """
    if not(
        isinstance(precisions, Sequence)
        and precisions
        and all(
            isinstance(precision, (float, int))
            for precision in precisions
        )
        and isinstance(max_order, int)
        and max_order > 0
    ):
        return None

    log_n_gram = 0.0
    for precision in precisions:
        if precision <= 0:
            return 0.0
        log_n_gram += log(precision)

    return exp(log_n_gram / max_order)


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compare two sequences by virtue of BLEU metric.

    Args:
        actual (str | None): Predicted sequence
        reference (str): Expected sequence
        max_order (int, optional): Max length of n-gram to consider for comparison. Defaults to 3

    Returns:
        float | None: A value of BLEU metric

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if not(
        isinstance(reference, str)
        and reference
        and isinstance(max_order, int)
        and max_order > 0
    ):
        return None

    if (
        not(isinstance(actual, str) and actual)
        or actual is None
    ):
        return None

    precisions = []

    for i in range(1, max_order + 1):
        act_n_gram = collect_ngrams(actual, i)
        ref_n_gram = collect_ngrams(reference, i)

        if act_n_gram is None or ref_n_gram is None:
            return None

        precision = calculate_precision(act_n_gram, ref_n_gram)

        if precision is None:
            return None

        precisions.append(precision)

    geo_mean = calculate_geo_mean(precisions, max_order)

    if geo_mean is None:
        return None

    return geo_mean * 100
