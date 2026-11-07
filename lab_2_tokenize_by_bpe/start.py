"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable

from lab_2_tokenize_by_bpe.main import (
    calculate_bleu,
    collect_frequencies,
    decode,
    encode,
    get_vocabulary,
    load_vocabulary,
    train,
)


def main() -> None:
    """
    Launches an implementation
    """

    with open(
        "lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8"
        )as text_file:
        text = text_file.read()

    with open(
        "lab_2_tokenize_by_bpe/assets/en_raw.txt", "r", encoding="utf-8"
        ) as text_file:
        text_reference_translation = text_file.read()

    with open(
        "lab_2_tokenize_by_bpe/assets/en_encoded.txt", "r", encoding="utf-8"
        ) as text_file:
        translation_encoded_raw = text_file.read()


    result = collect_frequencies(text, None, "</s>")
    assert result, "Frequency dictionary is None"
    print(result)

    word_frequencies = train(result, 100)
    if word_frequencies is None:
        return None
    print(word_frequencies)

    new_vocabularly = get_vocabulary(word_frequencies, "<unk>")
    if new_vocabularly is None:
        return None
    print(new_vocabularly)

    print()

    with open(
        "lab_2_tokenize_by_bpe/assets/ru_raw.txt", "r", encoding="utf-8"
        ) as text_file:
        ru_text = text_file.read()
    with open(
        "lab_2_tokenize_by_bpe/assets/ru_encoded.txt", "r", encoding="utf-8"
        ) as text_file:
        ru_encoded_to_compare = text_file.read()

    ru_vocabulary = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")

    if ru_vocabulary is None:
        return None


    ru_encoded = encode(
        ru_text,
        ru_vocabulary,
        start_of_word_token="\u2581",
        end_of_word_token=None,
        unknown_token="<unk>",
    )

    if ru_encoded is None:
        return None

    ru_to_compare = [int(index) for index in ru_encoded_to_compare.split()]

    if ru_encoded == ru_to_compare:
        print(
    f"""
Encoded text: {ru_encoded[:20]}
Encoded text to compare: {ru_to_compare[:20]}
    """
    )

    print()

    translation_encoded = [int(number) for number in translation_encoded_raw.split()]
    en_vocabulary = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")

    if en_vocabulary is None:
        return None

    translation_decoded = decode(
        translation_encoded,
        en_vocabulary,
        None,
    )

    assert translation_decoded, "Translation not working"

    translation_decoded = translation_decoded.replace("\u2581", " ")

    bleu = calculate_bleu(
        translation_decoded,
        text_reference_translation,
    )

    assert bleu, "BLEU is None"

    print(
    f"""
Predicted translation: {translation_decoded}
Reference translation: {text_reference_translation}
BLEU: {bleu}
    """
    )

    assert bleu, "BLUE is None"
    return None


if __name__ == "__main__":
    main()
