"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable
from pathlib import Path

from lab_2_tokenize_by_bpe.main import (
    collect_frequencies,
    get_vocabulary,
    train,
)


def main() -> None:
    """
    Launches an implementation
    """
    assets_dir = Path(__file__).parent / "assets"

    with open(assets_dir/"text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open(assets_dir/"en_raw.txt", "r", encoding="utf-8") as text_file:
        text_reference_translation = text_file.read()
    with open(assets_dir/"en_encoded.txt", "r", encoding="utf-8") as text_file:
        translation_encoded_raw = text_file.read()
    result = None
    result = collect_frequencies(text, None, "</s>")
    print(result)

    word_frequencies = train(result, 100)
    if word_frequencies is None:
        return
    new_vocabularly = get_vocabulary(word_frequencies, "<unk>")
    print(new_vocabularly)

    assert result, "Translation not working"

    result = collect_frequencies(text, None, "</s>")

    assert result, "Frequency dictionary is None"
    print(result)

    bpe_train = train(result, 100)
    if bpe_train is None:
        return
    print(bpe_train)

    vocabulary = get_vocabulary(bpe_train,'<unk>')
    if vocabulary is None:
        return

    # "secret"
    secret3 = [int(number) for number in secret3_raw.split()]
    secret = decode(secret3, vocabulary, '</s>')
    if secret is None:
        return
    print(secret)

    # "mark 8"
    with open("lab_2_tokenize_by_bpe/assets/ru_raw.txt", "r", encoding="utf-8") as text_file:
        ru_text = text_file.read()

    ru_vocabulary = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")
    if ru_vocabulary is None:
        return

    encoded_ru = encode(
        ru_text,
        ru_vocabulary,
        start_of_word_token="\u2581",
        end_of_word_token=None,
        unknown_token="<unk>",
    )
    if encoded_ru is None:
        return
    # "comparison"
    with open("lab_2_tokenize_by_bpe/assets/ru_encoded.txt", "r", encoding="utf-8") as encoded_file:
        ru_to_compare_raw = encoded_file.read()

    ru_to_compare = [int(number) for number in ru_to_compare_raw.split()]

    n = 20
    # print("Encoded: ", encoded_ru[:n])
    # print("To compare: ", ru_to_compare[:n])

    # "mark 10"
    translation_encoded = [int(number) for number in translation_encoded_raw.split()]

    en_vocabulary = load_vocabulary("lab_2_tokenize_by_bpe/assets/vocab.json")
    if en_vocabulary is None:
        return

    translation_decoded = decode(
        translation_encoded,
        en_vocabulary,
        None,
    )
    assert translation_decoded, "Decoded translation is None"

    translation_decoded = translation_decoded.replace("\u2581", " ")

    bleu = calculate_bleu(
        translation_decoded,
        text_reference_translation,
    )

    assert bleu, "BLEU is None"

    print("Predicted translation:", translation_decoded)
    print("Reference translation:", text_reference_translation)
    print("BLEU:", bleu)

if __name__ == "__main__":
    main()
