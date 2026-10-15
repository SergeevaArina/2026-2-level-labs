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
        return None
    result = get_vocabulary(word_frequencies, "<unk>")
    print(result)

    assert result, "Translation not working"


if __name__ == "__main__":
    main()
