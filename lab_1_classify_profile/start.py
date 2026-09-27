"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code, too-many-return-statements
from lab_1_classify_profile.main import (
    calculate_frequencies,
    check_profile,
    create_language_profile,
    detect_language_by_top_n,
    get_top_n_words,
    remove_stop_words,
    tokenize,
)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()
    result = None

    tokens = tokenize(de_text)
    if tokens is None:
        return None

    clean_tokens = remove_stop_words(tokens, stopwords)
    if clean_tokens is None:
        return None

    freq_dict = calculate_frequencies(clean_tokens)
    if freq_dict is None:
        return None

    result = get_top_n_words(freq_dict, top_n = 7)
    if result is None:
        return None

    print(result)

    de_prifile = create_language_profile("de", de_text, stopwords)
    en_prifile = create_language_profile("en", en_text, stopwords)
    unknown_prifile = create_language_profile("unknown", unknown_text, stopwords)

    if (
        de_prifile is None
        or en_prifile is None
        or unknown_prifile is None
    ):
        return None

    if (
        check_profile(unknown_prifile) is False
        or check_profile(de_prifile) is False
        or check_profile(en_prifile) is False
    ):
        return None

    result = detect_language_by_top_n(unknown_prifile, de_prifile, en_prifile, top_n = 15)
    if result is None:
        return None
    print(result)


    assert result, "Detection result is None"

if __name__ == "__main__":
    main()
