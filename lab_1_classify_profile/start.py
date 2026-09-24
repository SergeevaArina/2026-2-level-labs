"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code
from lab_1_classify_profile.main import (
    calculate_frequencies,
    check_profile,
    collect_profiles,
    create_language_profile,
    detect_language_advanced,
    detect_language_by_mse,
    detect_language_by_top_n,
    get_top_n_words,
    print_report,
    remove_stop_words,
    save_profile,
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

    top_n_list = get_top_n_words(freq_dict, top_n = 7)
    if top_n_list is None:
        return None

    print(top_n_list)

    de_prifile = create_language_profile("de", de_text, stopwords)
    en_prifile = create_language_profile("en", en_text, stopwords)
    unknown_prifile = create_language_profile("unknown", unknown_text, stopwords)

    # if (
    #     de_prifile is None
    #     or en_prifile is None
    #     or unknown_prifile is None
    # ):
    #     return None

    check_unknown_prifile = check_profile(unknown_prifile)
    check_de_prifile = check_profile(de_prifile)
    check_en_prifile = check_profile(en_prifile)

    if (
        check_unknown_prifile is False
        or check_de_prifile is False
        or check_en_prifile is False
    ):
        return None

    language = detect_language_by_top_n(unknown_prifile, en_prifile, en_prifile, top_n = 15)
    if language is None:
        return None
    print(language)

    #assert result, "Detection result is None"

if __name__ == "__main__":
    main()
