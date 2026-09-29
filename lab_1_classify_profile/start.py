"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code, too-many-return-statements
from lab_1_classify_profile.main import (
    calculate_frequencies,
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

    tokenized_text = tokenize(de_text)
    if tokenized_text is None:
        return None

    text_without_stopwords = remove_stop_words(tokenized_text, stopwords)
    if text_without_stopwords is None:
        return None

    calculated_frequencies = calculate_frequencies(text_without_stopwords)

    if calculated_frequencies is None:
        return None

    unk_profile = create_language_profile("unknown", unknown_text, stopwords)
    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)

    if (unk_profile is None
        or de_profile is None
        or en_profile is None):
        return None

    print(get_top_n_words(calculated_frequencies, 7))
    print(detect_language_by_top_n(unk_profile, en_profile, de_profile, 15))
    result = detect_language_by_mse(unk_profile, en_profile, de_profile)

    save_profile(unk_profile, 'lab_1_classify_profile/assets/profiles')
    save_profile(de_profile, 'lab_1_classify_profile/assets/profiles')
    save_profile(en_profile, 'lab_1_classify_profile/assets/profiles')

    list_of_paths = ['lab_1_classify_profile/assets/profiles/la.json',
                     'lab_1_classify_profile/assets/profiles/de.json',
                     'lab_1_classify_profile/assets/profiles/en.json']
    collected_profiles = collect_profiles(list_of_paths)

    if collected_profiles is None:
        return None

    advanced_detection = detect_language_advanced(unk_profile, collected_profiles, 15)

    if advanced_detection is None:
        return None

    print_report(unk_profile, advanced_detection, 15)

    assert result, "Detection result is None"
    return None

    tokens = tokenize(de_text)
    if tokens is None:
        return

    clean_tokens = remove_stop_words(tokens, stopwords)
    if clean_tokens is None:
        return

    freq_dict = calculate_frequencies(clean_tokens)
    if freq_dict is None:
        return

    result = get_top_n_words(freq_dict, top_n = 7)
    if result is None:
        return

    print(f"7 наиболее частотных токенов:{result}")

    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)
    unknown_profile = create_language_profile("unknown", unknown_text, stopwords)

    if (
        de_profile is None
        or en_profile is None
        or unknown_profile is None
    ):
        return

    if (
        check_profile(unknown_profile) is False
        or check_profile(de_profile) is False
        or check_profile(en_profile) is False
    ):
        return

    result = detect_language_by_top_n(unknown_profile, de_profile, en_profile, top_n = 15)
    if result is None:
        return
    print(f"Наиболее вероятный язык по метрике top_n слов:{result}")


    result = detect_language_by_mse(unknown_profile, en_profile, de_profile)
    if result is None:
        return
    print(f"Наиболее вероятный язык по mse метрике:{result}")
    assert result, "Detection result is None"

if __name__ == "__main__":
    main()
