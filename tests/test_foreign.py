import numpy as np
import pytest

from agregador.foreign import (
    CROSS_COS, SECTION_SHARE, centroid, entity_stems, judge, section, section_share,
)


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("url, expected", [
    ("https://elpais.com/internacional/2026-10-19/francia.html", "internacional"),
    ("https://www.elmundo.es/Mundo/x", "mundo"),
    ("https://www.abc.es/espana/politica/x.html", "espana"),
    ("https://www.example.com/", ""),
    ("not a url", "not a url"),
])
def test_section_is_the_first_path_segment_lowercased(url, expected):
    assert section(url) == expected


def test_section_share_counts_articles_under_international_sections():
    urls = ["https://a.es/internacional/1", "https://a.es/mundo/2", "https://a.es/espana/3", "https://a.es/espana/4"]
    assert section_share(urls) == 0.5
    assert section_share([]) == 0.0 and section_share(["https://a.es/espana/1"]) == 0.0


def test_entity_stems_skip_the_first_word_and_match_across_spanish_and_english():
    es = entity_stems(["Zelenski acusa a Trump de querer que Ucrania capitule en Francia"])
    en = entity_stems(["Zelensky Says Trump Is Pushing Ukraine to Surrender in France"])
    assert {"trump", "zelen"} <= es & en or {"trump"} <= es & en
    assert "zelen" not in es                      # "Zelenski" opens the headline: skipped like any sentence start
    assert "franc" in entity_stems(["Dimite el ministro en Francia"]) & entity_stems(["Minister quits in France"])


def test_centroid_is_normalized_mean():
    c = centroid(np.stack([unit(1, 0), unit(0, 1)]))
    assert pytest.approx(float(np.linalg.norm(c))) == 1.0 and pytest.approx(float(c[0])) == float(c[1])


FOREIGN_URLS = ["https://a.es/internacional/1", "https://a.es/internacional/2", "https://a.es/espana/3"]
SPAIN_URLS = ["https://a.es/espana/1", "https://a.es/espana/2", "https://a.es/politica/3"]


def test_enough_articles_in_international_sections_make_a_story_foreign():
    v = judge(FOREIGN_URLS, None, frozenset(), [])
    assert v.foreign and v.reason == "section" and v.section_share >= SECTION_SHARE
    assert not judge(SPAIN_URLS, None, frozenset(), []).foreign


def test_a_match_with_another_edition_needs_both_similarity_and_shared_names():
    others = [(unit(1, 0.0), frozenset({"trump", "putin", "china"}))]
    close = unit(1, 0.05)                                                   # cosine ~0.999
    assert judge(SPAIN_URLS, close, frozenset({"trump", "putin"}), others).reason == "match"
    assert not judge(SPAIN_URLS, close, frozenset({"trump"}), others).foreign           # one name is not enough
    far = unit(1, 0.6)                                                       # cosine ~0.86 < CROSS_COS
    assert far @ others[0][0] < CROSS_COS
    assert not judge(SPAIN_URLS, far, frozenset({"trump", "putin"}), others).foreign    # names alone are not enough


def test_both_signals_are_reported_and_nothing_to_compare_with_is_not_an_error():
    others = [(unit(1, 0.0), frozenset({"trump", "putin"}))]
    v = judge(FOREIGN_URLS, unit(1, 0.02), frozenset({"trump", "putin"}), others)
    assert v.foreign and v.reason == "section+match" and v.shared == 2
    assert not judge(SPAIN_URLS, unit(1, 0.0), frozenset({"trump"}), []).foreign


def test_the_best_match_is_chosen_by_similarity_plus_shared_names():
    others = [(unit(1, 0.30), frozenset()), (unit(1, 0.34), frozenset({"trump", "putin"}))]
    v = judge(SPAIN_URLS, unit(1, 0.30), frozenset({"trump", "putin"}), others)
    assert v.shared == 2                                   # the second wins thanks to its shared names
