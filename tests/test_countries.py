import pytest

from agregador.countries import MIN_MENTIONS, primary_country, tally


def test_a_place_counts_once_per_headline_and_demonyms_are_places():
    assert tally(["France imposes curfews in Paris"])["France"] == 2.0        # country + capital
    assert tally(["France, France, France"])["France"] == 1.0                 # the same term counts once per headline
    assert tally(["French cities calm"])["France"] == 1.0


def test_people_weigh_less_than_places():
    t = tally(["Macron visits Berlin"])
    assert t["France"] == 0.5 and t["Germany"] == 1.0


def test_uppercase_US_and_UK_are_countries_but_the_pronoun_us_is_not():
    assert tally(["US charges 28 in California"])["United States"] == 2.0     # "US" + California
    assert tally(["U.S. to launch farm program"])["United States"] == 1.0
    assert tally(["UK wage growth picks up"])["Britain"] == 1.0
    assert not tally(["Let us go home"])


def test_longer_terms_win_and_accents_do_not_matter():
    assert tally(["South Africa elects a president"]) == {"South Africa": 1.0}     # not just "africa"
    assert tally(["Lula visits São Paulo"])["Brazil"] == 1.5


def test_the_primary_country_needs_enough_mentions_and_a_clear_majority():
    assert primary_country(["France Imposes Curfews in Parts of Paris"]) == "France"
    assert primary_country(["Macron visits Berlin"]) is None                      # weight 1.5 < MIN_MENTIONS
    assert MIN_MENTIONS == 2.0
    assert primary_country(["Putin Floats a Summit With Trump and Xi to Discuss Ukraine Deal"]) is None   # mixed
    assert primary_country(["Possible wreckage of Amelia Earhart's plane likely to be trees"]) is None


def test_a_story_about_one_country_that_mentions_another_keeps_its_own():
    assert primary_country(["Trump warns Kansans that US will 'burn' like France if Democrats win"]) == "United States"


def test_all_the_headlines_of_a_story_vote_together():
    assert primary_country(["Rioting sweeps the nation", "France in shock", "Paris shops looted"]) == "France"
