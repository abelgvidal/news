from agregador.normalize import clean_text, normalize_url


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>Hola   <b>mundo</b></p>") == "Hola mundo"
    assert clean_text(None) == ""


def test_normalize_url_drops_tracking_and_fragment():
    url = "https://Example.com/a?utm_source=x&id=7&fbclid=z#frag"
    assert normalize_url(url) == "https://example.com/a?id=7"
