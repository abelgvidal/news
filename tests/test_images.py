import io

import httpx
from PIL import Image

from agregador import images
from agregador.images import fetch, save_story_image, thumbnail


def _png(width=1600, height=900, mode="RGB", color="red") -> bytes:
    out = io.BytesIO()
    Image.new(mode, (width, height), color).save(out, "PNG")
    return out.getvalue()


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_thumbnail_is_shrunk_to_webp_and_names_its_story_in_the_exif():
    webp, w, h = thumbnail(_png(), "Titular de la noticia", "Image from A (https://a.example/1)")
    im = Image.open(io.BytesIO(webp))
    assert im.format == "WEBP" and (w, h) == (images.WIDTH, 450) == im.size
    exif = im.getexif()
    assert exif[0x010E] == "Titular de la noticia" and "https://a.example/1" in exif[0x8298]


def test_a_small_image_is_not_enlarged_and_a_tiny_one_is_rejected():
    assert thumbnail(_png(400, 300), "t", "c")[1:] == (400, 300)
    assert thumbnail(_png(50, 50), "t", "c") is None                 # an icon or a tracking pixel


def test_transparency_is_flattened_and_garbage_is_rejected():
    assert thumbnail(_png(300, 300, "RGBA", (0, 0, 0, 0)), "t", "c") is not None
    assert thumbnail(b"<html>not an image</html>", "t", "c") is None


def test_private_and_non_http_hosts_are_never_fetched():
    called = []
    client = _client(lambda r: called.append(r) or httpx.Response(200, content=_png(), headers={"content-type": "image/png"}))
    for url in ("http://127.0.0.1/a.png", "http://169.254.169.254/latest", "file:///etc/passwd", "ftp://x.example/a"):
        assert fetch(client, url) is None
    assert not called


def test_only_images_are_taken(monkeypatch):
    monkeypatch.setattr(images, "_public", lambda url: True)
    ok = _client(lambda r: httpx.Response(200, content=_png(), headers={"content-type": "image/png"}))
    assert fetch(ok, "https://i.example/a.png") == _png()
    html = _client(lambda r: httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"}))
    assert fetch(html, "https://i.example/a.png") is None
    assert fetch(_client(lambda r: httpx.Response(404)), "https://i.example/a.png") is None


def test_an_oversized_download_is_dropped(monkeypatch):
    monkeypatch.setattr(images, "_public", lambda url: True)
    monkeypatch.setattr(images, "MAX_BYTES", 100)
    big = _client(lambda r: httpx.Response(200, content=b"x" * 1000, headers={"content-type": "image/png"}))
    assert fetch(big, "https://i.example/a.png") is None


def test_a_redirect_to_a_private_host_is_refused(monkeypatch):
    monkeypatch.setattr(images, "_public", lambda url: "127.0.0.1" not in url)
    client = _client(lambda r: httpx.Response(302, headers={"location": "http://127.0.0.1/secret.png"}))
    assert fetch(client, "https://i.example/a.png") is None


def test_save_story_image_skips_broken_candidates_and_records_where_it_came_from(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "_public", lambda url: True)

    def handler(request):
        if request.url.path == "/bad.png":
            return httpx.Response(404)
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    cands = [{"url": "https://i.example/bad.png", "outlet": "A", "article_url": "https://a.example/1"},
             {"url": "https://i.example/good.png", "outlet": "B", "article_url": "https://b.example/2"}]
    info = save_story_image(_client(handler), cands, "Titular", tmp_path / "img" / "es-01-7.webp")
    assert info["outlet"] == "B" and info["article_url"] == "https://b.example/2"
    assert info["original_url"] == "https://i.example/good.png" and len(info["sha256"]) == 64
    assert (tmp_path / "img" / "es-01-7.webp").exists()
    assert save_story_image(_client(lambda r: httpx.Response(404)), cands, "T", tmp_path / "x.webp") is None
