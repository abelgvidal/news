"""Story images: downloaded once per edition, shrunk to a thumbnail and re-encoded.

Feed images are untrusted: only public http(s) hosts are fetched, size and pixel counts are capped, and the
file written is always our own WebP re-encoding (no metadata of the original survives). The file carries the
story it belongs to in its EXIF (headline, outlet and article URL) so it stays meaningful away from its page.
"""
import hashlib
import io
import ipaddress
import logging
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)

WIDTH = 800  # thumbnails are at most this wide
QUALITY = 75
MIN_WIDTH = 200  # smaller than this is an icon or a tracking pixel, not a story image
MAX_BYTES = 8_000_000
MAX_PIXELS = 40_000_000
MAX_REDIRECTS = 3
MAX_TRIES = 3  # candidate images tried per story
TIMEOUT = 10.0
_EXIF_DESCRIPTION, _EXIF_SOFTWARE, _EXIF_COPYRIGHT = 0x010E, 0x0131, 0x8298


def _public(url: str) -> bool:
    """True for an http(s) URL whose host resolves only to public addresses (no loopback, LAN or metadata IPs)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def fetch(client: httpx.Client, url: str) -> bytes | None:
    """The bytes of an image URL, or None if it is unsafe, too big, not an image or fails."""
    try:
        for _ in range(MAX_REDIRECTS + 1):
            if not _public(url):
                return None
            with client.stream("GET", url, follow_redirects=False, timeout=TIMEOUT) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
                    return None
                data = bytearray()
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) > MAX_BYTES:
                        return None
                return bytes(data)
    except httpx.HTTPError as e:
        log.info("image %s: %s", url, e)
    return None


def thumbnail(data: bytes, description: str, copyright_: str) -> tuple[bytes, int, int] | None:
    """(WebP bytes, width, height) of the image shrunk to WIDTH, with the story in its EXIF; None if unusable."""
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        im = ImageOps.exif_transpose(im)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return None
    if im.width < MIN_WIDTH:
        return None
    rgba = im.convert("RGBA")  # flatten transparency onto white
    flat = Image.new("RGB", rgba.size, "white")
    flat.paste(rgba, mask=rgba.getchannel("A"))
    if flat.width > WIDTH:
        flat.thumbnail((WIDTH, 10 ** 5), Image.Resampling.LANCZOS)
    exif = Image.Exif()
    exif[_EXIF_DESCRIPTION], exif[_EXIF_SOFTWARE], exif[_EXIF_COPYRIGHT] = description, "agregador", copyright_
    out = io.BytesIO()
    flat.save(out, "WEBP", quality=QUALITY, exif=exif.tobytes())
    return out.getvalue(), flat.width, flat.height


def save_story_image(client: httpx.Client, candidates: list[dict], headline: str, target: Path) -> dict | None:
    """Download the first usable candidate ({url, outlet, article_url}) to `target`; return what we know of it."""
    for c in candidates[:MAX_TRIES]:
        data = fetch(client, c["url"])
        made = data and thumbnail(
            data, headline, f"Image from {c['outlet']} ({c['article_url']}); original: {c['url']}")
        if not made:
            continue
        webp, width, height = made
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(webp)
        return {"file": target.name, "width": width, "height": height, "sha256": hashlib.sha256(webp).hexdigest(),
                "original_url": c["url"], "outlet": c["outlet"], "article_url": c["article_url"]}
    return None
