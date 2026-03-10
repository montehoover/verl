# Implementation of create_forum_post to build a sanitized HTML forum post.

from html import escape
from typing import List
import re
from urllib.parse import urlparse

# Predefined HTML template string for forum posts, containing placeholders for the title, main content, and discussion points
FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

_LINK_MD_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

_ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}


def _safe_render_markdown_inline(md_text: str) -> str:
    """
    Safely render a minimal subset of inline Markdown within a string:
    - Links [text](url) with only http, https, or mailto schemes are converted to <a href="...">text</a>.
    - All other content is HTML-escaped.
    This function does not support full Markdown; it prioritizes safety.
    """
    if not md_text:
        return ""

    out_parts: List[str] = []
    idx = 0
    for m in _LINK_MD_RE.finditer(md_text):
        # Escape text leading up to the match
        if m.start() > idx:
            out_parts.append(escape(md_text[idx:m.start()], quote=True))

        link_text_raw = m.group(1)
        link_href_raw = m.group(2).strip()

        # Validate URL scheme for safety
        parsed = urlparse(link_href_raw)
        scheme = (parsed.scheme or "").lower()

        if scheme in _ALLOWED_URL_SCHEMES:
            safe_text = escape(link_text_raw, quote=True)
            # Escape URL for attribute context and add rel to mitigate tabnabbing
            safe_href = escape(link_href_raw, quote=True)
            out_parts.append(f'<a href="{safe_href}" rel="noopener noreferrer">{safe_text}</a>')
        else:
            # If not a safe URL, do not render as a link; escape literal markdown
            out_parts.append(escape(m.group(0), quote=True))

        idx = m.end()

    # Append any trailing text after the last match
    if idx < len(md_text):
        out_parts.append(escape(md_text[idx:], quote=True))

    return "".join(out_parts)


def _format_discussion_points(points: List[str]) -> str:
    """
    Pure function that formats discussion points into a sanitized HTML unordered list (<ul>).
    Each point supports a minimal subset of inline Markdown (links) via _safe_render_markdown_inline.
    Returns an empty string if no points are provided.
    """
    if not points:
        return ""

    items_html = []
    for p in points:
        rendered = _safe_render_markdown_inline(p)
        items_html.append(f"<li>{rendered}</li>")
    return f"<ul>{''.join(items_html)}</ul>"


def create_forum_post(topic: str, text_body: str, points: List[str]) -> str:
    """
    Create a structured, sanitized HTML representation of a forum post.

    Arguments:
    - topic: str, the title or main topic of the forum post
    - text_body: str, the main body or content of the forum post
    - points: list(str), discussion points/options in markdown format

    Returns:
    - str: fully formatted and sanitized HTML including header, message body, and discussion points rendered as an unordered list.
    """
    if not isinstance(topic, str):
        raise TypeError("topic must be a string")
    if not isinstance(text_body, str):
        raise TypeError("text_body must be a string")
    if not isinstance(points, list) or not all(isinstance(p, str) for p in points):
        raise TypeError("points must be a list of strings")

    # Sanitize header and message
    safe_header = escape(topic, quote=True)
    safe_message = escape(text_body, quote=True)

    # Render discussion points as a sanitized unordered list
    options_html = _format_discussion_points(points)

    return FORUM_POST_FORMAT.format(header=safe_header, message=safe_message, options=options_html)
