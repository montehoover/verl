"""
Utilities to build a sanitized HTML forum post.

This module exposes create_forum_post(...), which escapes user-provided
content and renders discussion points as an HTML unordered list while
supporting a minimal subset of inline Markdown (links).

Security considerations:
- HTML is escaped using html.escape to prevent injection.
- Only http, https, and mailto URLs are allowed for links in points.
- Links include rel="noopener noreferrer" to mitigate tabnabbing.
"""

import re
from html import escape
from typing import List
from urllib.parse import urlparse

# Predefined HTML template string for forum posts, containing placeholders for
# the title, main content, and discussion points.
FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

# Matches Markdown-style links: [text](url)
_LINK_MD_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Allowed URL schemes when rendering links from Markdown.
_ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}


def _safe_render_markdown_inline(md_text: str) -> str:
    """
    Safely render a minimal subset of inline Markdown within a string.

    Currently supported:
    - Links in the form [text](url) where the URL scheme is one of:
      http, https, or mailto.

    All other text is HTML-escaped to ensure safety. This function does not
    attempt full Markdown support; it prioritizes safety over completeness.

    Args:
        md_text: A string potentially containing inline Markdown.

    Returns:
        A string with safe HTML. Allowed links are converted to <a> elements,
        and all other content is HTML-escaped.
    """
    if not md_text:
        return ""

    out_parts: List[str] = []
    idx = 0

    for match in _LINK_MD_RE.finditer(md_text):
        # Escape text leading up to the match.
        if match.start() > idx:
            out_parts.append(escape(md_text[idx:match.start()], quote=True))

        link_text_raw = match.group(1)
        link_href_raw = match.group(2).strip()

        # Validate URL scheme for safety.
        parsed = urlparse(link_href_raw)
        scheme = (parsed.scheme or "").lower()

        if scheme in _ALLOWED_URL_SCHEMES:
            safe_text = escape(link_text_raw, quote=True)
            safe_href = escape(link_href_raw, quote=True)

            anchor = (
                f'<a href="{safe_href}" rel="noopener noreferrer">'
                f"{safe_text}</a>"
            )
            out_parts.append(anchor)
        else:
            # Not a safe URL: escape the literal Markdown instead of linking.
            out_parts.append(escape(match.group(0), quote=True))

        idx = match.end()

    # Append any trailing text after the last match.
    if idx < len(md_text):
        out_parts.append(escape(md_text[idx:], quote=True))

    return "".join(out_parts)


def _format_discussion_points(points: List[str]) -> str:
    """
    Format discussion points into a sanitized HTML unordered list.

    This is a pure function: given the same input list of strings, it always
    returns the same output and produces no side effects.

    Each point is rendered with a minimal subset of inline Markdown supported
    via _safe_render_markdown_inline, and all content is escaped as needed.

    Args:
        points: A list of strings representing discussion points in Markdown.

    Returns:
        A string containing a <ul> with <li> entries for each point, or an
        empty string if no points are provided.
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

    The resulting HTML includes a header (<h2>), a message body (<p>), and
    an unordered list (<ul>) of discussion points. The title and body are
    fully HTML-escaped. Discussion points support a minimal subset of inline
    Markdown (links) subject to safe URL schemes.

    Args:
        topic: The title or main topic of the forum post.
        text_body: The main body or content of the forum post.
        points: A list of discussion points in Markdown format.

    Returns:
        A string containing the formatted HTML for the forum post.

    Raises:
        TypeError: If argument types do not match the expected types.
    """
    if not isinstance(topic, str):
        raise TypeError("topic must be a string")

    if not isinstance(text_body, str):
        raise TypeError("text_body must be a string")

    if not isinstance(points, list) or not all(isinstance(p, str) for p in points):
        raise TypeError("points must be a list of strings")

    # Sanitize header and message.
    safe_header = escape(topic, quote=True)
    safe_message = escape(text_body, quote=True)

    # Render discussion points as a sanitized unordered list.
    options_html = _format_discussion_points(points)

    return FORUM_POST_FORMAT.format(
        header=safe_header,
        message=safe_message,
        options=options_html,
    )
