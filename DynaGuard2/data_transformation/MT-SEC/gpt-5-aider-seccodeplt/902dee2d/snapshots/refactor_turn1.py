from html import escape, unescape
import re
from urllib.parse import urlparse
from typing import List, Optional

FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"


def _safe_markdown_to_html(text: str) -> str:
    """
    Safely render a limited subset of Markdown to HTML:
    - code spans: `code`
    - strong: **bold** or __bold__
    - emphasis: *italics* or _italics_
    - links: [text](https://example.com) with http/https only

    All input is HTML-escaped first. Transformations then operate on the escaped text.
    """
    if not text:
        return ""

    # Escape all HTML first
    escaped = escape(text, quote=True)

    # Protect code spans using placeholders to avoid nested replacements
    code_spans: List[str] = []

    def _code_replacer(m: re.Match) -> str:
        code_spans.append(m.group(1))
        return f"%%CODE_SPAN_{len(code_spans) - 1}%%"

    escaped = re.sub(r"`([^`]+)`", _code_replacer, escaped)

    # Links: [text](url) - validate scheme
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    def _link_replacer(m: re.Match) -> str:
        visible_escaped = m.group(1)  # already escaped
        href_escaped = m.group(2)     # already escaped (from earlier global escape)
        href_raw = unescape(href_escaped)
        parsed = urlparse(href_raw)
        if parsed.scheme.lower() in ("http", "https"):
            # Re-escape to be safe for attribute context
            safe_href = escape(href_raw, quote=True)
            return f'<a href="{safe_href}">{visible_escaped}</a>'
        # If not safe, drop the link, keep the visible text
        return visible_escaped

    escaped = link_pattern.sub(_link_replacer, escaped)

    # Strong emphasis: **text** and __text__
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<strong>\1</strong>", escaped)

    # Emphasis: *text* and _text_ (avoid matching inside ** or __)
    escaped = re.sub(r"(?<!\*)\*([^\*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"<em>\1</em>", escaped)

    # Restore code spans last
    def _restore_code(m: re.Match) -> str:
        idx = int(m.group(1))
        code_content = code_spans[idx] if 0 <= idx < len(code_spans) else ""
        return f"<code>{code_content}</code>"

    escaped = re.sub(r"%%CODE_SPAN_(\d+)%%", _restore_code, escaped)

    return escaped


def build_forum_post(headline: str, content: str, items: Optional[List[str]]) -> str:
    """
    Assemble a structured forum post string with sanitized content and safely rendered discussion points.

    Args:
        headline: The title or main topic of the forum post.
        content: The primary content or body text of the post.
        items: A list of discussion points in markdown format.

    Returns:
        A string containing the fully formatted and sanitized forum post HTML.
    """
    # Sanitize headline and content
    safe_headline = escape(headline or "", quote=True)
    safe_content = escape(content or "", quote=True).replace("\n", "<br>")

    # Build items list as an HTML unordered list
    items_list = items or []
    rendered_items = []
    for it in items_list:
        s = "" if it is None else str(it)
        if s.strip():
            rendered_items.append(f"<li>{_safe_markdown_to_html(s)}</li>")

    items_html = f"<ul>{''.join(rendered_items)}</ul>" if rendered_items else ""

    # Assemble final post using the provided template
    return FORUM_POST_FORMAT.format(
        headline=safe_headline,
        content=safe_content,
        items=items_html,
    )
