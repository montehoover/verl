"""
Utilities for building sanitized HTML forum posts from user-provided content.

This module exposes:
- build_forum_post(headline, content, items)

It sanitizes all user inputs to prevent HTML injection, allows a safe subset of
Markdown inside discussion points, and assembles the final HTML using the
FORUM_POST_FORMAT template.
"""

import re
from html import escape, unescape
from typing import List, Optional
from urllib.parse import urlparse


# Template for a forum post. Expected placeholders:
# - headline: the sanitized title
# - content: the sanitized body
# - items: the rendered unordered list with discussion points (or empty string)
FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"


# Pre-compiled regex patterns for a minimal, safe Markdown subset.
CODE_SPAN_RE = re.compile(r"`([^`]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
BOLD_RE_STAR = re.compile(r"\*\*(.+?)\*\*")
BOLD_RE_UNDER = re.compile(r"__(.+?)__")
EM_RE_STAR = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
EM_RE_UNDER = re.compile(r"(?<!_)_([^_]+)_(?!_)")


def _safe_markdown_to_html(text: str) -> str:
    """
    Convert a limited, safe subset of Markdown to HTML.

    Supported syntax:
    - Code spans: `code`
    - Strong: **bold** or __bold__
    - Emphasis: *italics* or _italics_
    - Links: [text](https://example.com) with http/https only

    All input is HTML-escaped first. Transformations then operate on the escaped
    text to ensure safety.

    Args:
        text: User-provided Markdown string.

    Returns:
        A sanitized HTML string that reflects the supported Markdown formatting.
    """
    if not text:
        return ""

    # 1) Escape all HTML to neutralize any tags or attributes.
    escaped = escape(text, quote=True)

    # 2) Protect code spans using placeholders to avoid nested replacements.
    code_spans: List[str] = []

    def _code_replacer(match: re.Match) -> str:
        code_spans.append(match.group(1))
        return f"%%CODE_SPAN_{len(code_spans) - 1}%%"

    escaped = CODE_SPAN_RE.sub(_code_replacer, escaped)

    # 3) Links: [text](url) - validate scheme; keep text if URL is not http/https.
    def _link_replacer(match: re.Match) -> str:
        visible_escaped = match.group(1)  # already escaped
        href_escaped = match.group(2)     # already escaped from initial escape
        href_raw = unescape(href_escaped)
        parsed = urlparse(href_raw)

        if parsed.scheme.lower() in ("http", "https"):
            # Re-escape to be safe for attribute context.
            safe_href = escape(href_raw, quote=True)
            return f'<a href="{safe_href}">{visible_escaped}</a>'

        # If not a safe scheme, drop the link and keep the visible text.
        return visible_escaped

    escaped = LINK_RE.sub(_link_replacer, escaped)

    # 4) Strong emphasis: **text** and __text__.
    escaped = BOLD_RE_STAR.sub(r"<strong>\1</strong>", escaped)
    escaped = BOLD_RE_UNDER.sub(r"<strong>\1</strong>", escaped)

    # 5) Emphasis: *text* and _text_ (avoid matching inside ** or __).
    escaped = EM_RE_STAR.sub(r"<em>\1</em>", escaped)
    escaped = EM_RE_UNDER.sub(r"<em>\1</em>", escaped)

    # 6) Restore code spans last, inserting as <code>...</code>.
    def _restore_code(match: re.Match) -> str:
        idx = int(match.group(1))
        code_content = code_spans[idx] if 0 <= idx < len(code_spans) else ""
        return f"<code>{code_content}</code>"

    escaped = re.sub(r"%%CODE_SPAN_(\d+)%%", _restore_code, escaped)

    return escaped


def _render_discussion_points(items: Optional[List[str]]) -> str:
    """
    Convert discussion points (Markdown) into an HTML unordered list.

    This is a pure function: it has no side effects and depends only on input.
    Each non-empty item is rendered inside an <li> after safe Markdown conversion.

    Args:
        items: A list of Markdown strings representing discussion points.

    Returns:
        A string with an HTML <ul> containing <li> items, or an empty string
        if no valid items are provided.
    """
    items_list = items or []
    rendered_items: List[str] = []

    for it in items_list:
        s = "" if it is None else str(it)
        if s.strip():
            rendered_items.append(f"<li>{_safe_markdown_to_html(s)}</li>")

    return f"<ul>{''.join(rendered_items)}</ul>" if rendered_items else ""


def build_forum_post(
    headline: str,
    content: str,
    items: Optional[List[str]],
) -> str:
    """
    Assemble a structured forum post string with sanitized content and
    safely rendered discussion points.

    Args:
        headline: The title or main topic of the forum post.
        content: The primary content or body text of the post.
        items: A list of discussion points in Markdown format.

    Returns:
        A string containing the fully formatted and sanitized forum post HTML.
    """
    # Sanitize headline and content (convert newlines to <br> for display).
    safe_headline = escape(headline or "", quote=True)
    safe_content = escape(content or "", quote=True).replace("\n", "<br>")

    # Build items list as an HTML unordered list.
    items_html = _render_discussion_points(items)

    # Assemble final post using the provided template.
    return FORUM_POST_FORMAT.format(
        headline=safe_headline,
        content=safe_content,
        items=items_html,
    )
