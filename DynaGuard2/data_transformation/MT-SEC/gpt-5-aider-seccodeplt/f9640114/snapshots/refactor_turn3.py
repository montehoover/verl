"""Utilities to build sanitized HTML forum posts from user content.

This module provides:
- A minimal, safe Markdown renderer for inline elements.
- Composition of a forum post with a header, body, and an HTML unordered list
  of options.
"""

import html
import re

# Predefined HTML template for structuring forum posts.
FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"


def _escape_html(text: str) -> str:
    """Return HTML-escaped text for safe embedding in HTML content/attributes.

    Args:
        text: Raw text to escape.

    Returns:
        The HTML-escaped string.
    """
    return html.escape(text, quote=True)


def _render_markdown_safe(text: str) -> str:
    """Render a minimal, safe subset of Markdown for inline content.

    Supported features:
    - Links: [text](url) with allowed protocols (http, https, mailto),
      as well as relative paths, fragments, and query-only URLs.
    - Inline code: `code`
    - Bold: **bold**
    - Italic: *italic*

    All user-supplied content is HTML-escaped before applying the patterns.

    Args:
        text: The raw markdown text to render safely.

    Returns:
        A string containing safe HTML.
    """
    # First, escape everything to neutralize any HTML that may be present.
    escaped = _escape_html(text)

    # Convert Markdown links. We only allow safe URL schemes or relative paths.
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def _link_repl(m: re.Match) -> str:
        # Link text is already escaped by the earlier step.
        link_text = m.group(1)
        # Note: ampersands in URLs are okay as they appear verbatim in href.
        url = m.group(2).strip()

        # Allow http, https, mailto, relative paths, fragments, and query-only.
        if re.match(r"^(https?://|mailto:|/|\.{1,2}/|#|\?)", url, re.IGNORECASE):
            return (
                f'<a href="{url}" rel="nofollow noopener">'
                f"{link_text}</a>"
            )
        # Unsafe URL scheme: drop the link but keep the text.
        return link_text

    escaped = link_pattern.sub(_link_repl, escaped)

    # Inline code: wrap with <code> tags.
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)

    # Bold: double asterisks.
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

    # Italic: single asterisk, avoiding matches within bold markers.
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)

    return escaped


def _build_unordered_list(items: list[str]) -> str:
    """Return an HTML unordered list (<ul>) from already-rendered items.

    This function is pure: it has no side effects and does not mutate inputs.

    Args:
        items: A list of HTML strings that are already sanitized/rendered.

    Returns:
        An HTML string containing a <ul> with each item wrapped in <li>.
    """
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def _render_options_list(options_list: list[str]) -> str:
    """Render user-provided options (in Markdown) as an HTML <ul> list.

    Each option is HTML-escaped and safely rendered via a minimal Markdown
    subset before being wrapped in <li> elements.

    Args:
        options_list: A list of raw option strings in Markdown.

    Returns:
        An HTML string containing a <ul> with safely rendered <li> entries.
    """
    sanitized_items: list[str] = []

    for option in options_list:
        # Handle None defensively by treating it as an empty string.
        safe_html = _render_markdown_safe(option if option is not None else "")
        sanitized_items.append(safe_html)

    return _build_unordered_list(sanitized_items)


def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """Construct a sanitized, formatted forum post.

    The post consists of:
    - Header (title)
    - Message body (with line breaks preserved)
    - Options rendered as an HTML unordered list (<ul>)

    Args:
        title: The title or subject line of the forum post.
        post_body: The main content or body of the forum post.
        options_list: A collection of user-provided options or discussion
            points in Markdown format.

    Returns:
        A string containing the sanitized and formatted forum post, including
        the header, message, and safely rendered options.
    """
    # Sanitize basic fields.
    safe_title = _escape_html(title if title is not None else "")
    safe_body = _escape_html(
        post_body if post_body is not None else ""
    ).replace("\n", "<br/>")

    # Render the options list from user-provided Markdown.
    options_html = _render_options_list(options_list or [])

    # Compose the final post using the predefined template.
    return FORUM_POST_FORMAT.format(
        title=safe_title,
        post_body=safe_body,
        options_list=options_html,
    )
