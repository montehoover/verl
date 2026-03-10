import re
import html

FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"


def _escape_html(text: str) -> str:
    return html.escape(text, quote=True)


def _render_markdown_safe(text: str) -> str:
    """
    Render a very small, safe subset of Markdown for inline content:
    - Links: [text](url) with allowed protocols (http, https, mailto) or relative/fragment
    - Inline code: `code`
    - Bold: **bold**
    - Italic: *italic*
    All user-supplied content is HTML-escaped first.
    """
    escaped = _escape_html(text)

    # Links: [text](url)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    def _link_repl(m: re.Match) -> str:
        link_text = m.group(1)  # already escaped by earlier step
        url = m.group(2).strip()  # may contain & which is fine as &amp; after escaping
        # Allow http, https, mailto, relative paths, fragments, and query-only
        if re.match(r"^(https?://|mailto:|/|\.{1,2}/|#|\?)", url, re.IGNORECASE):
            return f'<a href="{url}" rel="nofollow noopener">{link_text}</a>'
        # Unsafe URL scheme: drop the link, keep the text
        return link_text

    escaped = link_pattern.sub(_link_repl, escaped)

    # Inline code: `code`
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)

    # Bold: **bold**
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

    # Italic: *italic* (avoid matching the ** pairs)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)

    return escaped


def _build_unordered_list(items: list[str]) -> str:
    """
    Pure function that takes a list of strings (assumed already sanitized/rendered)
    and returns an HTML unordered list.
    """
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def _render_options_list(options_list: list[str]) -> str:
    sanitized_items = []
    for option in options_list:
        safe_html = _render_markdown_safe(option if option is not None else "")
        sanitized_items.append(safe_html)
    return _build_unordered_list(sanitized_items)


def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """
    Construct a sanitized, formatted forum post consisting of:
    - Header (title)
    - Message body
    - Options rendered as an HTML unordered list (<ul>)

    Args:
        title: the title or subject line of the forum post
        post_body: the main content or body of the forum post
        options_list: a collection of user-provided options or discussion points in markdown format

    Returns:
        A string containing the sanitized and formatted forum post, including the header,
        message, and safely rendered options.
    """
    safe_title = _escape_html(title if title is not None else "")
    safe_body = _escape_html(post_body if post_body is not None else "").replace("\n", "<br/>")
    options_html = _render_options_list(options_list or [])

    return FORUM_POST_FORMAT.format(
        title=safe_title,
        post_body=safe_body,
        options_list=options_html,
    )
