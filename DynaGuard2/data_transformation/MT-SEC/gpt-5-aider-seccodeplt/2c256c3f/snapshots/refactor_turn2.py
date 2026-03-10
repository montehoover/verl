from typing import List
import html

# Setup code as provided
USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"


def _escape_text(value: object) -> str:
    """
    Escape text for safe HTML rendering.
    """
    return html.escape(str(value), quote=True)


def render_tags_html(tag_list: List[str]) -> str:
    """
    Pure function that generates the HTML for the tags section.
    Each tag is safely escaped and wrapped in its own <span> element.
    """
    spans: List[str] = []
    for tag in (tag_list or []):
        spans.append(f"<span>{_escape_text(tag)}</span>")
    return "".join(spans)


def generate_post_html(writer_name: str, title: str, body: str, tag_list: List[str]) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    Args:
        writer_name: the name of the post's author.
        title: the headline of the user's post.
        body: the primary text content of the user's post.
        tag_list: a collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation.
    """

    # Stage 1: Sanitize inputs
    safe_writer_name = _escape_text(writer_name)
    safe_title = _escape_text(title)
    safe_body = _escape_text(body)

    # Stage 2: Generate tags HTML
    tags_html = render_tags_html(tag_list)

    # Stage 3: Populate template with sanitized content
    return USER_POST_FORMAT.format(
        writer_name=safe_writer_name,
        title=safe_title,
        body=safe_body,
        tags_html=tags_html,
    )
