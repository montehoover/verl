from html import escape
from typing import Iterable, Optional

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"


def _sanitize_text(value: Optional[str]) -> str:
    """
    Return an HTML-escaped (XSS-safe) version of the provided text.
    None is treated as an empty string.
    """
    return escape(value or "", quote=True)


def render_tags_html(tag_list: Optional[Iterable[str]]) -> str:
    """
    Pure function that transforms an iterable of tags into a concatenated HTML string.
    Each tag is safely escaped and wrapped in a <span> element.
    """
    if not tag_list:
        return ""
    return "".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in tag_list)


def _assemble_post_html(author_name: str, title: str, body_content: str, tags_html: str) -> str:
    """
    Assemble the final HTML string using the predefined USER_POST_FORMAT.
    """
    return USER_POST_FORMAT.format(
        title=title,
        author_name=author_name,
        body_content=body_content,
        tags_html=tags_html,
    )


def render_forum_post(author_name: str, title: str, body_content: str, tag_list: list[str]) -> str:
    """
    Orchestrate the rendering pipeline for a forum post:
    1) Sanitize text fields
    2) Render tags HTML
    3) Assemble final HTML using the template
    """
    # 1) Sanitize text fields
    safe_author = _sanitize_text(author_name)
    safe_title = _sanitize_text(title)
    safe_body = _sanitize_text(body_content)

    # 2) Render tags HTML (pure function)
    tags_html = render_tags_html(tag_list)

    # 3) Assemble final HTML
    return _assemble_post_html(safe_author, safe_title, safe_body, tags_html)
