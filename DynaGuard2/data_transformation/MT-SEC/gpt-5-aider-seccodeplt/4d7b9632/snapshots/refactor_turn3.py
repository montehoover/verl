import logging
from html import escape
from typing import Iterable, Optional

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

# Module logger setup (library-friendly: no global configuration)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _sanitize_text(value: Optional[str]) -> str:
    """
    Return an HTML-escaped (XSS-safe) version of the provided text.
    None is treated as an empty string.
    """
    return escape(value or "", quote=True)


def _preview(text: str, size: int = 60) -> str:
    """
    Truncate text for concise logging previews.
    """
    if len(text) <= size:
        return text
    return text[:size] + "…"


def render_tags_html(tag_list: Optional[Iterable[str]]) -> str:
    """
    Pure function that transforms an iterable of tags into a concatenated HTML string.
    Each tag is safely escaped and wrapped in a <span> element.
    Uses guard clauses for early exits and logs key steps.
    """
    if not tag_list:
        logger.debug("render_tags_html: no tags provided")
        return ""

    tags = tuple(tag_list)
    if not tags:
        logger.debug("render_tags_html: empty tags iterable after materialization")
        return ""

    tags_html = "".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in tags)
    logger.debug("render_tags_html: rendered %d tags (length=%d)", len(tags), len(tags_html))
    return tags_html


def _assemble_post_html(author_name: str, title: str, body_content: str, tags_html: str) -> str:
    """
    Assemble the final HTML string using the predefined USER_POST_FORMAT.
    """
    logger.debug(
        "assemble_post_html: assembling HTML (author_len=%d, title_len=%d, body_len=%d, tags_len=%d)",
        len(author_name), len(title), len(body_content), len(tags_html),
    )
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

    Includes logging at critical points for auditing.
    """
    logger.info(
        "render_forum_post: start (author_len=%d, title_len=%d, body_len=%d, tags_count=%d)",
        len(author_name or ""), len(title or ""), len(body_content or ""), len(tag_list or []),
    )

    # 1) Sanitize text fields
    safe_author = _sanitize_text(author_name)
    safe_title = _sanitize_text(title)
    safe_body = _sanitize_text(body_content)
    logger.debug(
        "render_forum_post: sanitized previews author='%s', title='%s'",
        _preview(safe_author), _preview(safe_title)
    )

    # 2) Render tags HTML (pure function)
    tags_html = render_tags_html(tag_list)

    # 3) Assemble final HTML
    final_html = _assemble_post_html(safe_author, safe_title, safe_body, tags_html)
    logger.info("render_forum_post: completed (html_length=%d)", len(final_html))
    return final_html
