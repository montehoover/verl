import logging
from html import escape
from typing import List

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

logger = logging.getLogger(__name__)

def _sanitize_log_field(value: str, max_len: int = 128) -> str:
    """
    Produce a safe, single-line, truncated representation of a value for logs.
    Removes control characters and limits length to avoid log injection or leakage.
    """
    s = str(value) if value is not None else ""
    # Remove control characters and newlines
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = "".join(ch for ch in s if ch.isprintable())
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s

def build_label_html(labels: List[str]) -> str:
    """
    Construct XSS-safe HTML for labels, wrapping each in a <span> element.

    Args:
        labels: List of label strings.

    Returns:
        A string with each label wrapped in a <span>, separated by spaces.
    """
    if not labels:
        return ""
    safe_spans = [f"<span>{escape(str(label), quote=True)}</span>" for label in labels]
    return " ".join(safe_spans)

def assemble_post_html(author_id: str, post_heading: str, content: str, label_html: str) -> str:
    """
    Assemble the full post HTML using the global template.

    Args:
        author_id: XSS-safe author identifier.
        post_heading: XSS-safe post heading.
        content: XSS-safe post content.
        label_html: Pre-rendered (XSS-safe) label HTML.

    Returns:
        The assembled HTML string.
    """
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        author_id=author_id,
        content=content,
        label_html=label_html,
    )

def render_blog_post(author_id: str, post_heading: str, content: str, labels: List[str]) -> str:
    """
    Generate a secure HTML representation of a user's forum post.
    Logs each render attempt with minimal, sanitized context.

    Args:
        author_id: The identifier of the post's creator.
        post_heading: The heading of the user's post.
        content: The primary content of the user's post.
        labels: A sequence of labels associated with the post.

    Returns:
        A string representing the XSS-protected HTML version of the user's forum post.
    """
    # Prepare safe values for logging (avoid sensitive data leakage)
    log_author = _sanitize_log_field(author_id, max_len=64)
    log_heading = _sanitize_log_field(post_heading, max_len=120)
    content_length = len(str(content) if content is not None else "")
    labels_count = len(labels or [])

    logger.info(
        "Render attempt: author_id=%s, post_heading=%s, content_length=%d, labels_count=%d",
        log_author,
        log_heading,
        content_length,
        labels_count,
    )

    try:
        # Escape core fields for HTML context (XSS protection)
        safe_author = escape("" if author_id is None else str(author_id), quote=True)
        safe_heading = escape("" if post_heading is None else str(post_heading), quote=True)
        safe_content = escape("" if content is None else str(content), quote=True)

        # Build labels HTML (function handles escaping for labels)
        label_html = build_label_html(labels or [])

        # Assemble final HTML
        result = assemble_post_html(
            author_id=safe_author,
            post_heading=safe_heading,
            content=safe_content,
            label_html=label_html,
        )

        logger.debug(
            "Render success: author_id=%s, post_heading=%s",
            log_author,
            log_heading,
        )
        return result
    except Exception as exc:
        # Log the failure with sanitized context; re-raise to surface the error upstream
        logger.exception(
            "Render failed: author_id=%s, post_heading=%s, error=%s",
            log_author,
            log_heading,
            _sanitize_log_field(repr(exc), max_len=200),
        )
        raise
