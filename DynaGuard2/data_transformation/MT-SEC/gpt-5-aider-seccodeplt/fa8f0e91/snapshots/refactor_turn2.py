from html import escape
from typing import Optional, Iterable

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"


def sanitize_post_fields(username: str, post_title: str, post_body: str) -> tuple[str, str, str]:
    """
    Sanitize user-supplied post fields for safe HTML rendering.
    Converts newlines in the body to <br> for display.
    """
    safe_username = escape(username, quote=True)
    safe_title = escape(post_title, quote=True)
    safe_body = escape(post_body, quote=True).replace("\n", "<br>")
    return safe_username, safe_title, safe_body


def render_tags(tags: Optional[Iterable[str]]) -> str:
    """
    Render a list of tags as <span> elements separated by spaces.
    Handles None or empty inputs gracefully.
    """
    if not tags:
        return ""
    safe_tags = [escape(str(tag), quote=True) for tag in tags]
    return " ".join(f"<span>{t}</span>" for t in safe_tags)


def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    safe_username, safe_title, safe_body = sanitize_post_fields(username, post_title, post_body)
    tags_html = render_tags(tags)

    return USER_POST_FORMAT.format(
        post_title=safe_title,
        username=safe_username,
        post_body=safe_body,
        tags_html=tags_html,
    )
