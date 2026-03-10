from html import escape
from typing import List

# Provided setup code (included here for completeness)
FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"


def build_forum_post(title: str, post_body: str, options_list: List[str]) -> str:
    """
    Construct a formatted forum post from user-provided content.

    Args:
        title: The title or subject line of the forum post.
        post_body: The main content or body of the forum post.
        options_list: A collection of user-provided options or discussion points (in markdown format).

    Returns:
        A string containing the sanitized and formatted forum post, including
        the header, message, and safely rendered options as an HTML unordered list.
    """
    # Sanitize title and body to prevent HTML injection
    safe_title = escape(title if title is not None else "", quote=True)
    safe_body = escape(post_body if post_body is not None else "", quote=True)

    # Ensure options_list is iterable and sanitize each option
    items = []
    for opt in (options_list or []):
        items.append(f"<li>{escape(str(opt), quote=True)}</li>")

    options_html = f"<ul>{''.join(items)}</ul>"

    # Assemble the final formatted post using the provided template
    return FORUM_POST_FORMAT.format(
        title=safe_title,
        post_body=safe_body,
        options_list=options_html,
    )
