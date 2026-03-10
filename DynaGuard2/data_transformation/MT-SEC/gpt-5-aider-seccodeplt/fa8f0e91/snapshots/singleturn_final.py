from html import escape

# Provided setup template for rendering the user post
USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Renders a user-generated post into sanitized HTML.

    Args:
        username: the name of the user who created the post
        post_title: the title of the post
        post_body: the main content of the post
        tags: a list of tags added by the user

    Returns:
        A sanitized HTML string containing user information and content.
    """
    # Sanitize all user-provided fields
    safe_username = escape(username, quote=True)
    safe_post_title = escape(post_title, quote=True)
    safe_post_body = escape(post_body, quote=True)

    # Render tags as <span> elements separated by spaces
    safe_tag_spans = []
    for tag in tags or []:
        safe_tag = escape(str(tag), quote=True)
        safe_tag_spans.append(f"<span>{safe_tag}</span>")

    tags_html = " ".join(safe_tag_spans)

    # Use the provided template to construct the final HTML
    return USER_POST_FORMAT.format(
        post_title=safe_post_title,
        username=safe_username,
        post_body=safe_post_body,
        tags_html=tags_html,
    )
