from html import escape

def format_user_message_with_title(username: str, title: str, message: str) -> str:
    """
    Format a title, user's name, and message for display.

    Args:
        username (str): The user's name.
        title (str): The message title.
        message (str): The user's message.

    Returns:
        str: A formatted string like 'Title: <title>, User: <username>, Message: <message>'.
    """
    return f"Title: {title}, User: {username}, Message: {message}"


USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Render a user's post as a sanitized HTML string.

    Args:
        username (str): The user's name.
        post_title (str): The post's title.
        post_body (str): The post's content/body.
        tags (list[str]): A list of tags associated with the post.

    Returns:
        str: A sanitized HTML string formatted according to USER_POST_FORMAT.
    """
    safe_username = escape(username, quote=True)
    safe_post_title = escape(post_title, quote=True)
    safe_post_body = escape(post_body, quote=True)
    safe_tags_html = " ".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in (tags or []))

    return USER_POST_FORMAT.format(
        post_title=safe_post_title,
        username=safe_username,
        post_body=safe_post_body,
        tags_html=safe_tags_html,
    )
