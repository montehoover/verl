def format_user_message_with_title(username: str, title: str, message: str) -> str:
    """
    Formats a user's name, message title, and message for display.

    Args:
        username: The username of the sender.
        title: The title of the message.
        message: The message content.

    Returns:
        A string formatted as 'Title: <title>, User: <username>, Message: <message>'.
    """
    return f"Title: {title}, User: {username}, Message: {message}"


import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Renders a user post as a sanitized HTML string.

    Args:
        username: The username of the poster.
        post_title: The title of the post.
        post_body: The body content of the post.
        tags: A list of tags associated with the post.

    Returns:
        A sanitized HTML string representing the user post.
    """
    sanitized_username = html.escape(username)
    sanitized_post_title = html.escape(post_title)
    sanitized_post_body = html.escape(post_body)

    sanitized_tags = [html.escape(tag) for tag in tags]
    tags_html = " ".join(f"<span>{tag}</span>" for tag in sanitized_tags)

    return USER_POST_FORMAT.format(
        post_title=sanitized_post_title,
        username=sanitized_username,
        post_body=sanitized_post_body,
        tags_html=tags_html
    )
