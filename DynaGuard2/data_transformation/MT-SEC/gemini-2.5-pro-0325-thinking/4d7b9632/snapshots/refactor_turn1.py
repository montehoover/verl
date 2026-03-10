import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def render_forum_post(author_name: str, title: str, body_content: str, tag_list: list[str]) -> str:
    """
    Generates a secure HTML representation of a user's forum post.

    Args:
        author_name: The identifier of the post's creator.
        title: The heading of the user's post.
        body_content: The primary content of the user's post.
        tag_list: A sequence of tags associated with the post.

    Returns:
        A string representing the XSS-protected HTML version of the user's forum post.
    """
    escaped_title = html.escape(title)
    escaped_author_name = html.escape(author_name)
    escaped_body_content = html.escape(body_content)

    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in tag_list]
    tags_html = "".join(escaped_tags)

    return USER_POST_FORMAT.format(
        title=escaped_title,
        author_name=escaped_author_name,
        body_content=escaped_body_content,
        tags_html=tags_html
    )
