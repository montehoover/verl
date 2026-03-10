import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def render_forum_post(author_name: str, title: str, body_content: str, tag_list: list[str]) -> str:
    """
    Renders a forum post to a secure HTML string.

    Args:
        author_name: The name of the post's author.
        title: The title of the post.
        body_content: The main content of the post.
        tag_list: A list of tags associated with the post.

    Returns:
        A string containing the secure HTML representation of the post.
    """
    # Escape all user-provided input to prevent XSS
    safe_title = html.escape(title)
    safe_author_name = html.escape(author_name)
    safe_body_content = html.escape(body_content)

    # Generate HTML for tags, escaping each tag
    safe_tags_html = "".join(f"<span>{html.escape(tag)}</span> " for tag in tag_list).strip()

    # Format the post using the predefined template
    post_html = USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author_name,
        body_content=safe_body_content,
        tags_html=safe_tags_html
    )
    return post_html
