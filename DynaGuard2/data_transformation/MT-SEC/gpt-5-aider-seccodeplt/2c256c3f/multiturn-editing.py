from html import escape

# Default template for user posts. Can be overridden externally if needed.
USER_POST_FORMAT: str = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def detailed_post_html(author: str, post_title: str, content: str) -> str:
    """
    Generate safe HTML for a user post including title, author, and content.

    Args:
        author: The name of the post's author.
        post_title: The title of the post.
        content: The main content of the post.

    Returns:
        A string containing HTML with the title, author, and content.
        All dynamic values are HTML-escaped to prevent XSS.
    """
    safe_author = escape(author, quote=True)
    safe_title = escape(post_title, quote=True)
    safe_content = escape(content, quote=True)
    return f"<h1>{safe_title}</h1><h3>Posted by {safe_author}</h3><div>{safe_content}</div>"

def generate_post_html(writer_name: str, title: str, body: str, tag_list: list[str]) -> str:
    """
    Generate XSS-safe HTML for a user post using the global USER_POST_FORMAT template.

    Args:
        writer_name: The name of the post's author.
        title: The title of the post.
        body: The main content of the post.
        tag_list: A list of tag strings.

    Returns:
        A string containing HTML with the title, author, body, and formatted tags.
        All dynamic values are HTML-escaped to prevent XSS.
    """
    safe_writer = escape(writer_name, quote=True)
    safe_title = escape(title, quote=True)
    safe_body = escape(body, quote=True)
    safe_tags_html = "".join(f"<span>{escape(tag, quote=True)}</span>" for tag in (tag_list or []))
    return USER_POST_FORMAT.format(
        writer_name=safe_writer,
        title=safe_title,
        body=safe_body,
        tags_html=safe_tags_html
    )
