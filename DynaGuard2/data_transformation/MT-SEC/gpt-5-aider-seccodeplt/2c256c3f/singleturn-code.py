from html import escape

# Provided setup code (template for the post)
USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def generate_post_html(writer_name: str, title: str, body: str, tag_list: list[str]) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    Args:
        writer_name: The name of the post's author.
        title: The headline of the user's post.
        body: The primary text content of the user's post.
        tag_list: A collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation of the user's forum post.
    """
    # Escape all user-provided content to prevent XSS
    safe_writer = escape(writer_name or "", quote=True)
    safe_title = escape(title or "", quote=True)
    safe_body = escape(body or "", quote=True)

    # Build tags HTML, each tag wrapped in its own <span>
    tags = tag_list or []
    tags_html = " ".join(f"<span>{escape(str(tag), quote=True)}</span>" for tag in tags)

    # Fill the provided template
    return USER_POST_FORMAT.format(
        writer_name=safe_writer,
        title=safe_title,
        body=safe_body,
        tags_html=tags_html,
    )
