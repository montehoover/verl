import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def _generate_tags_html(tag_list: list[str]) -> str:
    """
    Generates an HTML string for a list of tags, with each tag escaped and wrapped in a <span>.
    """
    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in tag_list]
    return "".join(escaped_tags)

def generate_post_html(writer_name: str, title: str, body: str, tag_list: list[str]) -> str:
    """
    Constructs an HTML representation of a user's forum post.

    Args:
        writer_name: The name of the post's author.
        title: The headline of the user's post.
        body: The primary text content of the user's post.
        tag_list: A collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation of the user's forum post.
    """
    escaped_writer_name = html.escape(writer_name)
    escaped_title = html.escape(title)
    escaped_body = html.escape(body)

    # Generate HTML for tags using the helper function
    tags_html_content = _generate_tags_html(tag_list)

    # Format the final post HTML
    post_html = USER_POST_FORMAT.format(
        title=escaped_title,
        writer_name=escaped_writer_name,
        body=escaped_body,
        tags_html=tags_html_content
    )
    return post_html
