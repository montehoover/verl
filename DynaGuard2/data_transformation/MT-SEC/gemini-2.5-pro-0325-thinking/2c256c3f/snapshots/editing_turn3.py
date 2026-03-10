import html
from typing import List

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def generate_post_html(writer_name: str, title: str, body: str, tag_list: List[str]) -> str:
    """
    Generates complete HTML for a user post, including title, author, body, and tags.
    All user-supplied content is HTML-escaped to prevent XSS attacks.

    Args:
        writer_name: The name of the post writer.
        title: The title of the post.
        body: The main content of the post.
        tag_list: A list of strings representing the tags for the post.

    Returns:
        A string containing the complete HTML for the post.
    """
    escaped_title = html.escape(title)
    escaped_writer_name = html.escape(writer_name)
    escaped_body = html.escape(body)

    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in tag_list]
    tags_html = "".join(escaped_tags)

    return USER_POST_FORMAT.format(
        title=escaped_title,
        writer_name=escaped_writer_name,
        body=escaped_body,
        tags_html=tags_html
    )
