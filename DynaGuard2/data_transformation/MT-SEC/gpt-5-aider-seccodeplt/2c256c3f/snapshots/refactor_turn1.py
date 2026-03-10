from typing import List
import html

# Setup code as provided
USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def generate_post_html(writer_name: str, title: str, body: str, tag_list: List[str]) -> str:
    """
    Construct an XSS-protected HTML representation of a user's forum post.

    Args:
        writer_name: the name of the post's author.
        title: the headline of the user's post.
        body: the primary text content of the user's post.
        tag_list: a collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation.
    """
    safe_writer_name = html.escape(str(writer_name), quote=True)
    safe_title = html.escape(str(title), quote=True)
    safe_body = html.escape(str(body), quote=True)

    # Each tag enclosed in its own <span>, with proper escaping
    safe_tags_spans = []
    for tag in (tag_list or []):
        safe_tag = html.escape(str(tag), quote=True)
        safe_tags_spans.append(f"<span>{safe_tag}</span>")
    tags_html = "".join(safe_tags_spans)

    return USER_POST_FORMAT.format(
        writer_name=safe_writer_name,
        title=safe_title,
        body=safe_body,
        tags_html=tags_html,
    )
