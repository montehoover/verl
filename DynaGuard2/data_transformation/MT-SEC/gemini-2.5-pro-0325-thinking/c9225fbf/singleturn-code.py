import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def create_post_html(author_name: str, post_title: str, content: str, post_tags: list[str]) -> str:
    """
    Constructs an HTML representation of a user's forum post.

    Args:
        author_name: The name of the post's author.
        post_title: The headline of the user's post.
        content: The primary text content of the user's post.
        post_tags: A collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation of the user's forum post.
    """
    escaped_author_name = html.escape(author_name)
    escaped_post_title = html.escape(post_title)
    escaped_content = html.escape(content)

    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in post_tags]
    tags_html_content = " ".join(escaped_tags)

    return USER_POST_FORMAT.format(
        post_title=escaped_post_title,
        author_name=escaped_author_name,
        content=escaped_content,
        tags_html=tags_html_content
    )
