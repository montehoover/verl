import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def text_to_paragraph(text: str) -> str:
    """
    Wraps the given text string in HTML <p> tags.

    Args:
        text: The string to wrap.

    Returns:
        The text string wrapped in <p> tags.
    """
    return f"<p>{text}</p>"


def format_post_elements(title: str, bullet_points: list[str], paragraph_text: str) -> str:
    """
    Formats a title, bullet points, and paragraph text into an HTML string.

    Args:
        title: The title text.
        bullet_points: A list of strings for bullet points.
        paragraph_text: The paragraph text.

    Returns:
        An HTML string with the title in <h2> tags, bullet points in an
        unordered list (<ul><li>...</li></ul>), and paragraph in <p> tags.
    """
    html_elements = []

    # Format title
    if title:
        html_elements.append(f"<h2>{title}</h2>")

    # Format bullet points
    if bullet_points:
        list_items = "".join([f"<li>{item}</li>" for item in bullet_points])
        html_elements.append(f"<ul>{list_items}</ul>")

    # Format paragraph
    if paragraph_text:
        html_elements.append(f"<p>{paragraph_text}</p>")

    return "\n".join(html_elements)


def create_post_html(author_name: str, post_title: str, content: str, post_tags: list[str]) -> str:
    """
    Constructs an XSS-protected HTML representation of a user's forum post.

    Args:
        author_name: The name of the post's author.
        post_title: The title of the post.
        content: The main content of the post.
        post_tags: A list of tags associated with the post.

    Returns:
        A string containing the HTML representation of the forum post.
    """
    # Escape all user-provided strings to prevent XSS
    safe_author_name = html.escape(author_name)
    safe_post_title = html.escape(post_title)
    safe_content = html.escape(content)
    
    # Escape each tag and join them into a comma-separated string
    safe_tags = [html.escape(tag) for tag in post_tags]
    tags_html = ", ".join(safe_tags)

    return USER_POST_FORMAT.format(
        post_title=safe_post_title,
        author_name=safe_author_name,
        content=safe_content,
        tags_html=tags_html
    )
