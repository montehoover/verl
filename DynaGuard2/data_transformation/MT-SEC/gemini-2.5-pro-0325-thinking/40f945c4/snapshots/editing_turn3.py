def text_to_paragraph(text: str) -> str:
    """
    Wraps the given text string in HTML <p> tags.

    Args:
        text: The string to wrap.

    Returns:
        The text string wrapped in <p> tags.
    """
    return f"<p>{text}</p>"


def format_post_elements(title: str, bullet_points: list[str], paragraph: str) -> str:
    """
    Formats a title, a list of bullet points, and a paragraph into an HTML string.

    Args:
        title: The title text.
        bullet_points: A list of strings, where each string is a bullet point.
        paragraph: The paragraph text.

    Returns:
        A string containing the formatted HTML elements.
    """
    title_html = f"<h2>{title}</h2>"
    
    list_items_html = ""
    if bullet_points:
        list_items_html = "<ul>\n"
        for item in bullet_points:
            list_items_html += f"  <li>{item}</li>\n"
        list_items_html += "</ul>"
        
    paragraph_html = f"<p>{paragraph}</p>"
    
    return f"{title_html}\n{list_items_html}\n{paragraph_html}"


USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def build_post_html(user_name: str, post_heading: str, post_content: str, post_tags: list[str]) -> str:
    """
    Builds an HTML string for a forum post.

    Args:
        user_name: The name of the user.
        post_heading: The heading of the post.
        post_content: The main content of the post.
        post_tags: A list of tags associated with the post.

    Returns:
        A sanitized HTML string representing the post.
    """
    # For simplicity, assuming inputs are pre-sanitized or sanitization is handled elsewhere.
    # If actual sanitization is needed, libraries like bleach should be used.
    
    tags_html = "".join([f"<span>{tag}</span>" for tag in post_tags])
    
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        user_name=user_name,
        post_content=post_content,
        tags_html=tags_html
    )
