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
