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
