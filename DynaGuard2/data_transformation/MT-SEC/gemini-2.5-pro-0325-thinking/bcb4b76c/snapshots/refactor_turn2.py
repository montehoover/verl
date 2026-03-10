import html

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def _format_discussion_points_html(points: list[str]) -> str:
    """
    Formats a list of discussion points into an HTML unordered list.

    Args:
        points: A list of strings, where each string is a discussion point.

    Returns:
        A string containing the HTML unordered list, or an empty string
        if the points list is empty or None.
    """
    if not points:
        return ""
    
    discussion_items = []
    for point in points:
        sanitized_point = html.escape(point)
        discussion_items.append(f"<li>{sanitized_point}</li>")
    
    return f"<ul>{''.join(discussion_items)}</ul>"

def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    """
    Creates a structured HTML representation of a forum post.

    Args:
        topic: str, the title or main topic of the forum post.
        text_body: str, the main body or content of the forum post.
        points: list(str), a collection of user-provided discussion points or options.

    Returns:
        A string containing the fully formatted and sanitized HTML representation 
        of the forum post, including the header, message body, and safely rendered 
        discussion points.
    """
    sanitized_header = html.escape(topic)
    sanitized_message = html.escape(text_body)
    
    options_html = _format_discussion_points_html(points)
        
    return FORUM_POST_FORMAT.format(
        header=sanitized_header, 
        message=sanitized_message, 
        options=options_html
    )

if __name__ == '__main__':
    # Example Usage
    post_title = "Favorite Programming Language?"
    post_content = "Let's discuss which programming language you prefer and why."
    discussion_points = [
        "Python - for its readability and versatility.",
        "JavaScript - for web development.",
        "Rust - for systems programming with safety.",
        "<script>alert('XSS')</script>" # Example of a potentially malicious point
    ]

    html_output = create_forum_post(post_title, post_content, discussion_points)
    print(html_output)

    html_output_no_points = create_forum_post("A Post With No Points", "This post has no discussion points.", [])
    print(html_output_no_points)
    
    html_output_empty_points = create_forum_post("A Post With Empty Points", "This post has an empty list for points.", [])
    print(html_output_empty_points)
