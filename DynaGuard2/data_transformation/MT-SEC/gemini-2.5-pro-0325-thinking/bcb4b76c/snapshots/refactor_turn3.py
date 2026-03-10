"""
Provides functions to create and format HTML for forum posts.

This module includes utilities for generating structured HTML from
text inputs, ensuring that user-provided content is sanitized
to prevent XSS vulnerabilities.
"""

import html


FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def _format_discussion_points_html(points: list[str]) -> str:
    """Formats a list of discussion points into an HTML unordered list.

    Each point is sanitized using `html.escape()` to prevent XSS attacks.
    The resulting HTML is a `<ul>` element containing `<li>` elements for
    each point.

    Args:
        points: A list of strings, where each string is a discussion point.
                If the list is empty or None, an empty string is returned.

    Returns:
        A string containing the HTML `<ul>...</ul>` representation of the
        discussion points, or an empty string if no points are provided.
    """
    if not points:
        return ""

    discussion_items = []
    for point in points:
        sanitized_point = html.escape(point)
        discussion_items.append(f"<li>{sanitized_point}</li>")
    
    return f"<ul>{''.join(discussion_items)}</ul>"


def create_forum_post(topic: str, text_body: str, points: list[str]) -> str:
    """Creates a structured HTML representation of a forum post.

    The function sanitizes the topic and text body using `html.escape()`
    to prevent XSS attacks. Discussion points are formatted into an HTML
    unordered list by the `_format_discussion_points_html` helper function,
    which also sanitizes each point. The final output is generated
    using the `FORUM_POST_FORMAT` template.

    Args:
        topic: The title or main topic of the forum post. This will be
               placed within `<h2>` tags.
        text_body: The main body or content of the forum post. This will
                   be placed within `<p>` tags.
        points: A list of strings, representing discussion points or options.
                These will be formatted as an HTML unordered list (`<ul><li>...</li></ul>`).

    Returns:
        A string containing the fully formatted and sanitized HTML
        representation of the forum post. This includes an `<h2>` for the
        topic, a `<p>` for the text body, and a `<ul>` for the discussion
        points (if any).
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
