import html


# Global constant defining the HTML structure for a forum post.
# Placeholders {header}, {message}, and {options} will be replaced.
FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def _create_bullet_list_html(bullet_points: list[str]) -> str:
    """
    Creates an HTML unordered list from a list of strings.

    Each bullet point is sanitized to prevent HTML injection.

    Args:
        bullet_points: A list of strings to be included as list items.

    Returns:
        A string containing the HTML unordered list, with each item sanitized.
        Returns an empty string if the input list is empty.
    """
    if not bullet_points:
        return ""  # Return an empty string if there are no points.

    # Start the unordered list.
    options_html = "<ul>\n"
    # Add each bullet point as a list item, ensuring it's sanitized.
    for point in bullet_points:
        sanitized_point = html.escape(point)
        options_html += f"  <li>{sanitized_point}</li>\n"
    options_html += "</ul>"  # Close the unordered list.
    return options_html


def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    """
    Creates a structured HTML representation of a forum post.

    The function sanitizes the title and body content, and formats the
    discussion points as an HTML unordered list.

    Args:
        post_title: The title or main topic of the forum post.
        post_body: The main body or content of the forum post.
        bullet_points: A collection of user-provided discussion points or options.

    Returns:
        A string containing the fully formatted and sanitized HTML representation
        of the forum post, including the header, message body, and safely
        rendered discussion points.
    """
    # Sanitize the main title and body to prevent HTML injection.
    sanitized_title = html.escape(post_title)
    sanitized_body = html.escape(post_body)

    # Generate the HTML for the bullet points, if any.
    options_html = _create_bullet_list_html(bullet_points)

    # Format the final HTML string using the predefined template.
    return FORUM_POST_FORMAT.format(
        header=sanitized_title,
        message=sanitized_body,
        options=options_html
    )
