"""Module for creating structured HTML forum post entries."""

# Global constant for forum post HTML template
FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"


def create_html_list(items: list) -> str:
    """Create an HTML unordered list from a list of items.
    
    Args:
        items: A list of strings to be converted into HTML list items.
        
    Returns:
        A string containing an HTML unordered list (<ul>) with the provided items,
        or an empty string if the items list is empty.
    """
    if not items:
        return ''
    
    # Build list items and wrap them in <ul> tags
    list_items = ''.join([f'<li>{item}</li>' for item in items])
    return f'<ul>{list_items}</ul>'


def build_forum_entry(post_title: str, post_body: str, bullet_points: list) -> str:
    """Build a structured HTML representation of a forum post.
    
    Creates a formatted forum post with a title, main content, and discussion points.
    The discussion points are formatted as an HTML unordered list.
    
    Args:
        post_title: The title or main topic of the forum post.
        post_body: The main body or content of the forum post.
        bullet_points: A collection of user-provided discussion points or options
                      in markdown format.
        
    Returns:
        A string containing the fully formatted and sanitized HTML representation
        of the forum post, including the header, message body, and safely rendered
        discussion points.
    """
    # Create the HTML list for discussion points
    options_html = create_html_list(bullet_points)
    
    # Format the complete forum entry using the template
    forum_entry = FORUM_POST_FORMAT.format(
        header=post_title,
        message=post_body,
        options=options_html
    )
    
    return forum_entry
