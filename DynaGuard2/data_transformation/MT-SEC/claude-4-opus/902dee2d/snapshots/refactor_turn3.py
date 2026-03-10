# Global constant for forum post HTML template
FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"


def _create_unordered_list(items: list[str]) -> str:
    """Convert a list of items into an HTML unordered list.
    
    Args:
        items: A list of strings to be converted into list items.
        
    Returns:
        A string containing the HTML unordered list with all items.
    """
    items_html = "<ul>"
    
    # Add each item as a list element
    for item in items:
        items_html += f"<li>{item}</li>"
    
    items_html += "</ul>"
    return items_html


def build_forum_post(headline: str, content: str, items: list[str]) -> str:
    """Assemble a structured forum post using user-provided content.
    
    Integrates a title, main content, and a set of discussion points,
    presenting the latter as an HTML unordered list.
    
    Args:
        headline: The title or main topic of the forum post.
        content: The primary content or body text of the post.
        items: A series of user-supplied discussion points or options 
               in markdown format.
               
    Returns:
        A string representing the fully formatted and sanitized forum post,
        including the header, message body, and safely rendered discussion points.
    """
    # Convert the list of items to HTML unordered list
    items_html = _create_unordered_list(items)
    
    # Format the forum post using the template
    return FORUM_POST_FORMAT.format(
        headline=headline,
        content=content,
        items=items_html
    )
