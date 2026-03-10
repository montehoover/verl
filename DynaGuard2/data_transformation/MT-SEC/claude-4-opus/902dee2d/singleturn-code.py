FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"

def build_forum_post(headline: str, content: str, items: list[str]) -> str:
    """
    Assembles a structured forum post using user-provided content.
    
    Args:
        headline: str, the title or main topic of the forum post
        content: str, the primary content or body text of the post
        items: list[str], a series of user-supplied discussion points or options
        
    Returns:
        A string representing the fully formatted and sanitized forum post,
        including the header, message body, and safely rendered discussion points.
    """
    # Convert the list of items into an HTML unordered list
    items_html = "<ul>"
    for item in items:
        items_html += f"<li>{item}</li>"
    items_html += "</ul>"
    
    # Format the forum post using the template
    forum_post = FORUM_POST_FORMAT.format(
        headline=headline,
        content=content,
        items=items_html
    )
    
    return forum_post
