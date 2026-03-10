FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def create_html_list(items: list) -> str:
    """Create an HTML unordered list from a list of items."""
    if not items:
        return ''
    
    list_items = ''.join([f'<li>{item}</li>' for item in items])
    return f'<ul>{list_items}</ul>'

def build_forum_entry(post_title: str, post_body: str, bullet_points: list) -> str:
    # Create the HTML list for discussion points
    options_html = create_html_list(bullet_points)
    
    # Format the complete forum entry using the template
    forum_entry = FORUM_POST_FORMAT.format(
        header=post_title,
        message=post_body,
        options=options_html
    )
    
    return forum_entry
