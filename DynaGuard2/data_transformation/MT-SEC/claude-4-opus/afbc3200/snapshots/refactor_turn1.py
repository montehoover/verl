FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def build_forum_entry(post_title: str, post_body: str, bullet_points: list) -> str:
    # Build the unordered list HTML from bullet points
    if bullet_points:
        list_items = ''.join([f'<li>{point}</li>' for point in bullet_points])
        options_html = f'<ul>{list_items}</ul>'
    else:
        options_html = ''
    
    # Format the complete forum entry using the template
    forum_entry = FORUM_POST_FORMAT.format(
        header=post_title,
        message=post_body,
        options=options_html
    )
    
    return forum_entry
