import html

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def build_forum_entry(post_title, post_body, bullet_points):
    escaped_title = html.escape(post_title)
    escaped_body = html.escape(post_body)
    
    if bullet_points:
        list_items = ''.join([f"<li>{html.escape(point)}</li>" for point in bullet_points])
        options_html = f"<ul>{list_items}</ul>"
    else:
        options_html = ""
    
    return FORUM_POST_FORMAT.format(
        header=escaped_title,
        message=escaped_body,
        options=options_html
    )
