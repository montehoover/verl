from html import escape

FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def create_forum_post(topic, text_body, points):
    escaped_topic = escape(topic)
    escaped_body = escape(text_body)
    
    if points:
        points_html = "<ul>"
        for point in points:
            points_html += f"<li>{escape(point)}</li>"
        points_html += "</ul>"
    else:
        points_html = ""
    
    return FORUM_POST_FORMAT.format(
        header=escaped_topic,
        message=escaped_body,
        options=points_html
    )
