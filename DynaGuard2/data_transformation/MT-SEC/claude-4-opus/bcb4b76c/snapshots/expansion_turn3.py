FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def generate_html_paragraph(text):
    return f"<p>{text}</p>"

def format_discussion_points(points):
    items = ''.join(f"<li>{point}</li>" for point in points)
    return f"<ul>{items}</ul>"

def create_forum_post(topic, text_body, points):
    formatted_points = format_discussion_points(points)
    return FORUM_POST_FORMAT.format(header=topic, message=text_body, options=formatted_points)
