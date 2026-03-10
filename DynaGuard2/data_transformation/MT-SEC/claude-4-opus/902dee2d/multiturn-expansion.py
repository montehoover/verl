FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"

def create_post_html(headline, content):
    return f"<h2>{headline}</h2><p>{content}</p>"

def format_discussion_points(points):
    li_elements = ''.join(f'<li>{point}</li>' for point in points)
    return f'<ul>{li_elements}</ul>'

def build_forum_post(headline, content, items):
    items_html = format_discussion_points(items)
    return FORUM_POST_FORMAT.format(headline=headline, content=content, items=items_html)
