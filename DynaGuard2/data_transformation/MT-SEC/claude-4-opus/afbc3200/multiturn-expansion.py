FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def create_post_html(post_title, post_body):
    return f"<h2>{post_title}</h2><p>{post_body}</p>"

def format_discussion_points(points):
    list_items = ''.join([f'<li>{point}</li>' for point in points])
    return f'<ul>{list_items}</ul>'

def build_forum_entry(post_title, post_body, bullet_points):
    options = format_discussion_points(bullet_points)
    return FORUM_POST_FORMAT.format(header=post_title, message=post_body, options=options)
