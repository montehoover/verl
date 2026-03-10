def create_post_html(post_title, post_body):
    return f"<h2>{post_title}</h2><p>{post_body}</p>"

def format_discussion_points(points):
    list_items = ''.join([f'<li>{point}</li>' for point in points])
    return f'<ul>{list_items}</ul>'
