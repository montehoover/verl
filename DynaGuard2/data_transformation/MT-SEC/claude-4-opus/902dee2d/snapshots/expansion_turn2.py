def create_post_html(headline, content):
    return f"<h2>{headline}</h2><p>{content}</p>"

def format_discussion_points(points):
    li_elements = ''.join(f'<li>{point}</li>' for point in points)
    return f'<ul>{li_elements}</ul>'
