def generate_html_paragraph(text):
    return f"<p>{text}</p>"

def format_discussion_points(points):
    items = ''.join(f"<li>{point}</li>" for point in points)
    return f"<ul>{items}</ul>"
