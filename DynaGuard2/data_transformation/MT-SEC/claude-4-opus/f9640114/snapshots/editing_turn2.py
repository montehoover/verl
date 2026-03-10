def format_post_with_points(title, body, points):
    formatted_points = '\n'.join([f"• {point}" for point in points])
    return f"Title: {title}, Body: {body}, Points:\n{formatted_points}"
