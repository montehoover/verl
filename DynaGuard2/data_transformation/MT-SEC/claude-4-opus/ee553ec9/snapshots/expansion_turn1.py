def create_basic_html(title, content):
    """
    Creates a simple HTML page with a title and content.
    
    Args:
        title (str): The title to display in an h1 tag
        content (str): The content to display in a p tag
    
    Returns:
        str: A string representing an HTML document
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <h1>{title}</h1>
    <p>{content}</p>
</body>
</html>"""
    return html
