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


def insert_user_content(html_template, user_message):
    """
    Inserts user content into an HTML template.
    
    Args:
        html_template (str): HTML template with {user_content} placeholder
        user_message (str): User message to insert into the template
    
    Returns:
        str: HTML string with user content inserted
    """
    return html_template.format(user_content=user_message)
