def create_basic_html(title: str, content: str) -> str:
    """
    Constructs a simple HTML page.

    Args:
        title: The title of the page, to be displayed in an <h1> tag.
        content: The main content of the page, to be displayed in a <p> tag.

    Returns:
        A string representing an HTML document.
    """
    html_template = f"""
<!DOCTYPE html>
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
</html>
"""
    return html_template

def insert_user_content(html_template: str, user_message: str) -> str:
    """
    Inserts a user message into an HTML template.

    Args:
        html_template: The HTML template string with a {user_content} placeholder.
        user_message: The user message to insert.

    Returns:
        A new HTML string with the user message inserted.
    """
    return html_template.format(user_content=user_message)
