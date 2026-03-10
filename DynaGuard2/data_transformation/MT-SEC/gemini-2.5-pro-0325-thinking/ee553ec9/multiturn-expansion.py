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

import html

# Global variable for the HTML template
html_template_global = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Assembles an HTML response by securely embedding user-provided content
    into a predefined template.

    Args:
        client_message: The message from the client to be embedded.

    Returns:
        A formatted HTML string with the client message securely embedded.
    """
    # Securely embed the client message by escaping HTML special characters
    safe_client_message = html.escape(client_message)
    return html_template_global.format(user_content=safe_client_message)
