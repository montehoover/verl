import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def _insert_content_into_template(template: str, content: str) -> str:
    """
    Inserts content into an HTML template.

    Args:
        template: The HTML template string with a placeholder for content.
        content: The content to insert into the template.

    Returns:
        The HTML string with the content inserted.
    """
    return template.format(user_content=content)

def build_html_output(client_message: str) -> str:
    """
    Creates an HTML response by inserting user content into a given HTML structure.

    The generated response will display user-provided information within a
    predefined format.

    Args:
        client_message: The content contributed by a user which will be
                        inserted into the HTML template.

    Returns:
        The resulting HTML string that includes the user’s content safely
        embedded into it.
    """
    # Escape the client message to prevent XSS vulnerabilities
    safe_client_message = html.escape(client_message)
    return _insert_content_into_template(html_template, safe_client_message)
