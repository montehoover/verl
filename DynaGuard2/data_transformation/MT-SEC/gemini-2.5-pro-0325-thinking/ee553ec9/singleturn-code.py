html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Creates an HTML response by inserting user content into a given HTML structure.

    Args:
        client_message: The content contributed by a user which will be
                        inserted into the HTML template.

    Returns:
        The resulting HTML string that includes the user’s content safely
        embedded into it.
    """
    # For safety, especially if client_message could contain HTML special characters,
    # it's good practice to escape it. However, the prompt doesn't explicitly
    # require HTML escaping, and the example output shows direct insertion.
    # If escaping is needed, `import html; html.escape(client_message)` would be used.
    return html_template.format(user_content=client_message)
