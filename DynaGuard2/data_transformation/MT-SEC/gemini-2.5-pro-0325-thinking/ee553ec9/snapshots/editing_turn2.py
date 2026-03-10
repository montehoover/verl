def build_html_output(client_message: str) -> str:
    """
    Generates a simple HTML webpage displaying a header and user-provided content.

    Args:
        client_message: The content to display in the main section of the page.

    Returns:
        A complete HTML string representing the webpage.
    """
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>My Webpage</title>
</head>
<body>
    <header>
        <h1>Welcome to My Page</h1>
    </header>
    <main>
        <p>{client_message}</p>
    </main>
</body>
</html>
"""
    return html_template.strip()
