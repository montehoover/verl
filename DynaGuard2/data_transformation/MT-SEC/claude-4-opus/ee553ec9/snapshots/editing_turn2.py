def build_html_output(client_message: str) -> str:
    """
    Generate an HTML email body by inserting user-provided details into an email template.
    
    Args:
        client_message: The message to personalize the email's content
        
    Returns:
        A complete HTML string with the user's message inside the email body
    """
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Simple Webpage</title>
</head>
<body>
    <header>
        <h1>Welcome to Our Page</h1>
    </header>
    <main>
        <section>
            <p>{client_message}</p>
        </section>
    </main>
</body>
</html>"""
    
    return html_template
