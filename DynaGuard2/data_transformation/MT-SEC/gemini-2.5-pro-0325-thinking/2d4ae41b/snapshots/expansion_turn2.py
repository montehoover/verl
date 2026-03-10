import html

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Message</title>
</head>
<body>
    <div>
        {user_content}
    </div>
</body>
</html>
"""

def embed_user_content(template: str, content: str) -> str:
    """
    Safely embeds user content into an HTML template.

    Args:
        template: The HTML template string with a '{user_content}' placeholder.
        content: The user content string to embed.

    Returns:
        The HTML template with the user content safely embedded.
    """
    escaped_content = html.escape(content)
    return template.replace('{user_content}', escaped_content)
