import html


def sanitize_html_content(content: str) -> str:
    """
    Sanitize user-generated content for safe HTML rendering by escaping special characters.
    Converts characters like &, <, >, and quotes to their HTML-safe equivalents.
    """
    return html.escape(content, quote=True)


def create_basic_webpage(title: str, description: str) -> str:
    """
    Create a basic HTML page with the given title and description.

    Args:
        title: The page title, used in both the <title> and <h1>.
        description: The page description, rendered as a paragraph.

    Returns:
        A string containing a basic HTML document.
    """
    safe_title = sanitize_html_content(title)
    safe_description = sanitize_html_content(description)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
</head>
<body>
  <h1>{safe_title}</h1>
  <p>{safe_description}</p>
</body>
</html>"""
