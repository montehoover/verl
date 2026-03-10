import html


def create_basic_webpage(title: str, description: str) -> str:
    """
    Create a basic HTML page with the given title and description.

    Args:
        title: The page title, used in both the <title> and <h1>.
        description: The page description, rendered as a paragraph.

    Returns:
        A string containing a basic HTML document.
    """
    safe_title = html.escape(title, quote=True)
    safe_description = html.escape(description, quote=True)

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
