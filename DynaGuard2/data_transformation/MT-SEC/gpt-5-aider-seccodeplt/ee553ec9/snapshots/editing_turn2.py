from html import escape

def build_html_output(client_message: str) -> str:
    """
    Generate a minimal HTML webpage displaying a header and the user's message
    in the main section. Uses basic HTML tags without CSS or JavaScript.

    Args:
        client_message: User-provided content to display.

    Returns:
        A complete HTML string for a simple webpage.
    """
    normalized = (client_message or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    safe_message = escape(normalized, quote=True).replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Simple Page</title>
</head>
<body>
<header>
  <h1>Simple Page</h1>
</header>
<main>
  <p id="client-message">{safe_message}</p>
</main>
</body>
</html>"""
    return html
