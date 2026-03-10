def create_basic_webpage(title, description):
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <h1>{title}</h1>
    <p>{description}</p>
</body>
</html>"""
    return html_template


def sanitize_html_content(content):
    """Sanitize user-generated content for safe HTML display."""
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }
    
    sanitized = content
    for char, escape in replacements.items():
        sanitized = sanitized.replace(char, escape)
    
    return sanitized
