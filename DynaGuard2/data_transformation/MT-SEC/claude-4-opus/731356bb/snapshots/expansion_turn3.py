BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"


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


def render_article_page(headline, content):
    """Format and display user-submitted blog posts with sanitized content."""
    sanitized_headline = sanitize_html_content(headline)
    sanitized_content = sanitize_html_content(content)
    
    blog_html = BLOG_FORMAT.format(headline=sanitized_headline, content=sanitized_content)
    
    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{sanitized_headline}</title>
</head>
<body>
    {blog_html}
</body>
</html>"""
    
    return html_page
