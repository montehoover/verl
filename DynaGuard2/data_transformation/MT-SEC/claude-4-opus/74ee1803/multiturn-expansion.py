BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"

def create_html_paragraph(text):
    return f"<p>{text}</p>"

def sanitize_html(text):
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;'
    }
    for char, escape in replacements.items():
        text = text.replace(char, escape)
    return text

def display_blog_entry(headline, text):
    sanitized_headline = sanitize_html(headline)
    sanitized_text = sanitize_html(text)
    return BLOG_FORMAT.format(headline=sanitized_headline, text=sanitized_text)
