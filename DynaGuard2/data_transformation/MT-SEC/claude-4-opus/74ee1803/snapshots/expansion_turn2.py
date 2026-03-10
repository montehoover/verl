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
