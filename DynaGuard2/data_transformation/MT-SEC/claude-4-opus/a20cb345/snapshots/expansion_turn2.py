def create_article_template(title, content):
    return f"<h2>{title}</h2><p>{content}</p>"

def sanitize_content(content):
    content = content.replace("&", "&amp;")
    content = content.replace("<", "&lt;")
    content = content.replace(">", "&gt;")
    return content
