BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def create_article_template(title, content):
    return f"<h2>{title}</h2><p>{content}</p>"

def sanitize_content(content):
    content = content.replace("&", "&amp;")
    content = content.replace("<", "&lt;")
    content = content.replace(">", "&gt;")
    return content

def publish_blog_article(subject, description):
    return BLOG_FORMAT.format(subject=subject, description=description)
