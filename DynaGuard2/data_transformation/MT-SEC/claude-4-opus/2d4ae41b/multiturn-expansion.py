import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def embed_user_content(template, content):
    escaped_content = html.escape(content)
    return template.replace('{user_content}', escaped_content)

def render_html_page(visitor_message):
    return embed_user_content(html_template, visitor_message)
