import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def embed_user_content(template: str, content: str) -> str:
  """
  Replace the {user_content} placeholder in the template with HTML-escaped content.
  """
  escaped = html.escape(content, quote=True)
  return template.replace("{user_content}", escaped)

def render_html_page(visitor_message: str) -> str:
  """
  Generate an HTML page by safely embedding the visitor's message into the template.
  """
  return embed_user_content(html_template, visitor_message)
