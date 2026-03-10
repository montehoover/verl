import html

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>User Message</title>
</head>
<body>
  <main id="message">
    {user_content}
  </main>
</body>
</html>
"""

def embed_user_content(template: str, content: str) -> str:
  """
  Replace the {user_content} placeholder in the template with HTML-escaped content.
  """
  escaped = html.escape(content, quote=True)
  return template.replace("{user_content}", escaped)
