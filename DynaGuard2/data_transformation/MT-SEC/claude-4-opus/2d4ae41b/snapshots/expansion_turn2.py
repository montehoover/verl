import html

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Messages</title>
</head>
<body>
    <div class="container">
        <h1>User Message</h1>
        <div class="message-content">
            {user_content}
        </div>
    </div>
</body>
</html>"""

def embed_user_content(template, content):
    escaped_content = html.escape(content)
    return template.replace('{user_content}', escaped_content)
