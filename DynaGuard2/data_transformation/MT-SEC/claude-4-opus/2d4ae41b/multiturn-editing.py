import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def create_static_html():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome Page</title>
</head>
<body>
    <h1>Welcome to My Page</h1>
    <p>This is a simple static HTML page generated with Python.</p>
</body>
</html>"""
    return html

def create_dynamic_html(user_message):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome Page</title>
</head>
<body>
    <h1>Welcome to My Page</h1>
    <p>{user_message}</p>
</body>
</html>"""
    return html

def render_html_page(visitor_message):
    escaped_message = html.escape(visitor_message)
    return html_template.format(user_content=escaped_message)
