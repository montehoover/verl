def generate_base_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome</title>
</head>
<body>
    <header>
        <h1>Welcome to Our Site</h1>
    </header>
    <main>
        <!-- Main content placeholder -->
        {content}
    </main>
    <footer>
        <p>&copy; 2024 All rights reserved</p>
    </footer>
</body>
</html>"""

def sanitize_input(input_str):
    return (input_str
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))
