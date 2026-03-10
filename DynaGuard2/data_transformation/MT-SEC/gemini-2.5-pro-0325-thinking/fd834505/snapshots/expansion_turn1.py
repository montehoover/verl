def generate_base_html():
    """
    Generates a basic HTML structure with a placeholder for main content.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome</title>
</head>
<body>
    <header>
        <h1>My Web Page</h1>
    </header>
    <main>
        <!-- MAIN_CONTENT_PLACEHOLDER -->
    </main>
    <footer>
        <p>&copy; 2025 My Web Page</p>
    </footer>
</body>
</html>
"""
    return html_content

if __name__ == '__main__':
    # Example usage:
    base_html = generate_base_html()
    print(base_html)
