import html

def sanitize_input(input_str: str) -> str:
    """
    Sanitizes an input string by escaping HTML special characters.
    """
    return html.escape(input_str)

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
    print("Base HTML Structure:")
    print(base_html)

    # Example usage of sanitize_input:
    unsafe_input = "<script>alert('XSS')</script>"
    safe_input = sanitize_input(unsafe_input)
    print(f"\nOriginal Input: {unsafe_input}")
    print(f"Sanitized Input: {safe_input}")

    # Example of using sanitized input in the HTML structure
    content_with_safe_input = base_html.replace("<!-- MAIN_CONTENT_PLACEHOLDER -->", f"<p>{safe_input}</p>")
    print("\nHTML with Sanitized Input:")
    print(content_with_safe_input)
