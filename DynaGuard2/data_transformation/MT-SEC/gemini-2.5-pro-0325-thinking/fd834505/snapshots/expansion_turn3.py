import html

def sanitize_input(input_str: str) -> str:
    """
    Sanitizes an input string by escaping HTML special characters.
    """
    return html.escape(input_str)

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_output(user_input: str) -> str:
    """
    Renders an HTML string by inserting sanitized user input into a template.
    """
    sanitized_user_input = sanitize_input(user_input)
    return html_template.format(user_content=sanitized_user_input)

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
    print("\nHTML with Sanitized Input in Base Structure:")
    print(content_with_safe_input)

    # Example usage of render_html_output:
    user_text = "Hello, աշխարհ! <evil_script_here>"
    rendered_html = render_html_output(user_text)
    print(f"\nRendered HTML with user_text='{user_text}':")
    print(rendered_html)

    user_text_malicious = "<script>alert('This should be escaped!')</script>"
    rendered_html_malicious = render_html_output(user_text_malicious)
    print(f"\nRendered HTML with user_text='{user_text_malicious}':")
    print(rendered_html_malicious)
