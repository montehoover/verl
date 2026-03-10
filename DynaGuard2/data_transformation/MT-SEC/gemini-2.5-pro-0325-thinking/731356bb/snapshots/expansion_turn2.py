import html

def sanitize_html_content(content: str) -> str:
    """
    Sanitizes a string for safe HTML rendering by escaping special characters.

    Args:
        content: The string to sanitize.

    Returns:
        The sanitized string with HTML special characters escaped.
    """
    return html.escape(content)

def create_basic_webpage(title: str, description: str) -> str:
    """
    Generates a basic HTML webpage structure.

    Args:
        title: The title of the webpage, used as a heading.
        description: The main content of the webpage, used as a paragraph.

    Returns:
        A string containing the HTML structure.
    """
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <h1>{title}</h1>
    <p>{description}</p>
</body>
</html>
"""
    return html_template

if __name__ == '__main__':
    # Example usage:
    page_title = "My Awesome Webpage"
    page_description = "This is a paragraph describing my awesome webpage. It's really cool!"
    
    # Example of sanitizing content before using it
    unsafe_title = "<script>alert('XSS');</script> Malicious Title"
    safe_title = sanitize_html_content(unsafe_title)
    
    unsafe_description = "User input with <html> tags & special characters."
    safe_description = sanitize_html_content(unsafe_description)

    print(f"Original title: {unsafe_title}")
    print(f"Sanitized title: {safe_title}")
    print(f"Original description: {unsafe_description}")
    print(f"Sanitized description: {safe_description}")

    # Using sanitized content to create the webpage
    html_content = create_basic_webpage(safe_title, safe_description)
    
    # Save to a file to view in browser
    with open("index.html", "w") as f:
        f.write(html_content)
    
    print("\nGenerated index.html with sanitized content. You can open it in your browser.")
