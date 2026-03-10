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
    html_content = create_basic_webpage(page_title, page_description)
    
    # Save to a file to view in browser
    with open("index.html", "w") as f:
        f.write(html_content)
    
    print("Generated index.html. You can open it in your browser.")
