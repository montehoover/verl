import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

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

def render_article_page(headline: str, content: str) -> str:
    """
    Formats and displays a user-submitted blog post, ensuring content is sanitized.

    Args:
        headline: The headline of the blog post.
        content: The main content of the blog post.

    Returns:
        A sanitized and complete HTML representation of the blog post.
    """
    safe_headline = sanitize_html_content(headline)
    safe_content = sanitize_html_content(content)
    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)

if __name__ == '__main__':
    # Example usage for create_basic_webpage:
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
    html_content_basic_page = create_basic_webpage(safe_title, safe_description)
    
    # Save to a file to view in browser
    with open("index.html", "w") as f:
        f.write(html_content_basic_page)
    
    print("\nGenerated index.html with sanitized content. You can open it in your browser.")

    # Example usage for render_article_page:
    article_headline = "Exciting News: A New Blog Post!"
    article_content = "This is the content of our new blog post. It includes <html> tags and & special characters that need sanitizing. <script>alert('Still XSS?');</script>"
    
    formatted_article_html = render_article_page(article_headline, article_content)
    
    # For demonstration, let's create a full HTML page for the article
    full_article_page_html = create_basic_webpage(
        title=sanitize_html_content(article_headline),  # Sanitize title for the <title> tag
        description=formatted_article_html  # The body is already formatted and sanitized
    )

    print(f"\nFormatted Article HTML:\n{formatted_article_html}")

    # Save the article page to a file
    with open("article.html", "w") as f:
        f.write(full_article_page_html)
    
    print("\nGenerated article.html. You can open it in your browser.")
