import html

# Predefined HTML template for blog posts
BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"


def sanitize_text(text: str) -> str:
    """
    Sanitize user input to prevent XSS attacks by escaping HTML characters.
    
    Args:
        text: Raw text input from user that may contain HTML characters
        
    Returns:
        Sanitized text with HTML entities properly escaped
    """
    # Guard clause: handle None or non-string inputs
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    return html.escape(text)


def format_blog_html(heading: str, text: str) -> str:
    """
    Format sanitized heading and text into the blog HTML template.
    
    Args:
        heading: Sanitized heading text for the blog post
        text: Sanitized body text for the blog post
        
    Returns:
        Formatted HTML string using the BLOG_FORMAT template
    """
    return BLOG_FORMAT.format(heading=heading, text=text)


def create_blog_post(heading: str, text: str) -> str:
    """
    Prepare user-submitted blog posts for secure web display.
    
    Takes raw user input for heading and text, sanitizes them to prevent
    XSS attacks, and formats them into a proper HTML structure.
    
    Args:
        heading: The main heading or subject of the blog post
        text: The complete textual content of the blog post
        
    Returns:
        Sanitized and HTML-formatted blog post ready for secure presentation
        
    Example:
        >>> create_blog_post("My Blog", "Hello <script>alert('XSS')</script>")
        '<h1>My Blog</h1><p>Hello &lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;</p>'
    """
    # Sanitize both inputs to prevent XSS attacks
    sanitized_heading = sanitize_text(heading)
    sanitized_text = sanitize_text(text)
    
    # Format the sanitized content into HTML
    return format_blog_html(sanitized_heading, sanitized_text)
