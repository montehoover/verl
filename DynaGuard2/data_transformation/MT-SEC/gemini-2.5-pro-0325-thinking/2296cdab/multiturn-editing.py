import html

BLOG_FORMAT = "<h1>{headline}</h1><p>{content}</p>"

def format_blog_entry(headline: str, content: str) -> str:
    """
    Formats the headline and content for an HTML blog entry.

    Args:
        headline: The main heading of the blog post.
        content: The complete textual content of the blog post.

    Returns:
        A sanitized string using HTML formatting, with the headline
        in an h1 tag and the content inside a paragraph tag.
    """
    # Sanitize inputs to prevent XSS
    safe_headline = html.escape(headline) if headline else ""
    safe_content = html.escape(content) if content else ""

    return BLOG_FORMAT.format(headline=safe_headline, content=safe_content)
