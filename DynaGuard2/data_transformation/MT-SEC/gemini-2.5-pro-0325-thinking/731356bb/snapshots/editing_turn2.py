def render_article_page(headline: str, content: str):
    """
    Generates a plain text representation of a user's blog post submission.

    Args:
        headline: The title of the blog post.
        content: The text body of the blog post.
    """
    print(f"<h1>{headline}</h1>")
    print(f"<p>{content}</p>")
