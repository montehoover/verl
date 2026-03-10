def format_article_text(headline: str, content: str) -> str:
    """
    Formats the headline and main text of an article to be safely stored in a SQL database.

    Args:
        headline: The title of the article.
        content: A string containing the full text of the article.

    Returns:
        A string where the headline and content have been processed for safe SQL storage.
        The headline appears at the start of the string.
        Currently, this function escapes single quotes, which is a common
        requirement for SQL string literals. For robust SQL injection prevention,
        always use parameterized queries or an ORM.
    """
    safe_headline = headline.replace("'", "''") if headline else ""
    safe_content = content.replace("'", "''") if content else ""

    # Combine headline and content, ensuring a space if both exist and content is not empty.
    # If content is empty, no extra space is needed.
    # If headline is empty, just return safe_content.
    if safe_headline and safe_content:
        return f"{safe_headline}\n\n{safe_content}"
    elif safe_headline:
        return safe_headline
    elif safe_content:
        return safe_content
    else:
        return ""
