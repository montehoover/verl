def format_article_text(content: str) -> str:
    """
    Formats the main text of an article to be safely stored in a SQL database.

    Args:
        content: A string containing the full text of the article.

    Returns:
        A string that has been processed for safe SQL storage.
        Currently, this function escapes single quotes, which is a common
        requirement for SQL string literals. For robust SQL injection prevention,
        always use parameterized queries or an ORM.
    """
    if content is None:
        return ""
    # Escape single quotes for SQL by replacing them with two single quotes
    return content.replace("'", "''")
