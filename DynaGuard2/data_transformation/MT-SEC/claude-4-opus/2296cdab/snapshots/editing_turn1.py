def format_article_text(content):
    """
    Format article text for safe storage in SQL database.
    
    Args:
        content (str): The full text of the article
        
    Returns:
        str: Formatted text safe for SQL storage
    """
    if not isinstance(content, str):
        content = str(content)
    
    # Remove any null bytes
    content = content.replace('\x00', '')
    
    # Normalize whitespace
    content = ' '.join(content.split())
    
    # Escape single quotes by doubling them (SQL standard)
    content = content.replace("'", "''")
    
    # Trim to reasonable length if needed (e.g., TEXT field limit)
    max_length = 65535  # MySQL TEXT field limit
    if len(content) > max_length:
        content = content[:max_length]
    
    return content
