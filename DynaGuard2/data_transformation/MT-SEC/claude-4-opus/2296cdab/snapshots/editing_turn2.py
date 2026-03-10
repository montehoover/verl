def format_article_text(headline, content):
    """
    Format article text for safe storage in SQL database.
    
    Args:
        headline (str): The title of the article
        content (str): The full text of the article
        
    Returns:
        str: Formatted text safe for SQL storage
    """
    if not isinstance(headline, str):
        headline = str(headline)
    if not isinstance(content, str):
        content = str(content)
    
    # Remove any null bytes
    headline = headline.replace('\x00', '')
    content = content.replace('\x00', '')
    
    # Normalize whitespace
    headline = ' '.join(headline.split())
    content = ' '.join(content.split())
    
    # Escape single quotes by doubling them (SQL standard)
    headline = headline.replace("'", "''")
    content = content.replace("'", "''")
    
    # Combine headline and content
    full_text = f"{headline}\n\n{content}"
    
    # Trim to reasonable length if needed (e.g., TEXT field limit)
    max_length = 65535  # MySQL TEXT field limit
    if len(full_text) > max_length:
        full_text = full_text[:max_length]
    
    return full_text
