def count_words(text):
    """Count the number of words in a given string.
    
    Args:
        text: A string to count words in
        
    Returns:
        The total number of words in the string
    """
    # Split by whitespace and filter out empty strings
    words = text.split()
    return len(words)
