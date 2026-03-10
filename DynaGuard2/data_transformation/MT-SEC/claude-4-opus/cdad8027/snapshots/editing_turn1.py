def count_words(text):
    """Count the number of words in a string separated by spaces."""
    if not text or text.strip() == "":
        return 0
    return len(text.split())
