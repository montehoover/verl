def count_words(text):
    """Count the number of words in a string separated by spaces."""
    if not text or text.strip() == "":
        return 0
    return len(text.split())

def extract_words(text):
    """Extract words from a string and return them as a list."""
    if not text or text.strip() == "":
        return []
    return text.split()
