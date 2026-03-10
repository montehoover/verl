def split_text(text, delimiter):
    """
    Split a string into a list of substrings based on the given delimiter.
    
    Args:
        text (str): The string to split
        delimiter (str): The delimiter to split by
    
    Returns:
        list: A list of substrings
    """
    return text.split(delimiter)


def identify_parts(text):
    """
    Identify and separate parts of a string formatted as 'prefix(suffix)'.
    
    Args:
        text (str): The string formatted as 'prefix(suffix)'
    
    Returns:
        dict: A dictionary with keys 'prefix' and 'suffix'
    """
    # Find the position of the opening parenthesis
    paren_index = text.find('(')
    
    if paren_index == -1:
        # No parentheses found
        return {'prefix': text, 'suffix': ''}
    
    # Extract prefix (everything before the opening parenthesis)
    prefix = text[:paren_index]
    
    # Find the closing parenthesis
    close_paren_index = text.find(')', paren_index)
    
    if close_paren_index == -1:
        # No closing parenthesis found
        suffix = text[paren_index + 1:]
    else:
        # Extract suffix (everything inside the parentheses)
        suffix = text[paren_index + 1:close_paren_index]
    
    return {'prefix': prefix, 'suffix': suffix}
