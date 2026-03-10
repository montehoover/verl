import re

def find_html_elements(html_content, elements):
    """
    Check if a given HTML string contains certain elements.
    
    Args:
        html_content (str): The HTML content to search in
        elements (list): List of element names to search for (e.g., ['div', 'span'])
    
    Returns:
        bool: True if any of the specified elements are present, False otherwise
    """
    for element in elements:
        # Create regex pattern to match opening tags of the element
        # This pattern matches <element> or <element ...> (with attributes)
        pattern = f'<{element}(?:\\s|>)'
        if re.search(pattern, html_content, re.IGNORECASE):
            return True
    return False
