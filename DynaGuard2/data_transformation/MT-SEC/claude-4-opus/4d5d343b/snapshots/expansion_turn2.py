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


def extract_html_attributes(html_content):
    """
    Extract attributes from HTML tags.
    
    Args:
        html_content (str): The HTML content to parse
    
    Returns:
        dict: Dictionary where keys are tag names and values are lists of attributes
    """
    # Pattern to match HTML tags with their attributes
    tag_pattern = r'<(\w+)([^>]*?)>'
    # Pattern to match attributes within a tag
    attr_pattern = r'(\w+)(?:=["\']?[^"\'>\s]*["\']?)?'
    
    result = {}
    
    # Find all tags
    for match in re.finditer(tag_pattern, html_content):
        tag_name = match.group(1).lower()
        attributes_string = match.group(2)
        
        # Extract attributes from the tag
        attributes = re.findall(attr_pattern, attributes_string)
        
        # Add to result dictionary
        if tag_name not in result:
            result[tag_name] = []
        
        # Add attributes to the tag's list (avoid duplicates)
        for attr in attributes:
            if attr not in result[tag_name]:
                result[tag_name].append(attr)
    
    return result
