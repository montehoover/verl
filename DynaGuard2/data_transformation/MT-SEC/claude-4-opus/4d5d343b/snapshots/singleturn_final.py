import re

def grab_html_tags(html_code: str) -> list:
    """
    Scans a provided string for HTML tags using a regular expression.
    
    Args:
        html_code: An HTML string that serves as the input for parsing.
        
    Returns:
        A list of all HTML tags identified in the given input.
    """
    # Regular expression to match HTML tags
    # This pattern captures opening tags, closing tags, and self-closing tags
    tag_pattern = r'<(/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s+[^>]*)?)(?:/?)>'
    
    tags = []
    
    # Find all matches in the HTML string
    matches = re.finditer(tag_pattern, html_code)
    
    for match in matches:
        closing_slash = match.group(1)  # '/' for closing tags, empty for opening
        tag_name = match.group(2)       # The tag name
        attributes = match.group(3)     # Any attributes
        
        # Construct the tag string based on what was captured
        if closing_slash:
            # It's a closing tag
            tags.append(closing_slash + tag_name)
        else:
            # It's an opening tag
            full_tag = tag_name + attributes.rstrip()
            tags.append(full_tag)
    
    return tags
