import re


def get_html_tags(html_input: str) -> list:
    """Extract HTML tags from a provided string using regular expressions.
    
    Args:
        html_input: An HTML-formatted string as input.
        
    Returns:
        A list containing the tags that are present within the input HTML string.
    """
    # Define the regular expression pattern for matching HTML tags
    # This pattern captures tags like <tag>, </tag>, and self-closing tags like <tag/>
    html_tag_pattern = r'<[^>]+>'
    
    # Extract all HTML tags from the input string
    extracted_tags = re.findall(html_tag_pattern, html_input)
    
    return extracted_tags
