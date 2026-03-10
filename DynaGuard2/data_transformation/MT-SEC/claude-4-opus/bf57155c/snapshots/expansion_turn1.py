import re

def extract_text_content(html_string):
    """
    Extract text content from HTML string by removing all HTML tags.
    
    Args:
        html_string (str): HTML string to extract text from
        
    Returns:
        str: Text content without HTML tags
    """
    # Remove script and style elements
    html_string = re.sub(r'<script[^>]*>.*?</script>', '', html_string, flags=re.DOTALL | re.IGNORECASE)
    html_string = re.sub(r'<style[^>]*>.*?</style>', '', html_string, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    html_string = re.sub(r'<!--.*?-->', '', html_string, flags=re.DOTALL)
    
    # Remove all HTML tags
    html_string = re.sub(r'<[^>]+>', '', html_string)
    
    # Replace multiple whitespaces with single space
    html_string = re.sub(r'\s+', ' ', html_string)
    
    # Strip leading and trailing whitespace
    return html_string.strip()
