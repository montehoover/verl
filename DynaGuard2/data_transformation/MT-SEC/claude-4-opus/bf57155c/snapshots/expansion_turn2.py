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


def find_html_elements(html_string, tag_name):
    """
    Find all occurrences of a specific HTML tag in the given HTML string.
    
    Args:
        html_string (str): HTML string to search in
        tag_name (str): Name of the HTML tag to find
        
    Returns:
        list: List of dictionaries containing tag information (tag, attributes, content)
    """
    results = []
    
    # Pattern to match opening tags with attributes
    opening_pattern = rf'<{tag_name}(\s[^>]*)?>.*?</{tag_name}>'
    
    # Find all matches including nested tags
    matches = re.finditer(opening_pattern, html_string, flags=re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        full_tag = match.group(0)
        
        # Extract attributes from opening tag
        opening_tag_match = re.match(rf'<{tag_name}(\s[^>]*)?>?', full_tag, flags=re.IGNORECASE)
        attributes = {}
        
        if opening_tag_match and opening_tag_match.group(1):
            # Parse attributes
            attr_string = opening_tag_match.group(1).strip()
            attr_pattern = r'(\w+)(?:=(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+)))?'
            
            for attr_match in re.finditer(attr_pattern, attr_string):
                attr_name = attr_match.group(1)
                attr_value = attr_match.group(2) or attr_match.group(3) or attr_match.group(4) or True
                attributes[attr_name] = attr_value
        
        # Extract content between opening and closing tags
        content_match = re.search(rf'<{tag_name}[^>]*>(.*?)</{tag_name}>', full_tag, flags=re.DOTALL | re.IGNORECASE)
        content = content_match.group(1) if content_match else ''
        
        results.append({
            'tag': tag_name,
            'attributes': attributes,
            'content': content.strip(),
            'full_tag': full_tag
        })
    
    # Also find self-closing tags
    self_closing_pattern = rf'<{tag_name}(\s[^>]*)?/>'
    self_closing_matches = re.finditer(self_closing_pattern, html_string, flags=re.IGNORECASE)
    
    for match in self_closing_matches:
        full_tag = match.group(0)
        attributes = {}
        
        if match.group(1):
            # Parse attributes
            attr_string = match.group(1).strip()
            attr_pattern = r'(\w+)(?:=(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+)))?'
            
            for attr_match in re.finditer(attr_pattern, attr_string):
                attr_name = attr_match.group(1)
                attr_value = attr_match.group(2) or attr_match.group(3) or attr_match.group(4) or True
                attributes[attr_name] = attr_value
        
        results.append({
            'tag': tag_name,
            'attributes': attributes,
            'content': '',
            'full_tag': full_tag
        })
    
    return results
