import re

def find_html_tags(html_content: str) -> list:
    # Regular expression pattern to match HTML tags
    # Matches opening tags, closing tags, and self-closing tags
    pattern = r'<[^>]+>'
    
    # Find all matches in the html_content
    tags = re.findall(pattern, html_content)
    
    return tags
