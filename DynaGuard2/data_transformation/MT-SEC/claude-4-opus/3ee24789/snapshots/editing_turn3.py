import re

def detect_html_tags(html_str: str) -> list[str]:
    # Regular expression pattern to match HTML tags
    tag_pattern = r'<[^>]+>'
    
    # Find all HTML tags in the content
    tags = re.findall(tag_pattern, html_str)
    
    return tags
