import re

def get_html_tags(html_input: str) -> list:
    # Regular expression pattern to match HTML tags
    # This pattern captures tags like <tag>, </tag>, and self-closing tags like <tag/>
    pattern = r'<[^>]+>'
    
    # Find all matches in the input string
    tags = re.findall(pattern, html_input)
    
    return tags
