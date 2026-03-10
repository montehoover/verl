import re
from html.parser import HTMLParser

def tag_exists(html_string, tag):
    """
    Check if a specific HTML tag exists in the given string.
    
    Args:
        html_string (str): The HTML content to search in
        tag (str): The tag name to search for (without brackets)
    
    Returns:
        bool: True if the tag exists, False otherwise
    """
    # Create a pattern to match both opening and self-closing tags
    pattern = f'<{tag}(?:\\s|>|/)'
    return bool(re.search(pattern, html_string, re.IGNORECASE))


class TagExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = {}
        self.tag_stack = []
        self.current_data = []
        
    def handle_starttag(self, tag, attrs):
        self.tag_stack.append({
            'tag': tag,
            'attrs': dict(attrs),
            'content': [],
            'start_pos': self.getpos()
        })
        
    def handle_endtag(self, tag):
        if self.tag_stack and self.tag_stack[-1]['tag'] == tag:
            tag_info = self.tag_stack.pop()
            content = ''.join(tag_info['content'])
            
            if tag not in self.tags:
                self.tags[tag] = []
                
            self.tags[tag].append({
                'attributes': tag_info['attrs'],
                'content': content.strip(),
                'start_pos': tag_info['start_pos'],
                'end_pos': self.getpos()
            })
            
            # Add this tag's full content to parent if exists
            if self.tag_stack:
                full_tag = self.get_full_tag_string(tag, tag_info['attrs'], content)
                self.tag_stack[-1]['content'].append(full_tag)
                
    def handle_data(self, data):
        if self.tag_stack:
            self.tag_stack[-1]['content'].append(data)
            
    def handle_startendtag(self, tag, attrs):
        if tag not in self.tags:
            self.tags[tag] = []
            
        self.tags[tag].append({
            'attributes': dict(attrs),
            'content': '',
            'start_pos': self.getpos(),
            'end_pos': self.getpos()
        })
        
    def get_full_tag_string(self, tag, attrs, content):
        attr_str = ' '.join([f'{k}="{v}"' for k, v in attrs.items()])
        if attr_str:
            return f'<{tag} {attr_str}>{content}</{tag}>'
        return f'<{tag}>{content}</{tag}>'


def extract_tag_contents(html_string):
    """
    Extract all tags and their contents from an HTML string.
    
    Args:
        html_string (str): The HTML content to parse
        
    Returns:
        dict: A dictionary where keys are tag names and values are lists of dictionaries
              containing 'attributes', 'content', 'start_pos', and 'end_pos' for each occurrence
    """
    parser = TagExtractor()
    parser.feed(html_string)
    return parser.tags


def get_html_tags(html_input):
    """
    Extract all HTML tags from the input string using regular expressions.
    
    Args:
        html_input (str): The HTML-formatted input string
        
    Returns:
        list: A list of tag names found in the input
    """
    # Pattern to match HTML tags (opening, closing, and self-closing)
    # Captures the tag name after < or </
    pattern = r'</?(\w+)(?:\s[^>]*)?/?>'
    
    # Find all matches
    matches = re.findall(pattern, html_input)
    
    # Return unique tags while preserving order
    seen = set()
    result = []
    for tag in matches:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    
    return result
