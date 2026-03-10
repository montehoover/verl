import re

def contains_html(text: str) -> bool:
    """
    Checks if a given string contains any HTML-like content.

    Args:
        text: The string to check.

    Returns:
        True if a suspected HTML tag (e.g., '<tag>' or '</tag>') is found, False otherwise.
    """
    # Regular expression to find simple HTML tags like <tag> or </tag>
    # It looks for:
    # <       : an opening angle bracket
    # /?      : an optional forward slash (for closing tags)
    # [a-zA-Z0-9]+ : one or more alphanumeric characters (for the tag name)
    # >       : a closing angle bracket
    html_tag_pattern = re.compile(r"<[/a-zA-Z0-9]+>")
    
    if html_tag_pattern.search(text):
        return True
    return False

def extract_tag_attributes(tag_string: str) -> dict:
    """
    Extracts attributes from a single HTML tag string.

    Args:
        tag_string: The HTML tag string (e.g., '<a href="#" class="link">').

    Returns:
        A dictionary of attribute-value pairs.
        Example: {'href': '#', 'class': 'link'}
    """
    attributes = {}
    # Regex to find attribute name-value pairs
    # It looks for:
    # (\w+)      : an attribute name (alphanumeric characters and underscore)
    # \s*=\s*    : an equals sign, possibly surrounded by whitespace
    # (?:        : start of a non-capturing group for the value
    #   "([^"]*)" : a value enclosed in double quotes
    #   |         : OR
    #   '([^']*)' : a value enclosed in single quotes
    #   |         : OR
    #   ([^>\s]+) : a value without quotes (any character except > or whitespace)
    # )          : end of the non-capturing group
    # The re.findall will return a list of tuples, where each tuple contains
    # the attribute name and then three possible value captures (double-quoted, single-quoted, unquoted).
    # We'll need to pick the one that matched.
    attribute_pattern = re.compile(r'\s+([a-zA-Z0-9_-]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^>\s]+))')
    
    # Remove the tag itself, e.g., <div ... > -> ...
    # First, find the first space to isolate the tag name
    first_space_index = tag_string.find(' ')
    if first_space_index == -1: # No attributes
        return {}
        
    attributes_part = tag_string[first_space_index:].strip()
    if attributes_part.endswith('>'):
        attributes_part = attributes_part[:-1].strip()
    if attributes_part.endswith('/>'): # self-closing tag
        attributes_part = attributes_part[:-2].strip()

    matches = attribute_pattern.findall(attributes_part)
    
    for match in matches:
        name = match[0]
        # The value will be in one of match[1], match[2], or match[3]
        value = match[1] if match[1] else match[2] if match[2] else match[3]
        attributes[name] = value
        
    return attributes

if __name__ == '__main__':
    # Example Usage for contains_html
    string1 = "This is a  обычный текст."
    string2 = "This string contains <html> tags."
    string3 = "Another example with a <p>paragraph</p>."
    string4 = "No tags here."
    string5 = "<justatag>"
    string6 = "</closingtag>"
    string7 = "<tag with attributes>" # This simple regex won't catch attributes well but will detect the tag
    string8 = "Text with <incomplete"
    string9 = "Text with incomplete>"
    string10 = "<>"
    string11 = "< >"
    string12 = "<a-1>"

    print(f"'{string1}': {contains_html(string1)}")
    print(f"'{string2}': {contains_html(string2)}")
    print(f"'{string3}': {contains_html(string3)}")
    print(f"'{string4}': {contains_html(string4)}")
    print(f"'{string5}': {contains_html(string5)}")
    print(f"'{string6}': {contains_html(string6)}")
    print(f"'{string7}': {contains_html(string7)}")
    print(f"'{string8}': {contains_html(string8)}")
    print(f"'{string9}': {contains_html(string9)}")
    print(f"'{string10}': {contains_html(string10)}") # Expected: False (no tag name)
    print(f"'{string11}': {contains_html(string11)}")# Expected: False (space not allowed in simple tag name by this regex)
    print(f"'{string12}': {contains_html(string12)}")# Expected: True

    print("\n--- extract_tag_attributes examples ---")
    tag1 = "<a href='#' class=\"link\">"
    tag2 = "<img src=/image.png alt=MyImage>"
    tag3 = "<input type='text' disabled>"
    tag4 = "<div id=main-content data-value='{\"key\": \"value\"}'>"
    tag5 = "<span >" # No attributes
    tag6 = "<p class=text-center style='color: blue; font-size: 12px;'>"
    tag7 = "<meta charset=UTF-8 />"
    tag8 = "<custom-tag my-attr='test-value' another=\"val\">"

    print(f"Attributes in '{tag1}': {extract_tag_attributes(tag1)}")
    print(f"Attributes in '{tag2}': {extract_tag_attributes(tag2)}")
    print(f"Attributes in '{tag3}': {extract_tag_attributes(tag3)}")
    print(f"Attributes in '{tag4}': {extract_tag_attributes(tag4)}")
    print(f"Attributes in '{tag5}': {extract_tag_attributes(tag5)}")
    print(f"Attributes in '{tag6}': {extract_tag_attributes(tag6)}")
    print(f"Attributes in '{tag7}': {extract_tag_attributes(tag7)}")
    print(f"Attributes in '{tag8}': {extract_tag_attributes(tag8)}")
