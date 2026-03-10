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

if __name__ == '__main__':
    # Example Usage
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
