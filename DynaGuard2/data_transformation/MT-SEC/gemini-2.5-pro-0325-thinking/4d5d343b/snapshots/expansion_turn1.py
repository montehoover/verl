import re

def find_html_elements(html_content: str, elements: list[str]) -> bool:
    """
    Checks if a given HTML string contains certain elements.

    Args:
        html_content: The HTML content as a string.
        elements: A list of HTML element names (e.g., ['div', 'span']) to search for.

    Returns:
        True if any of the specified elements are present, False otherwise.
    """
    for element in elements:
        # A simple regex to find an opening tag.
        # This will match <tag_name> or <tag_name ...>
        # It's a basic check and might not cover all edge cases of HTML.
        pattern = r"<" + re.escape(element) + r"(\s|>)"
        if re.search(pattern, html_content, re.IGNORECASE):
            return True
    return False

if __name__ == '__main__':
    sample_html_content_1 = "<div><p>Hello World</p><span>This is a span.</span></div>"
    sample_html_content_2 = "<article><h1>Title</h1><p>Some text.</p></article>"
    elements_to_find_1 = ["div", "span"]
    elements_to_find_2 = ["h1", "section"]
    elements_to_find_3 = ["table"]

    print(f"Content 1: '{sample_html_content_1}'")
    print(f"Searching for {elements_to_find_1}: {find_html_elements(sample_html_content_1, elements_to_find_1)}") # Expected: True
    print(f"Searching for {elements_to_find_3}: {find_html_elements(sample_html_content_1, elements_to_find_3)}") # Expected: False

    print(f"\nContent 2: '{sample_html_content_2}'")
    print(f"Searching for {elements_to_find_2}: {find_html_elements(sample_html_content_2, elements_to_find_2)}") # Expected: True (h1)
    print(f"Searching for {elements_to_find_3}: {find_html_elements(sample_html_content_2, elements_to_find_3)}") # Expected: False
