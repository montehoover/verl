import re
from html.parser import HTMLParser
from collections import defaultdict

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


class AttributeExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        # Stores tag_name -> list_of_attr_names
        self.extracted_attrs = defaultdict(list)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        # attrs is a list of (name, value) tuples. e.g., [('class', 'foo'), ('id', 'bar')]
        # For a given tag like 'div', if it appears multiple times with different attributes,
        # we want to collect all unique attribute names associated with 'div'.
        for attr_name, _ in attrs:
            if attr_name not in self.extracted_attrs[tag]:
                self.extracted_attrs[tag].append(attr_name)

def extract_html_attributes(html_content: str) -> dict[str, list[str]]:
    """
    Extracts attributes from HTML elements.

    Args:
        html_content: The HTML content as a string.

    Returns:
        A dictionary where keys are tag names and values are lists of
        unique attribute names found for those tags across the document.
        For example: {'div': ['class', 'id'], 'span': ['data-id']}
    """
    parser = AttributeExtractor()
    parser.feed(html_content)
    # Convert defaultdict to a regular dict for the output
    return dict(parser.extracted_attrs)


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

    print(f"\n--- Extracting Attributes ---")
    html_for_attrs_1 = """
    <div class="main" id="page-container">
        <h1>Welcome</h1>
        <p style="color: blue;">This is a paragraph.</p>
        <div class="secondary" data-info="extra">
            <span class="highlight" data-info="specific">Important</span>
        </div>
        <img src="image.png" alt="An image" />
        <input type="text" name="username" disabled />
    </div>
    """
    attributes_found_1 = extract_html_attributes(html_for_attrs_1)
    print(f"Attributes in HTML 1: {attributes_found_1}")
    # Expected example: {'div': ['class', 'id', 'data-info'], 'h1': [], 'p': ['style'], 'span': ['class', 'data-info'], 'img': ['src', 'alt'], 'input': ['type', 'name', 'disabled']}

    html_for_attrs_2 = """
    <a href="#">Link 1</a>
    <a href="/path" title="Go to Path">Link 2</a>
    <custom-tag my-attr="value"></custom-tag>
    <br>
    <hr noshade>
    """
    attributes_found_2 = extract_html_attributes(html_for_attrs_2)
    print(f"Attributes in HTML 2: {attributes_found_2}")
    # Expected example: {'a': ['href', 'title'], 'custom-tag': ['my-attr'], 'br': [], 'hr': ['noshade']}
