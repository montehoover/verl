from bs4 import BeautifulSoup

def extract_html_elements(html_content: str, tag_name: str) -> list:
    """
    Extracts all occurrences of a specific HTML tag from a string of HTML content.

    Args:
        html_content: The string containing the HTML content.
        tag_name: The name of the HTML tag to extract (e.g., 'p', 'div', 'a').

    Returns:
        A list of BeautifulSoup Tag objects representing all occurrences of the specified tag.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    elements = soup.find_all(tag_name)
    return elements

def extract_nested_html(html_content: str, parent_tag_name: str) -> list:
    """
    Extracts all nested elements within the first occurrence of a specified parent HTML tag.

    Args:
        html_content: The string containing the HTML content.
        parent_tag_name: The name of the parent HTML tag (e.g., 'div', 'ul').

    Returns:
        A list of BeautifulSoup Tag objects representing all nested elements.
        Returns an empty list if the parent tag is not found.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    parent_element = soup.find(parent_tag_name)
    if parent_element:
        # find_all(True) finds all descendant tags
        nested_elements = parent_element.find_all(True)
        return nested_elements
    return []

if __name__ == '__main__':
    sample_html = """
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Main Heading</h1>
        <p>This is the first paragraph.</p>
        <div>
            <p>This is a nested paragraph.</p>
        </div>
        <p>This is the second paragraph.</p>
        <a href="#">Link 1</a>
        <a href="#">Link 2</a>
    </body>
    </html>
    """

    # Example usage:
    paragraphs = extract_html_elements(sample_html, 'p')
    print(f"Found {len(paragraphs)} paragraph elements:")
    for p in paragraphs:
        print(p)

    print("\n" + "="*20 + "\n")

    links = extract_html_elements(sample_html, 'a')
    print(f"Found {len(links)} anchor elements:")
    for a in links:
        print(a)

    print("\n" + "="*20 + "\n")

    divs = extract_html_elements(sample_html, 'div')
    print(f"Found {len(divs)} div elements:")
    for d in divs:
        print(d)

    print("\n" + "="*20 + "\n")

    # Example usage for extract_nested_html:
    nested_in_div = extract_nested_html(sample_html, 'div')
    print(f"Found {len(nested_in_div)} nested elements inside the first 'div':")
    for elem in nested_in_div:
        print(elem)

    nested_in_body = extract_nested_html(sample_html, 'body')
    print(f"\nFound {len(nested_in_body)} nested elements inside 'body':")
    for elem in nested_in_body:
        print(elem)

    nested_in_nonexistent = extract_nested_html(sample_html, 'nonexistenttag')
    print(f"\nFound {len(nested_in_nonexistent)} nested elements inside 'nonexistenttag':")
    for elem in nested_in_nonexistent:
        print(elem)
