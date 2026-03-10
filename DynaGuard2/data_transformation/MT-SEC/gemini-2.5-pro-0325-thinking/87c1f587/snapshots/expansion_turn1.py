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
