from bs4 import BeautifulSoup

def extract_text_content(html_string):
    """
    Extracts all text content from an HTML string.

    Args:
        html_string: The HTML string to process.

    Returns:
        A string containing all the text content without any HTML tags.
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    return soup.get_text()

def find_html_elements(html_string, tag_name):
    """
    Finds all occurrences of a specific HTML tag in an HTML string.

    Args:
        html_string: The HTML string to process.
        tag_name: The name of the HTML tag to find (e.g., 'p', 'div').

    Returns:
        A list of strings, where each string is an HTML element
        including its attributes.
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    elements = soup.find_all(tag_name)
    return [str(element) for element in elements]

if __name__ == '__main__':
    html_sample = "<p>Hello, <b>world!</b></p><div>This is <span>a test</span>.</div>"
    text_content = extract_text_content(html_sample)
    print(f"Original HTML: {html_sample}")
    print(f"Extracted text: {text_content}")

    html_sample_2 = "<html><head><title>My Page</title></head><body><h1>Welcome</h1><p>This is a paragraph.</p></body></html>"
    text_content_2 = extract_text_content(html_sample_2)
    print(f"Original HTML: {html_sample_2}")
    print(f"Extracted text: {text_content_2}")

    html_sample_3 = """
    <html>
        <body>
            <p class="main">First paragraph.</p>
            <div><p>Nested paragraph.</p></div>
            <p id="last">Last paragraph.</p>
        </body>
    </html>
    """
    p_elements = find_html_elements(html_sample_3, 'p')
    print(f"\nOriginal HTML for find_html_elements: {html_sample_3.strip()}")
    print(f"Found 'p' elements: {p_elements}")

    div_elements = find_html_elements(html_sample_3, 'div')
    print(f"Found 'div' elements: {div_elements}")
