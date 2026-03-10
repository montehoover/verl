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

if __name__ == '__main__':
    html_sample = "<p>Hello, <b>world!</b></p><div>This is <span>a test</span>.</div>"
    text_content = extract_text_content(html_sample)
    print(f"Original HTML: {html_sample}")
    print(f"Extracted text: {text_content}")

    html_sample_2 = "<html><head><title>My Page</title></head><body><h1>Welcome</h1><p>This is a paragraph.</p></body></html>"
    text_content_2 = extract_text_content(html_sample_2)
    print(f"Original HTML: {html_sample_2}")
    print(f"Extracted text: {text_content_2}")
