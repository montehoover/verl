import re

def find_urls(text: str) -> list[tuple[str, str]]:
    """
    Scans a string and extracts all URL-like substrings and their schemes.

    Args:
        text: The string to scan for URLs.

    Returns:
        A list of tuples, where each tuple contains (URL, scheme).
        For URLs starting with 'www.', the scheme is assumed to be 'http'.
    """
    # A simple regex to find common URL patterns (http, https, www)
    # This regex looks for:
    # - http:// or https:// followed by non-whitespace characters
    # - www. followed by non-whitespace characters
    # It's a basic pattern and might not catch all valid URLs or might catch some invalid ones.
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    results = []
    found_urls = url_pattern.findall(text)
    
    for url_string in found_urls:
        scheme = None
        if url_string.startswith("https://"):
            scheme = "https"
        elif url_string.startswith("http://"):
            scheme = "http"
        elif url_string.startswith("www."):
            scheme = "http"  # Assume http for www.
        
        if scheme:
            results.append((url_string, scheme))
            
    return results

if __name__ == '__main__':
    sample_text_1 = "Visit our website at http://example.com or www.example.org for more info."
    sample_text_2 = "No URLs here, just plain text."
    sample_text_3 = "Check out https://sub.domain.co.uk/path?query=value#fragment and also ftp://ignored.com"
    sample_text_4 = "A malformed url www. example.com and a good one: http://test.com"

    print(f"URLs in '{sample_text_1}': {find_urls(sample_text_1)}")
    print(f"URLs in '{sample_text_2}': {find_urls(sample_text_2)}")
    print(f"URLs in '{sample_text_3}': {find_urls(sample_text_3)}")
    print(f"URLs in '{sample_text_4}': {find_urls(sample_text_4)}")
