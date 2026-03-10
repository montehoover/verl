import re

def count_html_tags(html_content: str) -> int:
    """
    Counts the number of HTML tags in a given string.

    Args:
        html_content: The string containing HTML content.

    Returns:
        The count of HTML tags as an integer.
        Returns 0 if any error occurs during processing.
    """
    if not isinstance(html_content, str):
        return 0
    try:
        # A simple regex to find HTML tags (e.g., <tag>, <tag attr="value">, </tag>)
        # This regex matches anything that starts with <, ends with >, and has one or more characters in between.
        # It also handles self-closing tags like <br />
        tags = re.findall(r"<[^>]+>", html_content)
        return len(tags)
    except Exception:
        # In case of any unexpected error during regex processing or other issues
        return 0

def extract_unique_tags(html_content: str) -> list[str]:
    """
    Extracts unique HTML tag names from a given string.

    Args:
        html_content: The string containing HTML content.

    Returns:
        A list of unique HTML tag names (e.g., ['html', 'head', 'title', 'body', 'h1', 'p', 'br']).
        Returns an empty list if any error occurs or if the input is invalid.
    """
    if not isinstance(html_content, str):
        return []
    try:
        # Regex to find tag names. It captures the first word character(s) after a '<'
        # It handles opening tags like <p>, <div class="foo"> and self-closing tags like <br />
        # It ignores closing tags like </p> for the purpose of tag name extraction.
        # Tag names are converted to lowercase for uniqueness.
        # Example: <Html>, <HTML>, <html> will all be 'html'
        # Example: <P class="foo"> will be 'p'
        # Example: <br /> will be 'br'
        # Example: </p> will not be matched by this specific group capture for tag name
        
        # A regex to find opening or self-closing tags and capture the tag name.
        # <([a-zA-Z][a-zA-Z0-9]*)
        # This captures the tag name itself, e.g., "html", "p", "br"
        # It avoids capturing anything from closing tags like "/p"
        found_tags = re.findall(r"<([a-zA-Z][a-zA-Z0-9]*)(?:>|\s[^>]*>)", html_content)
        
        # Normalize to lowercase and get unique tags
        unique_tags = sorted(list(set(tag.lower() for tag in found_tags)))
        return unique_tags
    except Exception:
        # In case of any unexpected error during regex processing or other issues
        return []

if __name__ == '__main__':
    # Example Usage for count_html_tags
    sample_html_1 = "<html><head><title>Test</title></head><body><h1>Hello</h1><p>World</p><br /></body></html>"
    sample_html_2 = "<div class='test'><img src='image.png'><span>Text</span></div>"
    sample_html_3 = "This is a string with no HTML tags."
    sample_html_4 = "<p>One tag.</p>"
    sample_html_5 = ""
    sample_html_6 = "<incomplete tag" # Invalid HTML
    sample_html_7 = None # Invalid input type

    print(f"'{sample_html_1[:30]}...' has {count_html_tags(sample_html_1)} tags.")
    print(f"'{sample_html_2[:30]}...' has {count_html_tags(sample_html_2)} tags.")
    print(f"'{sample_html_3[:30]}...' has {count_html_tags(sample_html_3)} tags.")
    print(f"'{sample_html_4[:30]}...' has {count_html_tags(sample_html_4)} tags.")
    print(f"'{sample_html_5[:30]}...' has {count_html_tags(sample_html_5)} tags.")
    print(f"'{sample_html_6[:30]}...' has {count_html_tags(sample_html_6)} tags.")
    print(f"'None' has {count_html_tags(sample_html_7)} tags.")

    print("\n--- extract_unique_tags examples ---")
    # Example Usage for extract_unique_tags
    sample_html_8 = "<Html><Head><Title>Test</Title></Head><Body><H1>Hello</H1><P>World</P><br /><P>Another paragraph.</P></Body></Html>"
    sample_html_9 = "<DIV class='test'><IMG src='image.png'><SPAN>Text</SPAN></DIV><img src='another.gif'>" # Mixed case and duplicate img
    sample_html_10 = "No tags here."
    sample_html_11 = ""
    sample_html_12 = "<incomplete"
    sample_html_13 = None

    print(f"Unique tags in '{sample_html_8[:30]}...': {extract_unique_tags(sample_html_8)}")
    print(f"Unique tags in '{sample_html_9[:30]}...': {extract_unique_tags(sample_html_9)}")
    print(f"Unique tags in '{sample_html_10[:30]}...': {extract_unique_tags(sample_html_10)}")
    print(f"Unique tags in '{sample_html_11[:30]}...': {extract_unique_tags(sample_html_11)}")
    print(f"Unique tags in '{sample_html_12[:30]}...': {extract_unique_tags(sample_html_12)}")
    print(f"Unique tags in 'None': {extract_unique_tags(sample_html_13)}")
    print(f"Unique tags in '{sample_html_1[:30]}...': {extract_unique_tags(sample_html_1)}") # Using an old sample
