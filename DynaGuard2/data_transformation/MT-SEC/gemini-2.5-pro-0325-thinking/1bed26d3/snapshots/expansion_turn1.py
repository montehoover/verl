import re

def extract_components(text: str) -> list:
    """
    Extracts words, numbers, or symbols from the given text using regex.

    Args:
        text: The input string.

    Returns:
        A list of extracted components.
    """
    # \w+ matches one or more word characters (alphanumeric + underscore)
    # \S matches any non-whitespace character (catches symbols)
    pattern = r"\w+|\S"
    components = re.findall(pattern, text)
    return components

if __name__ == '__main__':
    sample_text = "Hello, world! This is a test with 123 numbers & symbols like $ and %."
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "Another-example_with_various.chars"
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")

    sample_text_3 = "  Leading and trailing spaces  "
    extracted_3 = extract_components(sample_text_3)
    print(f"Original text: '{sample_text_3}'")
    print(f"Extracted components: {extracted_3}")

    sample_text_4 = ""
    extracted_4 = extract_components(sample_text_4)
    print(f"Original text: '{sample_text_4}'")
    print(f"Extracted components: {extracted_4}")
