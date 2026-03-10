import re

def extract_components(text: str) -> list[str]:
    """
    Extracts words, numbers, and special characters from a text string.

    Args:
        text: The input string.

    Returns:
        A list of extracted components (words, numbers, special characters).
    """
    # Regex to find sequences of letters (words), sequences of digits (numbers),
    # or individual special characters (anything not a letter, digit, or whitespace).
    pattern = r"[a-zA-Z]+|\d+|[^a-zA-Z\d\s]"
    components = re.findall(pattern, text)
    return components

if __name__ == '__main__':
    sample_text = "Hello World! 123 testing... #$%^"
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "Python_3.9 is fun, isn't it? Cost: $50."
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")
