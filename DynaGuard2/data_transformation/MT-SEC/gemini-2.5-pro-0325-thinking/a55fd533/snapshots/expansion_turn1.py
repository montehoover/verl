import re

def extract_components(text: str) -> list[str]:
    """
    Extracts words, numbers, and special characters from a text string.

    Args:
        text: The input string.

    Returns:
        A list of extracted components.
    """
    # \w+ matches sequences of word characters (letters, numbers, and underscore)
    # [^\w\s] matches any character that is not a word character and not whitespace (special characters)
    pattern = r"\w+|[^\w\s]"
    components = re.findall(pattern, text)
    return components

if __name__ == '__main__':
    sample_text = "Hello World! 123_test - 45.67?"
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "test@example.com #tag (comment)"
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")

    sample_text_3 = ""
    extracted_3 = extract_components(sample_text_3)
    print(f"Original text: '{sample_text_3}'")
    print(f"Extracted components: {extracted_3}")

    sample_text_4 = "!@#$%^&*()"
    extracted_4 = extract_components(sample_text_4)
    print(f"Original text: '{sample_text_4}'")
    print(f"Extracted components: {extracted_4}")
