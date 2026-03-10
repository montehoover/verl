import re

def extract_components(text: str) -> list:
    """
    Extracts words, numbers, and special characters from a text string.

    Args:
        text: The input string.

    Returns:
        A list of extracted components (words, numbers, special characters).
    """
    # Regex to find words (sequences of letters), numbers (sequences of digits),
    # or any single non-whitespace character that isn't a letter or digit (special characters).
    pattern = re.compile(r"([a-zA-Z]+|\d+|[^a-zA-Z\d\s])")
    components = pattern.findall(text)
    return components

if __name__ == '__main__':
    sample_text = "Hello World! 123 testing... #tag @mention"
    extracted = extract_components(sample_text)
    print(f"Original text: '{sample_text}'")
    print(f"Extracted components: {extracted}")

    sample_text_2 = "Python 3.9 is great, isn't it?"
    extracted_2 = extract_components(sample_text_2)
    print(f"Original text: '{sample_text_2}'")
    print(f"Extracted components: {extracted_2}")

    sample_text_3 = "$%^&*()"
    extracted_3 = extract_components(sample_text_3)
    print(f"Original text: '{sample_text_3}'")
    print(f"Extracted components: {extracted_3}")

    sample_text_4 = "NoSpecialCharactersOrNumbers"
    extracted_4 = extract_components(sample_text_4)
    print(f"Original text: '{sample_text_4}'")
    print(f"Extracted components: {extracted_4}")

    sample_text_5 = "12345"
    extracted_5 = extract_components(sample_text_5)
    print(f"Original text: '{sample_text_5}'")
    print(f"Extracted components: {extracted_5}")
