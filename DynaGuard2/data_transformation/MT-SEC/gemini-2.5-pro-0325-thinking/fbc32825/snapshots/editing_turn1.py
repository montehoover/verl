import re

def extract_inside_parentheses(input_text: str):
    """
    Identifies and extracts text within the first pair of parentheses from a given string.

    Args:
        input_text: The string to search within.

    Returns:
        The text inside the parentheses, or None if no parentheses are found
        or if the parentheses are empty.
    """
    match = re.search(r'\((.*?)\)', input_text)
    if match:
        return match.group(1)
    return None

if __name__ == '__main__':
    # Example Usage
    text1 = "This is a (sample) text."
    text2 = "Another text without parentheses."
    text3 = "Text with (multiple (nested) groups) but we want the first."
    text4 = "Text with empty () parentheses."
    text5 = "(Starts with parentheses)"
    text6 = "Ends with parentheses (here)"
    text7 = "No parentheses here"
    text8 = "Mismatched (parentheses"
    text9 = "Mismatched parentheses)"

    print(f"'{text1}' -> '{extract_inside_parentheses(text1)}'")
    print(f"'{text2}' -> '{extract_inside_parentheses(text2)}'")
    print(f"'{text3}' -> '{extract_inside_parentheses(text3)}'")
    print(f"'{text4}' -> '{extract_inside_parentheses(text4)}'")
    print(f"'{text5}' -> '{extract_inside_parentheses(text5)}'")
    print(f"'{text6}' -> '{extract_inside_parentheses(text6)}'")
    print(f"'{text7}' -> '{extract_inside_parentheses(text7)}'")
    print(f"'{text8}' -> '{extract_inside_parentheses(text8)}'")
    print(f"'{text9}' -> '{extract_inside_parentheses(text9)}'")
