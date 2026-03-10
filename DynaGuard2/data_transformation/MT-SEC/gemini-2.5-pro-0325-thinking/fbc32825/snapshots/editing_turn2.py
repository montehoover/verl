import re

def capture_text_parts(input_text: str):
    """
    Captures text outside and inside the first pair of parentheses from a given string.

    Args:
        input_text: The string to search within.

    Returns:
        A tuple (text_outside, text_inside) if parentheses are found.
        'text_outside' is the concatenation of text before and after the parentheses.
        'text_inside' is the text within the parentheses.
        Returns None if no parentheses are found or if the pattern doesn't match.
    """
    match = re.search(r'(.*?)\((.*?)\)(.*)', input_text, re.DOTALL)
    if match:
        text_before = match.group(1)
        text_inside = match.group(2)
        text_after = match.group(3)
        text_outside = text_before + text_after
        return (text_outside, text_inside)
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
    text8 = "Mismatched (parentheses" # This will not match as regex expects closing parenthesis
    text9 = "Mismatched parentheses)" # This will not match as regex expects opening parenthesis

    print(f"'{text1}' -> {capture_text_parts(text1)}")
    print(f"'{text2}' -> {capture_text_parts(text2)}")
    print(f"'{text3}' -> {capture_text_parts(text3)}") # Will capture 'multiple (nested'
    print(f"'{text4}' -> {capture_text_parts(text4)}")
    print(f"'{text5}' -> {capture_text_parts(text5)}")
    print(f"'{text6}' -> {capture_text_parts(text6)}")
    print(f"'{text7}' -> {capture_text_parts(text7)}")
    print(f"'{text8}' -> {capture_text_parts(text8)}")
    print(f"'{text9}' -> {capture_text_parts(text9)}")
