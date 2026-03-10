import re

def match_strings(text: str):
    """
    Uses regular expressions to match a string and capture the text
    outside the parentheses and the text inside the parentheses.

    Args:
        text: The string to search within.

    Returns:
        A tuple containing the text outside the parentheses and the text
        inside if the pattern is matched. Otherwise, returns None.
        The text outside is a concatenation of text before and after parentheses.
    """
    match = re.search(r'(.*?)\((.*?)\)(.*)', text, re.DOTALL)
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

    print(f"'{text1}' -> {match_strings(text1)}")
    print(f"'{text2}' -> {match_strings(text2)}")
    print(f"'{text3}' -> {match_strings(text3)}") # Will capture 'multiple (nested'
    print(f"'{text4}' -> {match_strings(text4)}")
    print(f"'{text5}' -> {match_strings(text5)}")
    print(f"'{text6}' -> {match_strings(text6)}")
    print(f"'{text7}' -> {match_strings(text7)}")
    print(f"'{text8}' -> {match_strings(text8)}")
    print(f"'{text9}' -> {match_strings(text9)}")
