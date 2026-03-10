import re

def analyze_user_string(input_text: str) -> list:
    """
    Divides a string input from a user into individual words and returns these words as a list.

    Args:
        input_text: The input string provided by the user.

    Returns:
        A list containing the words extracted from the input string.

    Raises:
        ValueError: If the input_text is not a string or if processing encounters any issues.
    """
    if not isinstance(input_text, str):
        raise ValueError("Input must be a string.")
    
    try:
        # Use \w+ to match sequences of word characters (letters, numbers, and underscore)
        # This is a common way to tokenize, but might need adjustment based on specific needs
        # (e.g., handling punctuation, hyphens differently).
        words = re.findall(r'\w+', input_text)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error processing input string: {e}")

if __name__ == '__main__':
    # Example Usage
    sample_text_1 = "Hello world, this is a test!"
    sample_text_2 = "Another-example with numbers 123 and symbols #$%"
    sample_text_3 = ""
    sample_text_4 = "   Leading and trailing spaces   "
    invalid_input = 12345

    print(f"Input: '{sample_text_1}'")
    try:
        print(f"Output: {analyze_user_string(sample_text_1)}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)

    print(f"Input: '{sample_text_2}'")
    try:
        print(f"Output: {analyze_user_string(sample_text_2)}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)

    print(f"Input: '{sample_text_3}' (empty string)")
    try:
        print(f"Output: {analyze_user_string(sample_text_3)}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)
    
    print(f"Input: '{sample_text_4}'")
    try:
        print(f"Output: {analyze_user_string(sample_text_4)}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)

    print(f"Input: {invalid_input} (invalid type)")
    try:
        print(f"Output: {analyze_user_string(invalid_input)}")
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)
