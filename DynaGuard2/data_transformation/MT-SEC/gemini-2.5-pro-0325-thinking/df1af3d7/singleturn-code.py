import re

def parse_user_input(text):
    """
    Divides a string input from a user into individual words and returns these words as a list.

    Args:
        text: The input string provided by the user.

    Returns:
        list: A list containing the words extracted from the input string.

    Raises:
        ValueError: If the input text is not a string or if processing encounters other issues.
    """
    if not isinstance(text, str):
        raise ValueError("Input 'text' must be a string.")
    
    try:
        # \w+ matches one or more word characters (alphanumeric + underscore)
        # This will effectively split by non-word characters and handle punctuation.
        words = re.findall(r'\w+', text)
        return words
    except Exception as e:
        # Catch any unexpected errors during regex processing
        raise ValueError(f"Error processing input: {e}")

if __name__ == '__main__':
    # Example Usage:
    example_input_1 = "This is a simple test sentence."
    output_1 = parse_user_input(example_input_1)
    print(f"Input: \"{example_input_1}\"")
    print(f"Output: {output_1}")
    # Expected: ['This', 'is', 'a', 'simple', 'test', 'sentence']

    example_input_2 = "Hello, world! How are you today?"
    output_2 = parse_user_input(example_input_2)
    print(f"Input: \"{example_input_2}\"")
    print(f"Output: {output_2}")
    # Expected: ['Hello', 'world', 'How', 'are', 'you', 'today']

    example_input_3 = "Python3 is fun_to_learn."
    output_3 = parse_user_input(example_input_3)
    print(f"Input: \"{example_input_3}\"")
    print(f"Output: {output_3}")
    # Expected: ['Python3', 'is', 'fun_to_learn']

    example_input_4 = ""
    output_4 = parse_user_input(example_input_4)
    print(f"Input: \"{example_input_4}\"")
    print(f"Output: {output_4}")
    # Expected: []

    try:
        parse_user_input(123)
    except ValueError as e:
        print(f"Error for non-string input: {e}")
        # Expected: Error for non-string input: Input 'text' must be a string.
