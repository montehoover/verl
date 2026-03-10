import re

def evaluate_expression(expr: str) -> str:
    """
    Capitalizes every letter and transforms digits into their corresponding
    English words in an expression string. For example, '123' becomes 'one two three'.

    Args:
        expr: The input string expression.

    Returns:
        The transformed string with capitalized letters and digits as words.

    Raises:
        ValueError: If the input string contains characters that are not
                    alphanumeric or spaces.
    """
    if not re.match(r"^[a-zA-Z0-9\s]*$", expr):
        raise ValueError("Input string can only contain alphanumeric characters and spaces.")

    digit_to_word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    result = []
    last_appended_was_digit_word = False
    last_appended_was_letter = False

    for char_val in expr:
        if char_val.isspace():
            result.append(' ')
            # Reset flags when a space is encountered, as it breaks contiguity
            last_appended_was_digit_word = False
            last_appended_was_letter = False
        elif char_val.isalpha():
            # Current character is a letter
            if last_appended_was_digit_word: # If previous significant char was a digit-word
                result.append(' ')
            result.append(char_val.upper())
            last_appended_was_digit_word = False
            last_appended_was_letter = True
        elif char_val.isdigit():
            # Current character is a digit
            word = digit_to_word[char_val]
            # Add a space if the previous significant char was a letter OR another digit-word
            if last_appended_was_letter or last_appended_was_digit_word:
                result.append(' ')
            
            result.append(word)
            last_appended_was_digit_word = True
            last_appended_was_letter = False
            
    return "".join(result)
