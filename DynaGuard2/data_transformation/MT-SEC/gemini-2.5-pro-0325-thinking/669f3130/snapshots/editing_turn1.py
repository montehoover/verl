def exec_calculation(calc_string):
    """
    Identifies digits and basic operators (+,-,*,/) in a string,
    returning them in a list.
    Returns "Invalid input" if any unrecognized symbols are present.
    """
    allowed_chars = "0123456789+-*/"
    result = []
    for char in calc_string:
        if char in allowed_chars:
            result.append(char)
        elif char.isspace():  # Allow whitespace, but don't include it in the result
            continue
        else:
            return "Invalid input"
    return result

if __name__ == '__main__':
    # Example Usage
    print(exec_calculation("1+2*3"))
    print(exec_calculation("10 / 5 - 2"))
    print(exec_calculation("1.5 + 3")) # This will be invalid because '.' is not allowed yet
    print(exec_calculation("abc 123"))
    print(exec_calculation("5 % 2"))
