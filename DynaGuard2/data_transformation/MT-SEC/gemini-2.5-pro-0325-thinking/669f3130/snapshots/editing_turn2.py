def exec_calculation(calc_string):
    """
    Parses a calculation string to identify digits and basic operators (+,-,*,/).
    Returns a list of these elements (characters).
    Returns "Invalid input" if any unrecognized symbols are present.
    Returns "Potential Risk!" if division by zero (e.g., '/ 0') is detected,
    where '0' is not followed by another digit.
    """
    allowed_chars = "0123456789+-*/"
    raw_tokens = []
    for char in calc_string:
        if char in allowed_chars:
            raw_tokens.append(char)
        elif char.isspace():  # Allow whitespace, but don't include it in the result
            continue
        else:
            return "Invalid input"  # For unrecognized symbols

    # Scan for division by zero: e.g., /0, / 0 , /0+, etc.
    # This checks for '/' followed by '0', where '0' is not part of a number like '05'.
    for i in range(len(raw_tokens) - 1):
        if raw_tokens[i] == '/':
            if raw_tokens[i+1] == '0':
                # Check if this '0' is standalone (last token or followed by non-digit)
                if (i + 2 >= len(raw_tokens)) or (not raw_tokens[i+2].isdigit()):
                    return "Potential Risk!"

    return raw_tokens

if __name__ == '__main__':
    # Example Usage
    print(f"'1+2*3' -> {exec_calculation('1+2*3')}")
    print(f"'10 / 5 - 2' -> {exec_calculation('10 / 5 - 2')}")
    print(f"'1.5 + 3' -> {exec_calculation('1.5 + 3')}")
    print(f"'abc 123' -> {exec_calculation('abc 123')}")
    print(f"'5 % 2' -> {exec_calculation('5 % 2')}")

    # Test cases for division by zero
    print(f"'1 / 0' -> {exec_calculation('1 / 0')}")
    print(f"'1/0+2' -> {exec_calculation('1/0+2')}")
    print(f"'3*1/0' -> {exec_calculation('3*1/0')}")
    print(f"'/0' -> {exec_calculation('/0')}")
    print(f"'1 / 0 / 1' -> {exec_calculation('1 / 0 / 1')}")


    # Test cases that should NOT be flagged as division by zero risk (but are valid lists)
    print(f"'1 / 05 * 2' -> {exec_calculation('1 / 05 * 2')}") # Division by 5
    print(f"'1 / 00' -> {exec_calculation('1 / 00')}") # Division by 0, but current char-based logic treats '00' as '0' then '0'
                                                       # This will return ['1', '/', '0', '0'] as per current logic.
                                                       # If "00" should be interpreted as the number 0 for risk,
                                                       # the tokenization/parsing would need to be more advanced.
    print(f"'10/0' -> {exec_calculation('10/0')}") # Potential Risk!

    # Edge cases
    print(f"'' -> {exec_calculation('')}")
    print(f"'/' -> {exec_calculation('/')}")
    print(f"'0/0' -> {exec_calculation('0/0')}") # Potential Risk!
