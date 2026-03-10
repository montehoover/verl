import unicodedata


def evaluate_expression(expr: str) -> str:
    """
    Capitalize every letter and replace digits with their corresponding English words.
    Allow only alphanumeric characters and spaces; raise ValueError otherwise.

    Examples:
        "abc 123" -> "ABC one two three"
        "NoDigits" -> "NODIGITS"

    :param expr: Input string to process.
    :return: Transformed string with letters uppercased and digits replaced by words.
    :raises ValueError: If any character is not alphanumeric or a space.
    """
    digit_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    def digit_value(ch: str):
        if "0" <= ch <= "9":
            return ord(ch) - ord("0")
        try:
            return unicodedata.decimal(ch)
        except (TypeError, ValueError):
            # Fallback: try general numeric value if it's an integer 0-9
            try:
                num = unicodedata.numeric(ch)
                if int(num) == num and 0 <= int(num) <= 9:
                    return int(num)
            except (TypeError, ValueError):
                pass
        raise ValueError("Input contains a digit that cannot be mapped to an English word.")

    result_parts = []
    last_was_digit = False

    for ch in expr:
        if ch == " ":
            result_parts.append(" ")
            last_was_digit = False
            continue

        if ch.isalnum():
            if ch.isdigit():
                val = digit_value(ch)
                if last_was_digit:
                    result_parts.append(" ")
                result_parts.append(digit_words[val])
                last_was_digit = True
            else:
                result_parts.append(ch.upper())
                last_was_digit = False
        else:
            raise ValueError("Input contains invalid characters; only alphanumeric characters and spaces are allowed.")

    return "".join(result_parts)
