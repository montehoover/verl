def evaluate_expression_safely(input_expr: str) -> str:
    """
    Processes an input string. If it starts with "SUM:" or "PRODUCT:",
    it performs the respective arithmetic operation on the numbers that follow.
    Otherwise, it counts the number of vowels in the string.

    Args:
        input_expr: The string to process.

    Returns:
        The result of the operation or vowel count as a string,
        or "Processing Failed!" if the input is invalid or an error occurs.
    """
    if not isinstance(input_expr, str) or not input_expr:
        return "Processing Failed!"

    try:
        if input_expr.startswith("SUM:"):
            numbers_str = input_expr[len("SUM:"):].strip()
            if not numbers_str: # Handle case like "SUM:" with no numbers
                return "Processing Failed!"
            numbers = [int(n) for n in numbers_str.split()]
            if not numbers: # Handle case like "SUM: " (empty after split)
                return "Processing Failed!"
            return str(sum(numbers))
        elif input_expr.startswith("PRODUCT:"):
            numbers_str = input_expr[len("PRODUCT:"):].strip()
            if not numbers_str: # Handle case like "PRODUCT:" with no numbers
                return "Processing Failed!"
            numbers = [int(n) for n in numbers_str.split()]
            if not numbers: # Handle case like "PRODUCT: " (empty after split)
                return "Processing Failed!"
            result = 1
            for num in numbers:
                result *= num
            return str(result)
        else:
            vowel_count = 0
            vowels = "aeiouAEIOU"
            for char in input_expr:
                if char in vowels:
                    vowel_count += 1
            return str(vowel_count)
    except ValueError: # Handles errors from int() conversion
        return "Processing Failed!"
    except Exception:  # Catch any other unexpected errors during processing
        return "Processing Failed!"
