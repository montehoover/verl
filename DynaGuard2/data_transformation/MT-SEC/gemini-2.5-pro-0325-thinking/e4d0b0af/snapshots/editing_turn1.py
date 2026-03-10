def evaluate_expression_safely(input_expr: str) -> str:
    """
    Counts the number of vowels in the input string.

    Args:
        input_expr: The string to process.

    Returns:
        The count of vowels as a string, or "Processing Failed!"
        if the input is empty, None, or not a string.
    """
    if not isinstance(input_expr, str) or not input_expr:
        return "Processing Failed!"

    vowel_count = 0
    vowels = "aeiouAEIOU"
    try:
        for char in input_expr:
            if char in vowels:
                vowel_count += 1
        return str(vowel_count)
    except Exception:  # Catch any unexpected errors during processing
        return "Processing Failed!"
