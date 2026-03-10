def evaluate_expression_safely(input_expr):
    """
    Counts the number of vowels in the given input_expr string.
    Returns the count as a string.
    If input_expr is empty or invalid, returns "Processing Failed!".
    """
    try:
        if not isinstance(input_expr, str):
            return "Processing Failed!"
        if input_expr == "":
            return "Processing Failed!"

        vowels = set("aeiouAEIOU")
        count = sum(1 for ch in input_expr if ch in vowels)
        return str(count)
    except Exception:
        return "Processing Failed!"
