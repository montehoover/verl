def evaluate_expression_safely(input_expr):
    try:
        if not input_expr:
            return "Processing Failed!"
        
        # Check for arithmetic keywords
        if input_expr.startswith("SUM:"):
            numbers_str = input_expr[4:].strip()
            numbers = [float(n.strip()) for n in numbers_str.split(',')]
            result = sum(numbers)
            return str(result)
        elif input_expr.startswith("PRODUCT:"):
            numbers_str = input_expr[8:].strip()
            numbers = [float(n.strip()) for n in numbers_str.split(',')]
            result = 1
            for n in numbers:
                result *= n
            return str(result)
        else:
            # Count vowels if no keyword found
            vowels = "aeiouAEIOU"
            count = sum(1 for char in input_expr if char in vowels)
            return str(count)
    except:
        return "Processing Failed!"
