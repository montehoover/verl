def evaluate_expression_safely(input_expr):
    try:
        if not input_expr:
            return "Processing Failed!"
        
        vowels = "aeiouAEIOU"
        count = sum(1 for char in input_expr if char in vowels)
        return str(count)
    except:
        return "Processing Failed!"
