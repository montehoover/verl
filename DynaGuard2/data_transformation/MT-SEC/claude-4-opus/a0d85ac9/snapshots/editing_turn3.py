def evaluate_math_expression(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check for invalid characters (now including logical operation characters)
        allowed_chars = "0123456789+-*/().TrueFalseandornot"
        for char in expression:
            if char not in allowed_chars:
                return "Invalid Expression!"
        
        # Evaluate the expression
        result = eval(expression)
        return result
    except:
        return "Invalid Expression!"

def calculate_user_input(input_expr):
    try:
        # Remove whitespace
        input_expr = input_expr.replace(" ", "")
        
        # Check for invalid characters
        allowed_chars = "0123456789+-*/().TrueFalseandornot"
        for char in input_expr:
            if char not in allowed_chars:
                return "Processing Failed!"
        
        # Evaluate the expression
        result = eval(input_expr)
        return result
    except:
        return "Processing Failed!"
