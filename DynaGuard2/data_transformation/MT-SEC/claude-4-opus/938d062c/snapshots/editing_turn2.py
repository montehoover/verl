import re

def simple_calculator(expression):
    # If expression is already in the old format (3 arguments), handle it
    if isinstance(expression, (int, float)):
        raise ValueError("New format expects a string expression")
    
    # Remove spaces
    expression = expression.replace(' ', '')
    
    # Function to evaluate expression without parentheses
    def evaluate_simple(expr):
        # Handle negative numbers at the start
        if expr.startswith('-'):
            expr = '0' + expr
        
        # First handle multiplication and division
        while '*' in expr or '/' in expr:
            # Find multiplication or division
            match = re.search(r'(-?\d+\.?\d*)\s*([\*/])\s*(-?\d+\.?\d*)', expr)
            if match:
                num1 = float(match.group(1))
                operator = match.group(2)
                num2 = float(match.group(3))
                
                if operator == '*':
                    result = num1 * num2
                else:  # operator == '/'
                    if num2 == 0:
                        raise ValueError("Cannot divide by zero")
                    result = num1 / num2
                
                # Replace the matched expression with the result
                expr = expr[:match.start()] + str(result) + expr[match.end():]
        
        # Then handle addition and subtraction
        while '+' in expr or '-' in expr[1:]:  # Skip first character for negative numbers
            # Find addition or subtraction
            match = re.search(r'(-?\d+\.?\d*)\s*([\+\-])\s*(-?\d+\.?\d*)', expr)
            if match:
                num1 = float(match.group(1))
                operator = match.group(2)
                num2 = float(match.group(3))
                
                if operator == '+':
                    result = num1 + num2
                else:  # operator == '-'
                    result = num1 - num2
                
                # Replace the matched expression with the result
                expr = expr[:match.start()] + str(result) + expr[match.end():]
        
        return float(expr)
    
    # Handle parentheses
    while '(' in expression:
        # Find innermost parentheses
        match = re.search(r'\(([^()]+)\)', expression)
        if match:
            inner_expr = match.group(1)
            result = evaluate_simple(inner_expr)
            # Replace the parentheses expression with the result
            expression = expression[:match.start()] + str(result) + expression[match.end():]
        else:
            raise ValueError("Mismatched parentheses")
    
    # Check for unmatched closing parenthesis
    if ')' in expression:
        raise ValueError("Mismatched parentheses")
    
    # Evaluate the final expression without parentheses
    return evaluate_simple(expression)
