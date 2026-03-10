import re

def evaluate_user_expression(expression):
    # Validate the expression contains only allowed characters
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        raise ValueError("Invalid characters in expression")
    
    # Remove whitespace
    expression = expression.replace(' ', '')
    
    # Check for empty expression
    if not expression:
        raise ValueError("Empty expression")
    
    try:
        # Evaluate the expression
        result = eval(expression)
        return float(result)
    except (SyntaxError, ZeroDivisionError, NameError) as e:
        raise ValueError(f"Invalid expression: {str(e)}")

def basic_calculator(num1, num2=None, operator=None):
    # If only one argument is provided, assume it's a string expression
    if num2 is None and operator is None:
        expression = num1
        # Split the expression into parts
        parts = expression.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid expression: {expression}")
        
        num1 = float(parts[0])
        operator = parts[1]
        num2 = float(parts[2])
    else:
        num1 = float(num1)
        num2 = float(num2)
    
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}")
