import re

def parse_math_expression(expression):
    """
    Parse a mathematical expression string and return a list of recognized mathematical elements.
    
    Args:
        expression (str): The mathematical expression to parse
        
    Returns:
        list: A list of recognized mathematical elements (numbers and operators)
    """
    # Remove whitespace
    expression = expression.replace(" ", "")
    
    # Regular expression pattern to match numbers (including decimals) and operators
    # Matches: integers, decimals, and common mathematical operators
    pattern = r'(\d+\.?\d*|[\+\-\*/\(\)\^])'
    
    # Find all matches
    elements = re.findall(pattern, expression)
    
    # Filter out empty strings
    elements = [elem for elem in elements if elem]
    
    return elements


def evaluate_safe_math(elements):
    """
    Evaluate a list of mathematical elements safely.
    
    Args:
        elements (list): List of mathematical elements (numbers and operators)
        
    Returns:
        float or str: The result of the calculation or an error message
    """
    # Define allowed operators
    allowed_operators = {'+', '-', '*', '/', '(', ')', '^'}
    
    # Validate all elements
    for element in elements:
        # Check if it's a number
        try:
            float(element)
        except ValueError:
            # If not a number, check if it's an allowed operator
            if element not in allowed_operators:
                return f"Error: Invalid operation '{element}' detected"
    
    # Convert list to expression string
    expression = ''.join(elements)
    
    # Replace ^ with ** for Python exponentiation
    expression = expression.replace('^', '**')
    
    try:
        # Evaluate the expression safely
        # Create a restricted namespace with only math operations
        safe_dict = {
            '__builtins__': None,
            'abs': abs,
            'min': min,
            'max': max,
        }
        
        # Evaluate the expression
        result = eval(expression, safe_dict)
        
        return float(result)
    
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: Invalid expression - {str(e)}"


# Example usage
if __name__ == "__main__":
    # Test cases
    test_expressions = [
        "2 + 3 * 4",
        "10.5 - 3.2",
        "(5 + 3) * 2",
        "2^3 + 1",
        "100/5-10",
        "10/0",
        "2 + + 3"
    ]
    
    for expr in test_expressions:
        parsed = parse_math_expression(expr)
        result = evaluate_safe_math(parsed)
        print(f"Expression: {expr}")
        print(f"Parsed elements: {parsed}")
        print(f"Result: {result}")
        print()
