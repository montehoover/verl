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


def math_expression_safety_check(math_input):
    """
    Securely evaluate user-supplied mathematical expressions.
    
    Args:
        math_input (str): The mathematical expression to evaluate
        
    Returns:
        float or str: The result of the calculation or 'Unsafe Code Detected!' if unsafe
    """
    # Check for forbidden keywords and patterns that could indicate code injection
    forbidden_patterns = [
        '__', 'import', 'eval', 'exec', 'open', 'file', 'input', 'raw_input',
        'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'hasattr', 'chr', 'ord', 'hex', 'oct', 'bin', 'format',
        'bytes', 'bytearray', 'memoryview', 'type', 'help', 'print', 'lambda',
        'def', 'class', 'for', 'while', 'if', 'else', 'elif', 'try', 'except',
        'finally', 'with', 'as', 'assert', 'break', 'continue', 'pass', 'raise',
        'return', 'yield', 'del', 'from', 'global', 'nonlocal', 'is', 'not',
        'and', 'or', 'in', '=', ';', ':', '[', ']', '{', '}', '\\', '"', "'",
        '`', '@', '$', '%', '&', '|', '~', '!', '?', ',', '<', '>', '\n', '\r',
        '\t', '#'
    ]
    
    # Convert to lowercase for case-insensitive checking
    lower_input = math_input.lower()
    
    # Check for forbidden patterns
    for pattern in forbidden_patterns:
        if pattern in lower_input:
            return 'Unsafe Code Detected!'
    
    # Parse the expression
    parsed_elements = parse_math_expression(math_input)
    
    # If no elements were parsed, it might be an invalid expression
    if not parsed_elements:
        return 'Unsafe Code Detected!'
    
    # Check if the parsed expression when joined matches the original (minus spaces)
    # This helps detect if there were any unparsed characters
    original_no_spaces = math_input.replace(" ", "")
    parsed_joined = ''.join(parsed_elements)
    
    if original_no_spaces != parsed_joined:
        return 'Unsafe Code Detected!'
    
    # Evaluate the expression safely
    result = evaluate_safe_math(parsed_elements)
    
    # If evaluation returned an error, treat it as unsafe
    if isinstance(result, str) and result.startswith("Error:"):
        return 'Unsafe Code Detected!'
    
    return result


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
        "2 + + 3",
        "__import__('os').system('ls')",
        "eval('2+2')",
        "2 + 3; print('hacked')",
        "lambda x: x+1",
        "2 + 3 # comment"
    ]
    
    for expr in test_expressions:
        result = math_expression_safety_check(expr)
        print(f"Expression: {expr}")
        print(f"Result: {result}")
        print()
