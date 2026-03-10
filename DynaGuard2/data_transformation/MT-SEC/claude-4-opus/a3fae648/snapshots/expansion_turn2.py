import re
import operator

def substitute_variables(expression, variables):
    """
    Substitutes variables in a mathematical expression with their values.
    
    Args:
        expression: A string representing a mathematical expression
        variables: A dictionary mapping variable names to their values
    
    Returns:
        A string with variables replaced by their values
    """
    # Sort variables by length in descending order to handle cases like 'x' and 'x1'
    sorted_vars = sorted(variables.keys(), key=len, reverse=True)
    
    result = expression
    for var in sorted_vars:
        # Use word boundaries to match whole variable names only
        pattern = r'\b' + re.escape(var) + r'\b'
        result = re.sub(pattern, str(variables[var]), result)
    
    return result

def evaluate_expression(expression):
    """
    Evaluates a mathematical expression string and returns the result.
    
    Args:
        expression: A string representing a mathematical expression with no variables
    
    Returns:
        The computed result as a float
    """
    # Remove whitespace
    expression = expression.replace(' ', '')
    
    # Define operator precedence and functions
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '^': operator.pow,
        '**': operator.pow
    }
    
    # Use eval for basic arithmetic (with safety check)
    # Only allow digits, operators, parentheses, and decimal points
    allowed_chars = set('0123456789+-*/().**^ ')
    if all(c in allowed_chars for c in expression):
        # Replace ^ with ** for Python's power operator
        expression = expression.replace('^', '**')
        try:
            result = eval(expression)
            return float(result)
        except:
            raise ValueError(f"Invalid expression: {expression}")
    else:
        raise ValueError(f"Expression contains invalid characters: {expression}")
