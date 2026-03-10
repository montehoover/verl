import re

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
