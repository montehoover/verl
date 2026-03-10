import re

def substitute_variables(expression, variables):
    """
    Substitutes variables in a mathematical expression with their values.
    
    Args:
        expression: A string containing the mathematical expression
        variables: A dictionary mapping variable names to their values
        
    Returns:
        A string with variables substituted by their values
    """
    def replace_var(match):
        var_name = match.group(0)
        if var_name in variables:
            return str(variables[var_name])
        return var_name
    
    # Pattern to match variable names (letters, numbers, underscores)
    # but must start with a letter or underscore
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    
    return re.sub(pattern, replace_var, expression)
