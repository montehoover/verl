import operator
import re

def evaluate_simple_expression(expression):
    """
    Evaluates a simple arithmetic expression and returns the result as a float.
    
    Args:
        expression: A string representing a mathematical expression
        
    Returns:
        float: The calculated result
        
    Raises:
        ValueError: If the expression is invalid
    """
    # Define allowed operators
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }
    
    try:
        # Remove whitespace
        expression = expression.replace(' ', '')
        
        # Find the operator
        op_found = None
        op_index = -1
        
        # Check for each operator (reverse order for subtraction to handle negative numbers)
        for i in range(len(expression) - 1, 0, -1):
            if expression[i] in ops:
                op_found = expression[i]
                op_index = i
                break
        
        if op_found is None:
            raise ValueError("No valid operator found")
        
        # Split the expression
        left_part = expression[:op_index]
        right_part = expression[op_index + 1:]
        
        # Convert to numbers
        left_num = float(left_part)
        right_num = float(right_part)
        
        # Perform the operation
        result = ops[op_found](left_num, right_num)
        
        return result
        
    except (ValueError, IndexError, KeyError) as e:
        raise ValueError(f"Invalid expression: {expression}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")


def substitute_variables(expression, variables):
    """
    Substitutes variables in an expression with their corresponding values.
    
    Args:
        expression: A string representing a mathematical expression with variables
        variables: A dictionary mapping variable names to their values
        
    Returns:
        str: The expression with variables substituted by their values
        
    Raises:
        ValueError: If a variable is not found in the dictionary
    """
    # Find all variable names in the expression (alphanumeric and underscore)
    var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    
    def replace_var(match):
        var_name = match.group(0)
        if var_name not in variables:
            raise ValueError(f"Variable '{var_name}' not found in dictionary")
        return str(variables[var_name])
    
    # Replace all variables with their values
    result = re.sub(var_pattern, replace_var, expression)
    
    return result
