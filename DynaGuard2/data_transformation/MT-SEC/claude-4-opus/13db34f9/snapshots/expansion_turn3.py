import operator
import re
import ast

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


def evaluate_math_expression(formula, variables):
    """
    Evaluates a mathematical expression with variables and returns the result as a string.
    
    Args:
        formula: A string representing a mathematical expression with variables
        variables: A dictionary mapping variable names to their values
        
    Returns:
        str: The evaluated result as a string
        
    Raises:
        ValueError: If the expression is invalid or contains undefined variables
    """
    try:
        # Substitute variables in the formula
        substituted_formula = substitute_variables(formula, variables)
        
        # Parse the expression safely using ast
        node = ast.parse(substituted_formula, mode='eval')
        
        # Validate that only safe operations are used
        for n in ast.walk(node):
            if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                ast.Num, ast.Constant, ast.Name, ast.Load,
                                ast.Add, ast.Sub, ast.Mult, ast.Div, 
                                ast.USub, ast.UAdd)):
                raise ValueError("Invalid operation in expression")
        
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        
        # Return result as string
        return str(result)
        
    except (SyntaxError, TypeError, NameError) as e:
        raise ValueError(f"Invalid expression: {formula}")
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {formula}")
