import re
import operator
import ast
import logging

# Set up logger
logger = logging.getLogger(__name__)


def substitute_variables(expression, variables):
    """
    Replace variables in the expression with their corresponding values.
    
    Args:
        expression (str): Mathematical expression containing variables
        variables (dict): Dictionary mapping variable names to their values
        
    Returns:
        str: Expression with variables replaced by their values
    """
    result = expression
    for var_name, var_value in variables.items():
        # Use word boundaries to avoid partial replacements
        result = re.sub(r'\b' + re.escape(var_name) + r'\b', str(var_value), result)
        logger.debug(f"Substituted '{var_name}' with '{var_value}'")
    return result


def validate_expression(parsed_node):
    """
    Validate that the parsed AST node contains only allowed operations.
    
    Args:
        parsed_node: AST node to validate
        
    Raises:
        ValueError: If the expression contains disallowed operations
    """
    for element in ast.walk(parsed_node):
        if isinstance(element, ast.Call):
            raise ValueError("Function calls are not allowed")
        elif isinstance(element, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed")
        elif isinstance(element, ast.Name):
            raise ValueError(f"Undefined variable: {element.id}")


def evaluate_expression(expression):
    """
    Parse and evaluate a mathematical expression safely.
    
    Args:
        expression (str): Mathematical expression to evaluate
        
    Returns:
        The result of the evaluated expression
        
    Raises:
        ValueError: If the expression is invalid or evaluation fails
    """
    try:
        # Parse the expression
        node = ast.parse(expression, mode='eval')
        
        # Validate the expression
        validate_expression(node)
        
        # Compile and evaluate the expression
        code = compile(node, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, {})
        
        return result
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Computation failed: {e}")


def execute_calculation_string(calc_expr, var_dict):
    """
    Takes a mathematical expression as a string and computes the simplified result.
    
    Args:
        calc_expr (str): A string representing a mathematical expression
        var_dict (dict): A dictionary where keys are variable names and values are their corresponding numbers
        
    Returns:
        str: The result of the simplified expression as a string
        
    Raises:
        ValueError: If the input expression is invalid or the computation fails
    """
    # Log the initial expression and variables
    logger.info(f"Evaluating expression: '{calc_expr}'")
    logger.info(f"Variables: {var_dict}")
    
    # Replace variables with their values
    expr_with_values = substitute_variables(calc_expr, var_dict)
    logger.info(f"Expression after substitution: '{expr_with_values}'")
    
    # Evaluate the expression
    result = evaluate_expression(expr_with_values)
    logger.info(f"Result: {result}")
    
    # Return the result as a string
    return str(result)
