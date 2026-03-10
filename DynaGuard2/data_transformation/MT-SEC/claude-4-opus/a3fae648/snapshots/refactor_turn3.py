import re
import operator
import ast
import logging

# Configure logger
logger = logging.getLogger(__name__)


def _substitute_variables(formula_str, vars_mapping):
    """
    Substitutes variables in the formula string with their numeric values.
    
    Args:
        formula_str (str): The mathematical formula containing variables
        vars_mapping (dict): Mapping of variable names to their values
        
    Returns:
        str: The formula with variables replaced by their values
    """
    processed_formula = formula_str
    
    # Sort variables by length (longest first) to avoid partial replacements
    sorted_vars = sorted(vars_mapping.keys(), key=len, reverse=True)
    
    for var in sorted_vars:
        # Use word boundaries to match whole variable names only
        pattern = r'\b' + re.escape(var) + r'\b'
        value_str = str(vars_mapping[var])
        
        # Add parentheses around negative values to preserve operation order
        if vars_mapping[var] < 0:
            value_str = f"({value_str})"
            
        processed_formula = re.sub(pattern, value_str, processed_formula)
        logger.debug(f"Substituted variable '{var}' with value {value_str}")
    
    logger.debug(f"Formula after substitution: {processed_formula}")
    return processed_formula


def _validate_ast_nodes(tree):
    """
    Validates that the AST only contains allowed operations.
    
    Args:
        tree: The parsed AST tree
        
    Raises:
        ValueError: If forbidden operations or undefined variables are found
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            raise ValueError("Function calls are not allowed")
        elif isinstance(node, ast.Name):
            raise ValueError(f"Undefined variable: {node.id}")


def _evaluate_expression(expression_str):
    """
    Safely evaluates a mathematical expression string.
    
    Args:
        expression_str (str): The mathematical expression to evaluate
        
    Returns:
        The evaluated result
        
    Raises:
        ValueError: If the expression is invalid
    """
    # Parse the expression into an AST
    tree = ast.parse(expression_str, mode='eval')
    
    # Validate the AST
    _validate_ast_nodes(tree)
    
    # Compile and evaluate the expression
    code = compile(tree, '<string>', 'eval')
    result = eval(code)
    
    logger.debug(f"Evaluated expression '{expression_str}' to: {result}")
    return result


def simplify_math_expression(formula_str, vars_mapping):
    """
    Processes a mathematical expression string with variables and returns the calculated result.
    
    Args:
        formula_str (str): A string representing a mathematical formula potentially containing variables
        vars_mapping (dict): A mapping of variable names to their numeric values for evaluation
        
    Returns:
        str: The result after computing the expression, returned in string format
        
    Raises:
        ValueError: If an error occurs due to an invalid expression or unsuccessful processing
    """
    logger.info(f"Processing expression: {formula_str}")
    logger.info(f"Variable mappings: {vars_mapping}")
    
    try:
        # Replace variables in the formula with their values
        processed_formula = _substitute_variables(formula_str, vars_mapping)
        
        # Evaluate the expression
        result = _evaluate_expression(processed_formula)
        
        # Return the result as a string
        result_str = str(result)
        logger.info(f"Final result: {result_str}")
        return result_str
        
    except Exception as e:
        logger.error(f"Error processing expression: {str(e)}")
        raise ValueError(f"Invalid expression or unsuccessful processing: {str(e)}")
