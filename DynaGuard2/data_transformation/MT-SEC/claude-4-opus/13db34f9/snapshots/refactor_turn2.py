import re
import operator
import ast

def substitute_variables(formula, vars):
    """
    Replace variables in the formula with their corresponding values.
    
    Args:
        formula (str): Mathematical formula containing variables
        vars (dict): Variable name to value mapping
        
    Returns:
        str: Formula with variables replaced by their values
    """
    processed_formula = formula
    for var_name, var_value in vars.items():
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(var_name) + r'\b'
        processed_formula = re.sub(pattern, str(var_value), processed_formula)
    return processed_formula


def parse_expression(expression):
    """
    Parse a mathematical expression into an AST node.
    
    Args:
        expression (str): Mathematical expression to parse
        
    Returns:
        ast.Expression: Parsed AST node
        
    Raises:
        ValueError: If expression cannot be parsed
    """
    try:
        return ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")


def validate_ast_node(node):
    """
    Validate that the AST node contains only safe operations.
    
    Args:
        node (ast.Node): AST node to validate
        
    Raises:
        ValueError: If node contains invalid operations or undefined variables
    """
    allowed_node_types = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, 
        ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, 
        ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv
    )
    
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            # Check for any remaining unsubstituted variables
            raise ValueError(f"Undefined variable: {n.id}")
        elif not isinstance(n, allowed_node_types):
            raise ValueError("Invalid operation in expression")


def evaluate_ast_node(node):
    """
    Evaluate a validated AST node.
    
    Args:
        node (ast.Expression): AST node to evaluate
        
    Returns:
        float or int: Result of the evaluation
        
    Raises:
        ValueError: If evaluation fails
    """
    try:
        return eval(compile(node, '<string>', 'eval'))
    except (TypeError, ZeroDivisionError, NameError) as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")


def evaluate_math_expression(formula, vars):
    """
    Process a mathematical expression string with variables and return the calculated result.
    
    Args:
        formula (str): A string representing a mathematical formula potentially containing variables
        vars (dict): A mapping of variable names to their numeric values for evaluation
        
    Returns:
        str: The result after computing the expression, returned in string format
        
    Raises:
        ValueError: If an error occurs due to an invalid expression or unsuccessful processing
    """
    try:
        # Pipeline: substitute -> parse -> validate -> evaluate -> convert to string
        substituted = substitute_variables(formula, vars)
        parsed_node = parse_expression(substituted)
        validate_ast_node(parsed_node)
        result = evaluate_ast_node(parsed_node)
        return str(result)
        
    except ValueError:
        # Re-raise ValueError as is
        raise
    except Exception as e:
        # Wrap any other unexpected errors
        raise ValueError(f"Unexpected error: {str(e)}")
