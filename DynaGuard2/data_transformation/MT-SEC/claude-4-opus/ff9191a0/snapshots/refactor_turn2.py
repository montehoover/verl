import re
import operator
import ast

def substitute_variables(expression, variable_mapping):
    """
    Substitute variables in the expression with their values.
    
    Args:
        expression (str): Mathematical expression with variables
        variable_mapping (dict): Variable names to values mapping
    
    Returns:
        str: Expression with variables replaced by values
    """
    result = expression
    for var, value in variable_mapping.items():
        # Use word boundaries to avoid partial matches
        result = re.sub(r'\b' + re.escape(var) + r'\b', str(value), result)
    return result

def parse_expression(expression):
    """
    Parse the expression into an AST and validate it.
    
    Args:
        expression (str): Mathematical expression to parse
    
    Returns:
        ast.AST: Parsed and validated AST
    
    Raises:
        ValueError: If expression is invalid or contains unsafe operations
    """
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    
    # Check if the AST only contains safe operations
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            # Check if there are any remaining variables
            raise ValueError(f"Undefined variable: {node.id}")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, 
                                 ast.Add, ast.Sub, ast.Mult, ast.Div, 
                                 ast.Pow, ast.Mod, ast.FloorDiv,
                                 ast.USub, ast.UAdd, ast.Constant, ast.Num)):
            raise ValueError("Invalid operation in expression")
    
    return tree

def evaluate_ast(tree):
    """
    Evaluate the AST and return the result.
    
    Args:
        tree (ast.AST): Parsed AST to evaluate
    
    Returns:
        Union[int, float]: Evaluation result
    
    Raises:
        ValueError: If evaluation fails
    """
    try:
        compiled = compile(tree, '<string>', 'eval')
        result = eval(compiled)
        return result
    except (TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Computation failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

def evaluate_expression_safely(math_expression, variable_mapping):
    """
    Evaluate a mathematical expression with variable substitution.
    
    Args:
        math_expression (str): A string representing a mathematical expression
        variable_mapping (dict): A dictionary where keys are variable names and values are their corresponding numbers
    
    Returns:
        str: The result of the simplified expression as a string
    
    Raises:
        ValueError: If the input expression is invalid or the computation fails
    """
    try:
        # Pipeline: substitute -> parse -> evaluate
        substituted = substitute_variables(math_expression, variable_mapping)
        parsed_tree = parse_expression(substituted)
        result = evaluate_ast(parsed_tree)
        
        # Return the result as a string
        return str(result)
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")
