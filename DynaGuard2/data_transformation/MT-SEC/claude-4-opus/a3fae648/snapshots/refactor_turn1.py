import re
import operator
import ast

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
    try:
        # Replace variables in the formula with their values
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
        
        # Parse and evaluate the expression safely using ast
        # First, parse the expression into an AST
        tree = ast.parse(processed_formula, mode='eval')
        
        # Check that the expression only contains allowed operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(node, ast.Name):
                raise ValueError(f"Undefined variable: {node.id}")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Return the result as a string
        return str(result)
        
    except Exception as e:
        raise ValueError(f"Invalid expression or unsuccessful processing: {str(e)}")
