import re
import operator
import ast

def evaluate_expression_safely(math_expression, variable_mapping):
    """
    Evaluates a mathematical expression with variable substitution.
    
    Args:
        math_expression (str): A string representing a mathematical expression
        variable_mapping (dict): A dictionary where keys are variable names and values are their corresponding numbers
    
    Returns:
        str: The result of the simplified expression as a string
    
    Raises:
        ValueError: If the input expression is invalid or the computation fails
    """
    try:
        # First, validate that all variables in the expression are in the mapping
        # Find all potential variable names (sequences of letters)
        potential_vars = re.findall(r'[a-zA-Z_]\w*', math_expression)
        
        # Check if all variables are in the mapping
        for var in potential_vars:
            if var not in variable_mapping:
                raise ValueError(f"Variable '{var}' not found in variable mapping")
        
        # Create a safe namespace with only allowed operations
        safe_dict = {
            '__builtins__': None,
            'abs': abs,
            'min': min,
            'max': max,
        }
        
        # Add the variable mappings to the safe namespace
        safe_dict.update(variable_mapping)
        
        # Parse the expression into an AST
        try:
            tree = ast.parse(math_expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        
        # Validate that the AST only contains safe operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Only allow specific function calls
                if not (isinstance(node.func, ast.Name) and node.func.id in ['abs', 'min', 'max']):
                    raise ValueError("Function calls not allowed except abs, min, max")
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, 
                                 ast.ClassDef, ast.Delete, ast.Assign, ast.AugAssign)):
                raise ValueError("Invalid operation in expression")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, safe_dict)
        
        # Convert result to string
        # If it's a float that's actually an integer, show it as an integer
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return str(result)
            
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")
