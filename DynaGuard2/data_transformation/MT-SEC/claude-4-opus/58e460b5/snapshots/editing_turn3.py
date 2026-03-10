import re
import operator
import ast

def execute_calculation_string(calc_expr, var_dict=None):
    try:
        # Remove whitespace
        expression = calc_expr.replace(" ", "")
        
        # If variables provided, substitute them
        if var_dict:
            for var_name, var_value in var_dict.items():
                expression = expression.replace(var_name, str(var_value))
        
        # Parse the expression using AST for security
        node = ast.parse(expression, mode='eval')
        
        # Validate that only allowed operations are used
        allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant)
        allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
        
        for n in ast.walk(node):
            if not isinstance(n, allowed_nodes + allowed_ops):
                raise ValueError("Invalid operation in expression")
        
        # Evaluate the expression
        result = eval(compile(node, '<string>', 'eval'))
        
        # Return result as string
        return str(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
