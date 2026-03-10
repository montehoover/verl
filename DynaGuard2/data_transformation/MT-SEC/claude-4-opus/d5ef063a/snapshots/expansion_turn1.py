import ast

def parse_python_code(code_string):
    """
    Parse a string of Python code and return its AST if valid.
    
    Args:
        code_string (str): The Python code to parse
        
    Returns:
        ast.AST: The abstract syntax tree of the code
        
    Raises:
        ValueError: If the code has syntax errors
    """
    try:
        tree = ast.parse(code_string)
        return tree
    except SyntaxError as e:
        raise ValueError(f"Syntax error in code: {e}")
