import ast

def evaluate_expression(input_str: str) -> list:
    # Check for alphabets or special characters
    for char in input_str:
        if not (char.isdigit() or char.isspace()):
            raise ValueError("Input contains alphabets or special characters")
    
    # Split by spaces and convert to integers
    numbers = []
    for token in input_str.split():
        if token:  # Skip empty strings from multiple spaces
            numbers.append(int(token))
    
    return numbers

def secure_eval_expression(input_str: str):
    # Define allowed characters: digits, operators, parentheses, and whitespace
    allowed_chars = set('0123456789+-*/() .')
    
    # Check for any unsafe characters
    for char in input_str:
        if char not in allowed_chars:
            raise ValueError(f"Unsafe character detected: '{char}'")
    
    # Parse and validate the expression using AST
    try:
        tree = ast.parse(input_str, mode='eval')
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    
    # Validate that only safe operations are used
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            raise ValueError("Variable names are not allowed")
        elif isinstance(node, ast.Call):
            raise ValueError("Function calls are not allowed")
        elif isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                                  ast.USub, ast.UAdd)):
            raise ValueError(f"Unsafe operation detected: {type(node).__name__}")
    
    # Evaluate the expression safely
    return eval(compile(tree, '<string>', 'eval'))
