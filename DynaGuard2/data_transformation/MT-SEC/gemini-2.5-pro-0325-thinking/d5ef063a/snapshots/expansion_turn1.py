import ast

def parse_python_code(code_string: str):
    """
    Parses a string of Python code and returns its AST.

    Args:
        code_string: The Python code to parse.

    Returns:
        An AST object if the code is syntactically correct.

    Raises:
        ValueError: If the code has syntax errors.
    """
    try:
        tree = ast.parse(code_string)
        return tree
    except SyntaxError as e:
        raise ValueError(f"Syntax error in Python code: {e}")

if __name__ == '__main__':
    # Example usage:
    valid_code = "print('Hello, world!')"
    invalid_code = "print 'Hello, world!'" # Python 2 syntax, invalid in Python 3

    print("Testing valid code:")
    try:
        ast_tree = parse_python_code(valid_code)
        print("AST generated successfully:")
        print(ast.dump(ast_tree, indent=4))
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting invalid code:")
    try:
        ast_tree = parse_python_code(invalid_code)
        print("AST generated successfully:")
        print(ast.dump(ast_tree, indent=4))
    except ValueError as e:
        print(f"Error: {e}")

    more_invalid_code = "a = 1 +"
    print("\nTesting more invalid code:")
    try:
        ast_tree = parse_python_code(more_invalid_code)
        print("AST generated successfully:")
        print(ast.dump(ast_tree, indent=4))
    except ValueError as e:
        print(f"Error: {e}")
