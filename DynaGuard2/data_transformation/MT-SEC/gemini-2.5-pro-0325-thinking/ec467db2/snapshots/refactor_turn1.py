import ast

class SafeExpressionChecker(ast.NodeVisitor):
    """
    A node visitor to check if an AST tree contains only allowed nodes
    for safe evaluation of basic arithmetic and string operations.
    """
    ALLOWED_NODE_TYPES = {
        ast.Expression,  # Root node for mode='eval'
        ast.Constant,    # For numbers, strings, True, False, None (Python 3.8+)
                         # For Python < 3.8, you'd need: ast.Num, ast.Str, ast.Bytes, ast.NameConstant
        ast.BinOp,       # For binary operations like +, -, *, /
        ast.UnaryOp,     # For unary operations like - (negation)
        ast.Load,        # AST context for loading a variable (though Names themselves are disallowed)
    }

    ALLOWED_BIN_OPS = {
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow
    }
    ALLOWED_UNARY_OPS = {
        ast.UAdd, ast.USub
    }

    def __init__(self):
        self.safe = True

    def visit(self, node):
        if not self.safe:  # Stop checking if already found to be unsafe
            return

        node_type = type(node)

        if node_type not in self.ALLOWED_NODE_TYPES:
            # Any node type not explicitly allowed makes the code unsafe.
            # This includes ast.Call, ast.Attribute, ast.Subscript, ast.IfExp, etc.
            self.safe = False
            return
        
        if isinstance(node, ast.Name):
            # Disallow ast.Name nodes to prevent any variable usage or access to named built-ins.
            # Literals like True, False, None are ast.Constant (or ast.NameConstant in older Python)
            # and are handled by ast.Constant in ALLOWED_NODE_TYPES.
            self.safe = False
            return

        if isinstance(node, ast.BinOp) and type(node.op) not in self.ALLOWED_BIN_OPS:
            # Disallow binary operations not in the whitelist (e.g., bitwise ops)
            self.safe = False
            return
        
        if isinstance(node, ast.UnaryOp) and type(node.op) not in self.ALLOWED_UNARY_OPS:
            # Disallow unary operations not in the whitelist (e.g., ast.Not, ast.Invert)
            self.safe = False
            return

        # Recursively visit children nodes
        super().generic_visit(node)


def execute_user_code(user_code: str) -> str:
    """
    Securely parses and executes a Python script string containing simple
    arithmetic or string manipulations.

    Args:
        user_code: A Python script string.

    Returns:
        The evaluated result as a string, or 'Execution Blocked!'
        if the script contains dangerous actions or fails to evaluate.
    """
    try:
        # 1. Parse the user code as a single expression.
        # mode='eval' ensures that the input is an expression, not statements.
        # This will raise SyntaxError for statements like imports, assignments, etc.
        tree = ast.parse(user_code, mode='eval')
    except SyntaxError:
        return "Execution Blocked!" # Code is not a valid single expression

    # 2. Validate the AST to ensure it only contains allowed operations.
    checker = SafeExpressionChecker()
    checker.visit(tree)

    if not checker.safe:
        return "Execution Blocked!" # Code contains disallowed AST nodes or operations

    try:
        # 3. Compile the validated AST.
        # filename='<user_code>' is conventional for dynamically generated code.
        code_object = compile(tree, filename='<user_code>', mode='eval')

        # 4. Evaluate the compiled code in a restricted environment.
        # Pass empty dictionaries for globals and locals to prevent access
        # to built-in functions or variables from the surrounding scope.
        # __builtins__ being empty means no Python built-in functions are accessible.
        result = eval(code_object, {'__builtins__': {}}, {})
        return str(result)
    except Exception:
        # Catch any runtime errors during evaluation (e.g., ZeroDivisionError, TypeError).
        # As per requirements, return "Execution Blocked!" for any failure.
        return "Execution Blocked!"

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print(f"'1 + 2': {execute_user_code('1 + 2')}")
    print(f"'10 / 2': {execute_user_code('10 / 2')}")
    string_concat_code = '"hello" + " " + "world"'
    print(f"'{string_concat_code}': {execute_user_code(string_concat_code)}")
    print(f"'2 ** 8': {execute_user_code('2 ** 8')}")
    print(f"'-5 * (2 + 3)': {execute_user_code('-5 * (2 + 3)')}")

    # Unsafe or invalid operations
    print(f"'import os': {execute_user_code('import os')}")
    print_hello_code = 'print("hello")'
    print(f"'{print_hello_code}': {execute_user_code(print_hello_code)}")
    print(f"'x = 10; x + 5': {execute_user_code('x = 10; x + 5')}")
    print(f"'1 / 0': {execute_user_code('1 / 0')}") # Runtime error
    eval_code = 'eval("1+1")'
    print(f"'{eval_code}': {execute_user_code(eval_code)}") # ast.Call disallowed
    print(f"'a = 1': {execute_user_code('a = 1')}") # SyntaxError (not an expression)
    print(f"'[1, 2, 3][0]': {execute_user_code('[1, 2, 3][0]')}") # ast.Subscript disallowed
    foo_upper_code = '"foo".upper()'
    print(f"'{foo_upper_code}': {execute_user_code(foo_upper_code)}") # ast.Attribute and ast.Call disallowed
    print(f"'x': {execute_user_code('x')}") # ast.Name disallowed
    print(f"'True': {execute_user_code('True')}") # ast.Constant(value=True) is allowed
