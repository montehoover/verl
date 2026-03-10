import ast

class SyntaxChecker(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.is_safe = True
        
        # Whitelist of AST node types that are considered safe.
        # This set defines what constitutes "simple arithmetic or string operations."
        self.allowed_nodes = {
            ast.Module,     # The root node of a script.
            ast.Expr,       # For statements that are expressions (e.g., "1 + 1" on a line).
            ast.Assign,     # For assignment statements (e.g., "x = 1").
            ast.Name,       # For variable names (in load or store contexts).
            ast.Constant,   # For literals: numbers, strings, True, False, None.
                            # Note: ast.Constant is for Python 3.8+. For older versions,
                            # you might need ast.Num, ast.Str, ast.NameConstant.
            ast.BinOp,      # For binary operations (e.g., "a + b", "s1 + s2").
            ast.UnaryOp,    # For unary operations (e.g., "-a", "not b").
            ast.Compare,    # For comparison operations (e.g., "a == b", "x > 0").
            ast.BoolOp,     # For boolean operations ("and", "or").
        }
        
        # Whitelist of binary operators (op attribute of ast.BinOp)
        self.allowed_bin_ops = {
            ast.Add,        # Addition (numbers), Concatenation (strings)
            ast.Sub,        # Subtraction
            ast.Mult,       # Multiplication
            ast.Div,        # Division
            ast.FloorDiv,   # Integer Division
            ast.Mod,        # Modulo
            ast.Pow,        # Power
        }
        
        # Whitelist of unary operators (op attribute of ast.UnaryOp)
        self.allowed_unary_ops = {
            ast.UAdd,       # Unary Plus (e.g., "+5")
            ast.USub,       # Unary Minus (e.g., "-5")
            ast.Not,        # Logical Not (e.g., "not True")
        }
        
        # Whitelist of comparison operators (items in ops list of ast.Compare)
        self.allowed_compare_ops = {
            ast.Eq,         # Equal (==)
            ast.NotEq,      # Not Equal (!=)
            ast.Lt,         # Less Than (<)
            ast.LtE,        # Less Than or Equal (<=)
            ast.Gt,         # Greater Than (>)
            ast.GtE,        # Greater Than or Equal (>=)
        }
        
        # Whitelist of boolean operators (op attribute of ast.BoolOp)
        self.allowed_bool_ops = {
            ast.And,        # Logical And
            ast.Or,         # Logical Or
        }

    def visit(self, node):
        # If a disallowed construct has already been found, stop further traversal.
        if not self.is_safe:
            return

        # Check if the type of the current node is in our whitelist.
        if type(node) not in self.allowed_nodes:
            self.is_safe = False
            # For debugging:
            # print(f"Disallowed node type: {type(node).__name__} at line {getattr(node, 'lineno', 'N/A')}")
            return

        # Perform more specific checks for node types that have operator subtypes.
        if isinstance(node, ast.BinOp) and type(node.op) not in self.allowed_bin_ops:
            self.is_safe = False
            # For debugging:
            # print(f"Disallowed binary operator: {type(node.op).__name__} at line {getattr(node, 'lineno', 'N/A')}")
            return
        
        if isinstance(node, ast.UnaryOp) and type(node.op) not in self.allowed_unary_ops:
            self.is_safe = False
            # For debugging:
            # print(f"Disallowed unary operator: {type(node.op).__name__} at line {getattr(node, 'lineno', 'N/A')}")
            return

        if isinstance(node, ast.Compare):
            # node.ops is a list of comparison operators (e.g., in "a < b <= c")
            for op_type in node.ops:
                if type(op_type) not in self.allowed_compare_ops:
                    self.is_safe = False
                    # For debugging:
                    # print(f"Disallowed comparison operator: {type(op_type).__name__} at line {getattr(node, 'lineno', 'N/A')}")
                    return
        
        if isinstance(node, ast.BoolOp) and type(node.op) not in self.allowed_bool_ops:
            self.is_safe = False
            # For debugging:
            # print(f"Disallowed boolean operator: {type(node.op).__name__} at line {getattr(node, 'lineno', 'N/A')}")
            return
        
        # If ast.Name node has a 'Del' context (ast.Del()), it's part of an ast.Delete statement.
        # Since ast.Delete is not in self.allowed_nodes, constructs like "del x" will be disallowed
        # when the ast.Delete node itself is visited. So, no specific check for Name.ctx is needed here.

        # Continue traversing the AST.
        super().visit(node)

def check_script_syntax(script_string: str) -> bool:
    """
    Verifies if the given script string contains only simple arithmetic or string operations.

    This function parses the script into an Abstract Syntax Tree (AST) and checks each node
    against a whitelist of allowed language constructs. The goal is to ensure that the script
    does not contain any potentially harmful or overly complex operations.

    "Simple operations" are conservatively defined as:
    - Literals: Numbers (integers, floats), strings, booleans (True, False), None.
    - Variable assignments and usage (e.g., x = 10, y = x + 5).
    - Arithmetic operators: +, -, *, /, // (floor division), % (modulo), ** (power).
    - String concatenation using the '+' operator.
    - Comparison operators: ==, !=, <, <=, >, >=.
    - Boolean operators: and, or, not.

    The function explicitly disallows (among other things):
    - Imports (import os).
    - Function calls (print(), open(), eval(), exec(), etc.).
    - Attribute access (e.g., "mystring".upper(), obj.method()).
    - Control flow statements (if, for, while, try-except).
    - Definitions of functions or classes (def, class).
    - List, dictionary, set, or tuple literals and comprehensions ([1,2], { 'a':1 }, {1,2}).
    - Subscripting (my_list[0]).
    - Lambda expressions.

    Args:
        script_string: A string containing the Python script to check.

    Returns:
        True if the script contains only allowed simple operations, False otherwise.
        Returns False if the script has a syntax error.
    """
    try:
        tree = ast.parse(script_string)
    except SyntaxError:
        # The script is not syntactically valid Python.
        return False

    checker = SyntaxChecker()
    checker.visit(tree)
    return checker.is_safe

if __name__ == '__main__':
    # Example Usage and Tests:
    
    # Safe scripts
    safe_scripts = [
        "x = 1 + 2",
        "y = x * (3 - 1) / 2.0",
        "s = 'hello' + ' ' + 'world'",
        "result = 10 // 3",
        "power = 2 ** 8",
        "remainder = 10 % 3",
        "is_greater = x > y",
        "is_equal = x == 10",
        "logical_and = (x > 0) and (y < 0)",
        "logical_or = (x == 0) or (y == 0)",
        "logical_not = not (x == y)",
        "a = True\nb = False\nc = None",
        "num = -5.5\npositive_num = +num",
        "# This is a comment\nx=1 # inline comment",
        "123\n'abc'", # Expressions as statements
        "var_1 = 100\n_my_var = 200" # Valid variable names
    ]

    print("Testing safe scripts:")
    for i, script in enumerate(safe_scripts):
        is_safe = check_script_syntax(script)
        print(f"Script {i+1}: {is_safe} -> {script.splitlines()[0]}")
        assert is_safe, f"Script expected to be safe but was not: {script}"

    # Unsafe scripts
    unsafe_scripts = [
        "import os",
        "print('hello')",
        "eval('1+1')",
        "__import__('os').system('clear')",
        "x = open('file.txt', 'w')",
        "s = 'hello'.upper()",
        "l = [1, 2, 3]",
        "d = {'a': 1}",
        "t = (1, 2)",
        "my_set = {1, 2, 3}",
        "if x > 0: x = 1",
        "for i in range(5): pass",
        "while True: break",
        "def my_func(): pass",
        "class MyClass: pass",
        "try:\n  x=1/0\nexcept ZeroDivisionError:\n  pass",
        "with open('f.txt') as f: data = f.read()",
        "del x",
        "lambda y: y + 1",
        "x = y[0]",
        "assert x > 0",
        "global z",
        "nonlocal q",
        "yield 1",
        "async def my_async_func(): await some_call()"
    ]

    print("\nTesting unsafe scripts:")
    for i, script in enumerate(unsafe_scripts):
        is_safe = check_script_syntax(script)
        print(f"Script {i+1}: {not is_safe} -> {script.splitlines()[0]}")
        assert not is_safe, f"Script expected to be unsafe but was not: {script}"

    # Script with syntax error
    syntax_error_script = "x = 1 + "
    print("\nTesting script with syntax error:")
    is_safe = check_script_syntax(syntax_error_script)
    print(f"Script: {not is_safe} -> {syntax_error_script}")
    assert not is_safe, "Script with syntax error should be unsafe."
    
    print("\nAll tests passed.")
