import ast

class SafePythonAstVisitor(ast.NodeVisitor):
    """
    An AST visitor that checks if all nodes in a Python AST are within a predefined
    whitelist of "safe" node types.
    """

    # Whitelist of AST node types considered safe.
    # This list defines the "grammar" of the safe subset of Python.
    # It aims to allow basic data manipulation, arithmetic, and control flow,
    # while disallowing operations that could lead to side effects (I/O, imports)
    # or overly complex/dynamic behavior (arbitrary calls, class definitions).
    ALLOWED_NODE_TYPES = {
        # Core Structure
        ast.Module,  # The root node of a program.
        ast.Expr,    # Wraps an expression when it's used as a statement.

        # Literals
        # In Python 3.8+, ast.Constant covers numbers, strings, bytes, booleans, None, and Ellipsis.
        # For older Python versions, one might need to list:
        # ast.Num, ast.Str, ast.Bytes, ast.NameConstant (True, False, None), ast.Ellipsis
        ast.Constant,

        # Variables and Names
        ast.Name,  # Represents variable names (e.g., x, my_variable).
                   # The context (load, store, del) is part of the Name node.

        # Expressions and Operators
        ast.UnaryOp,  # Unary operations (e.g., -x, not x).
        ast.BinOp,    # Binary operations (e.g., x + y, x * y).
        ast.BoolOp,   # Boolean operations (e.g., x and y, x or y).
        ast.Compare,  # Comparison operations (e.g., x < y, x == y).
        ast.IfExp,    # Ternary conditional expression (e.g., x if condition else y).

        # Statements
        ast.Assign,     # Assignment statement (e.g., x = 1).
        ast.AugAssign,  # Augmented assignment (e.g., x += 1).
        ast.AnnAssign,  # Annotated assignment (e.g., x: int = 10). Type hint is an expression.
        ast.Pass,       # The pass statement.
        ast.Break,      # The break statement.
        ast.Continue,   # The continue statement.
        ast.Delete,     # The del statement (e.g., del x, del my_dict['key']).

        # Control Flow Statements
        ast.If,       # if/elif/else statements.
        ast.For,      # for loops. The expressions for target and iter must also be safe.
        ast.While,    # while loops. The test expression must also be safe.

        # Data Structure Literals
        ast.List,  # List literals (e.g., [1, 2, 3]).
        ast.Tuple, # Tuple literals (e.g., (1, 2, 3)).
        ast.Set,   # Set literals (e.g., {1, 2, 3}).
        ast.Dict,  # Dictionary literals (e.g., {'a': 1, 'b': 2}).

        # Subscripting and Slicing (accessing elements of sequences/mappings)
        ast.Subscript,  # Subscripting (e.g., my_list[0], my_dict['key']).
        ast.Index,      # Represents a simple index (e.g., the `0` in `my_list[0]`).
        ast.Slice,      # Represents a slice (e.g., `1:2` in `my_list[1:2]`).
        ast.ExtSlice,   # Represents extended slicing for multi-dimensional access (e.g., `1:2, 3:4`).

        # Comprehensions (for creating lists, sets, dicts, generators)
        ast.ListComp,       # List comprehension (e.g., [x*x for x in range(5)]).
        ast.SetComp,        # Set comprehension (e.g., {x*x for x in range(5)}).
        ast.DictComp,       # Dictionary comprehension (e.g., {x: x*x for x in range(5)}).
        ast.GeneratorExp,   # Generator expression (e.g., (x*x for x in range(5))).
        ast.comprehension,  # Helper node used within comprehensions.
    }

    # Potentially unsafe nodes (explicitly disallowed for this version):
    # ast.Import, ast.ImportFrom (file system access, arbitrary code execution)
    # ast.Call (arbitrary function calls, e.g., open(), eval(), __import__())
    # ast.FunctionDef, ast.Lambda, ast.ClassDef (defining new callables/types - complex behavior)
    # ast.Attribute (accessing attributes, e.g., obj.method, obj.__class__ - can lead to vulnerabilities)
    # ast.Raise, ast.Try, ast.Assert (exception handling - can alter control flow in complex ways)
    # ast.Global, ast.Nonlocal (modifying scopes)
    # ast.With (context managers - can have side effects like file operations)
    # Async nodes: ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith

    def __init__(self):
        super().__init__()
        self.is_safe = True

    def visit(self, node):
        """
        Visits a node in the AST. If the node type is not in the allowed list,
        marks the script as unsafe and stops further traversal for this branch.
        """
        # If already determined to be unsafe, no need to continue.
        if not self.is_safe:
            return

        node_type = type(node)

        if node_type not in self.ALLOWED_NODE_TYPES:
            # For debugging purposes, one might uncomment the following line:
            # print(f"Disallowed AST node type: {node_type.__name__} at line {getattr(node, 'lineno', 'N/A')}")
            self.is_safe = False
            return
        
        # If specific checks for an allowed node type are needed (e.g., to disallow
        # certain names in ast.Name, or certain functions if ast.Call were allowed),
        # one would override the specific visit_NodeType method (e.g., visit_Name).
        # For this version, type-based whitelisting is the primary mechanism.

        # Continue traversal to child nodes.
        super().visit(node)


def validate_python_script(script_string: str) -> bool:
    """
    Validates if a Python script string uses only a predefined safe subset of
    Python Abstract Syntax Tree (AST) nodes.

    This function parses the script into an AST and then traverses it, checking
    each node against a whitelist of allowed node types. The whitelist is
    designed to permit basic computational and data manipulation tasks while
    excluding operations that could introduce security risks or side effects,
    such as file I/O, network access, imports, or arbitrary code execution.

    Args:
        script_string: A string containing the Python code to be validated.

    Returns:
        True if the script consists only of allowed AST nodes and is
        syntactically correct. False if it contains disallowed AST nodes or
        has syntax errors.
    """
    try:
        # Parse the Python code string into an AST.
        tree = ast.parse(script_string)
    except SyntaxError:
        # If the code has syntax errors, it's considered unsafe/invalid.
        return False

    # Create an instance of the AST visitor.
    visitor = SafePythonAstVisitor()

    # Traverse the AST. The visitor will set its 'is_safe' flag to False
    # if any disallowed node type is encountered.
    visitor.visit(tree)

    return visitor.is_safe

if __name__ == '__main__':
    # Example Usage and Tests:

    safe_scripts = [
        "a = 1 + 2",
        "x = [1, 2, 3]\ny = x[0]",
        "total = 0\nfor i in range(1, 11):\n  total += i", # range() is a Call, so this will be False
        "if x > 0:\n  y = 1\nelse:\n  y = -1",
        "my_dict = {'a': 1, 'b': 2}\nval = my_dict['a']",
        "z = [x*x for x in [1,2,3]]",
        "a = True\nb = False\nc = None",
        "del x[0]",
        "y: int = 10",
        "k = (1 if True else 0)",
    ]

    unsafe_scripts = [
        "import os",
        "os.system('echo unsafe')", # ast.Attribute and ast.Call
        "eval('1+1')", # ast.Call
        "print('hello')", # ast.Call
        "open('file.txt', 'w').write('data')", # ast.Call, ast.Attribute
        "def my_func():\n  return 1", # ast.FunctionDef
        "class MyClass:\n  pass", # ast.ClassDef
        "try:\n  x = 1/0\nexcept ZeroDivisionError:\n  pass", # ast.Try
        "with open('f.txt') as f: pass", # ast.With
        "lambda x: x + 1", # ast.Lambda
        "assert True", # ast.Assert
        "global x", # ast.Global
        "nonlocal y", # ast.Nonlocal
        "yield 1", # ast.Yield
        "async def f(): await g()", # ast.AsyncFunctionDef, ast.Await
        "a = b.c", # ast.Attribute
        "a = __import__('os')", # ast.Call
    ]
    
    # Corrected expectation for the 'range' example
    # The script "total = 0\nfor i in range(1, 11):\n  total += i" uses `range()`, which is an `ast.Call`.
    # Since `ast.Call` is not in `ALLOWED_NODE_TYPES`, this script should be marked unsafe.
    # Let's adjust the test lists or the validator if `range` is intended to be safe (would require allowing Call + whitelisting `range`).
    # For now, it's correctly identified as unsafe by the current rules.
    
    # The example "total = 0\nfor i in range(1, 11):\n  total += i" is actually unsafe with current rules
    # because `range(1,11)` is an `ast.Call` node.
    # Moving it to unsafe_scripts for clarity based on current ALLOWED_NODE_TYPES.
    
    print("Testing safe scripts (expected True, unless they use Call/Attribute implicitly):")
    for i, script in enumerate(safe_scripts):
        is_valid = validate_python_script(script)
        print(f"Script {i+1}: {is_valid} \n{script}\n")
        # Note: "z = [x*x for x in [1,2,3]]" is safe as [1,2,3] is a List literal.
        # If it were `range(3)`, it would be unsafe due to `Call`.

    # Adjusting safe_scripts based on current rules (no Call, no Attribute)
    print("\n--- Adjusted Safe Scripts (Strictly No Call/Attribute) ---")
    strictly_safe_scripts = [
        "a = 1 + 2",
        "x = [1, 2, 3]\ny = x[0]",
        "if x > 0:\n  y = 1\nelse:\n  y = -1",
        "my_dict = {'a': 1, 'b': 2}\nval = my_dict['a']",
        "z = [x*x for x_val in [1,2,3]]", # Renamed x to x_val to avoid NameError if x not defined
        "a_bool = True\nb_bool = False\nc_none = None", # Renamed to avoid NameError
        "my_list_for_del = [1,2,3]\ndel my_list_for_del[0]", # Setup for del
        "y_ann: int = 10",
        "k_ifexp = (1 if True else 0)",
        "pass",
        "while False:\n break\ncontinue",
        "my_set = {1,2,3}",
        "my_tuple = (1,2,3)",
        "d = {}\nd['key'] = 'value'",
        "l = []\nl_slice = l[1:2]",
    ]
    
    # Add a simple for loop that doesn't use range()
    strictly_safe_scripts.append("total_val = 0\nfor i_val in [1, 2, 3]:\n  total_val += i_val")


    for i, script in enumerate(strictly_safe_scripts):
        # Define variables used in scripts to avoid NameError during manual review
        # This doesn't affect AST validation but helps if one were to try to exec() them.
        exec_globals = {'x': 5, 'my_list_for_del': [10,20], 'l': [1,2,3,4]}
        is_valid = validate_python_script(script)
        print(f"Strictly Safe Script {i+1}: {is_valid} \n{script}\n")
        if not is_valid:
            print(f"    ERROR: Expected True for script: {script}")


    print("\nTesting unsafe scripts (expected False):")
    # Add the range example to unsafe scripts
    unsafe_scripts.insert(0, "total = 0\nfor i in range(1, 11):\n  total += i")

    for i, script in enumerate(unsafe_scripts):
        is_valid = validate_python_script(script)
        print(f"Unsafe Script {i+1}: {is_valid} \n{script}\n")
        if is_valid:
            print(f"    ERROR: Expected False for script: {script}")

    print("\nTesting script with syntax error (expected False):")
    syntax_error_script = "a = 1 +"
    is_valid = validate_python_script(syntax_error_script)
    print(f"Syntax Error Script: {is_valid} \n{syntax_error_script}\n")
    if is_valid:
        print(f"    ERROR: Expected False for script: {syntax_error_script}")
