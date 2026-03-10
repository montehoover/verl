import ast
from typing import Any

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


def execute_safe_code(script_ast: ast.Module) -> Any:
    """
    Executes a pre-validated Python Abstract Syntax Tree (AST) in a controlled
    environment and returns the result of the last expression statement.

    The AST is expected to be an ast.Module object, validated by
    `validate_python_script` to ensure it only contains safe operations.

    Args:
        script_ast: A validated ast.Module object representing the Python script.

    Returns:
        The result of the last expression statement in the script. If the script
        is empty or its last statement is not an expression, None is returned.

    Raises:
        Exception: Propagates any exceptions that occur during the compilation
                   or execution of the script (e.g., ZeroDivisionError, NameError).
    """
    if not isinstance(script_ast, ast.Module):
        raise TypeError("Input must be an ast.Module object.")

    execution_context = {}  # Serves as both globals and locals

    try:
        # Compile the entire module for execution
        # All statements will be executed, modifying execution_context
        code_obj = compile(script_ast, filename='<safe_script_exec>', mode='exec')
        exec(code_obj, execution_context)

        # If the script is not empty and the last statement was an ast.Expr,
        # evaluate that expression and return its result.
        if script_ast.body and isinstance(script_ast.body[-1], ast.Expr):
            # The last statement is an expression. Its value is what we want.
            # We need to compile and evaluate this expression node.
            last_expr_node = script_ast.body[-1].value
            
            # Wrap the expression node in an ast.Expression for 'eval' mode compilation
            eval_ast = ast.Expression(body=last_expr_node)
            
            eval_code_obj = compile(eval_ast, filename='<safe_script_eval>', mode='eval')
            result = eval(eval_code_obj, execution_context)
            return result
        else:
            # Script is empty or last statement is not an expression (e.g., an assignment)
            return None
    except Exception as e:
        # Propagate any runtime errors (NameError, ZeroDivisionError, etc.)
        # print(f"Execution error: {e}") # For debugging
        raise


def run_user_script(user_script: str) -> Any:
    """
    Validates and executes a user-provided Python script string.

    This function first parses the script string into an AST.
    It then validates the AST using `validate_python_script` to ensure it
    only contains safe operations. If validation passes, the AST is executed
    using `execute_safe_code`.

    Args:
        user_script: A string containing the Python code to be executed.

    Returns:
        The result of the script's last expression, or None if the script
        doesn't end with an expression or is empty.

    Raises:
        ValueError: If the script has a syntax error or contains disallowed
                    Python operations as per `validate_python_script`.
        Exception: Propagates any runtime exceptions that occur during the
                   execution of the script (e.g., ZeroDivisionError, NameError)
                   if the script is otherwise valid and safe.
    """
    try:
        script_ast = ast.parse(user_script)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in user script: {e}") from e

    if not validate_python_script(user_script): # validate_python_script re-parses, could optimize by passing AST
        # To optimize, validate_python_script could accept an AST directly.
        # For now, keeping it simple and calling with string as per its current signature.
        # Re-parsing is a small overhead for typical script sizes in this context.
        # A more robust way to get line numbers for disallowed nodes would be to use the AST
        # from the initial parse, but validate_python_script's visitor doesn't currently
        # expose that detail easily for this high-level message.
        raise ValueError("User script contains disallowed operations or is invalid.")

    # If validation passed, execute the code
    # execute_safe_code will propagate its own runtime exceptions
    return execute_safe_code(script_ast)


if __name__ == '__main__':
    # Example Usage and Tests for validate_python_script:

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

    print("\n--- Testing execute_safe_code ---")

    scripts_for_execution = {
        "simple_arithmetic": "a = 10\nb = 20\na + b",
        "string_literal_result": "'hello world'",
        "list_result": "[1, 2, 3*2]",
        "dict_result": "{'key': 1+1}",
        "last_stmt_assign": "x = 1\ny = x + 2", # Expected: None
        "empty_script": "", # Expected: None
        "only_pass": "pass", # Expected: None
        "name_error_runtime": "a + b", # Expected: NameError
        "zero_division_runtime": "1 / 0", # Expected: ZeroDivisionError
        "complex_expr_result": "x = [10, 20]\ny = { 'a': 1 }\nx[0] + y['a']",
        "if_expr_result": "limit = 10\nval = 5\n'ok' if val < limit else 'too high'",
        "list_comp_result": "squares = [i*i for i in [1,2,3,4]]\nsquares[-1]",
    }

    for name, script_text in scripts_for_execution.items():
        print(f"\nExecuting script: {name}")
        print(f"Code:\n{script_text}")
        
        is_valid_syntax = True
        try:
            script_ast = ast.parse(script_text)
        except SyntaxError:
            is_valid_syntax = False
            print("Result: SyntaxError during parsing (as expected for invalid Python, not covered by execute_safe_code tests directly)")
            continue

        is_safe = validate_python_script(script_text)
        print(f"Validated as safe: {is_safe}")

        if is_safe:
            try:
                result = execute_safe_code(script_ast)
                print(f"Execution Result: {result} (Type: {type(result).__name__})")
            except Exception as e:
                print(f"Execution Result: Exception - {type(e).__name__}: {e}")
        else:
            # If validate_python_script marks it as unsafe, we typically wouldn't execute it.
            # For testing, some "unsafe" (due to Call, Attribute etc.) might still be parsable
            # and could be run if execute_safe_code was called directly, but the flow is validate then execute.
            # The current unsafe_scripts list mostly contains things that are syntactically valid Python
            # but use disallowed AST nodes.
            # Example: "print('hello')" is unsafe by our rules. If forced into execute_safe_code,
            # it would cause NameError for 'print' if not in context.
            print("Result: Not executed because it's deemed unsafe by validate_python_script.")
            # Test if an "unsafe" script (by our rules) that is still simple Python would run
            # This part is more about understanding interactions if validation was bypassed
            if name == "name_error_runtime" or name == "zero_division_runtime": # These are "safe" by AST structure but cause runtime errors
                 # These are structurally safe but cause runtime errors, already handled above.
                 pass
            elif script_text == "import os": # This is unsafe and would also fail compilation if exec'd directly usually.
                print("   (Note: 'import os' would also fail execution if not handled by AST validation)")


    # Example of a script that is "safe" by AST structure but would cause NameError
    # if a variable is used before assignment, and that variable isn't part of an unsafe AST node.
    print("\n--- Testing specific execution scenarios ---")
    
    structurally_safe_name_error_script = "x = y + 1" # y is not defined
    print(f"\nExecuting script: structurally_safe_name_error\nCode:\n{structurally_safe_name_error_script}")
    script_ast_ne = ast.parse(structurally_safe_name_error_script)
    if validate_python_script(structurally_safe_name_error_script):
        try:
            execute_safe_code(script_ast_ne)
        except NameError as e:
            print(f"Execution Result: Correctly caught NameError: {e}")
        except Exception as e:
            print(f"Execution Result: Unexpected Exception - {type(e).__name__}: {e}")
    else:
        print("Result: Not executed (should be safe by structure, error in test logic if this prints)")

    print("\n--- Testing run_user_script ---")

    test_scripts_for_run_user_script = {
        "valid_simple_expr": ("a = 10\na + 5", 15),
        "valid_list_output": ("my_list = [1,2,3]\nmy_list", [1,2,3]),
        "valid_no_output_assign": ("b = 100", None),
        "valid_empty_script": ("", None),
        "valid_pass_script": ("pass", None),
        "invalid_syntax": ("a = 1 +", ValueError),
        "unsafe_import": ("import os", ValueError),
        "unsafe_print": ("print('hello')", ValueError), # Disallowed ast.Call
        "runtime_name_error": ("x = y + 1", NameError), # Structurally safe, but runtime error
        "runtime_zero_division": ("z = 1 / 0\nz", ZeroDivisionError), # Structurally safe, but runtime error
        "complex_safe_script_result": ("data = {'val': [10, 20]}\ndata['val'][0] * 2", 20),
    }

    for name, (script, expected) in test_scripts_for_run_user_script.items():
        print(f"\nRunning test case: {name}")
        print(f"Script:\n{script}")
        try:
            result = run_user_script(script)
            print(f"Actual Result: {result}")
            if isinstance(expected, type) and issubclass(expected, Exception):
                print(f"ERROR: Expected exception {expected.__name__}, but got result {result}")
            elif result != expected:
                print(f"ERROR: Expected {expected}, but got {result}")
            else:
                print("Result: PASSED")
        except Exception as e:
            print(f"Actual Exception: {type(e).__name__}: {e}")
            if isinstance(expected, type) and issubclass(expected, Exception):
                if isinstance(e, expected):
                    print("Result: PASSED (Caught expected exception)")
                else:
                    print(f"ERROR: Expected exception {expected.__name__}, but got {type(e).__name__}")
            else:
                print(f"ERROR: Expected result {expected}, but got exception {type(e).__name__}")

    structurally_safe_zero_division_script = "1/0"
    print(f"\nExecuting script: structurally_safe_zero_division\nCode:\n{structurally_safe_zero_division_script}")
    script_ast_zd = ast.parse(structurally_safe_zero_division_script)
    if validate_python_script(structurally_safe_zero_division_script): # This is an Expr(BinOp(...)) which is safe
        try:
            execute_safe_code(script_ast_zd)
        except ZeroDivisionError as e:
            print(f"Execution Result: Correctly caught ZeroDivisionError: {e}")
        except Exception as e:
            print(f"Execution Result: Unexpected Exception - {type(e).__name__}: {e}")
    else:
        print("Result: Not executed (should be safe by structure, error in test logic if this prints)")
