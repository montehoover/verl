import ast

class OperationVisitor(ast.NodeVisitor):
    def __init__(self):
        self.operations = []
        # Whitelist of AST node types allowed in the script
        self.allowed_nodes = (
            ast.Module, ast.Expr, ast.Constant, ast.Name, ast.Load,
            ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
            ast.Assign, ast.Store,
            # For simple data structures, if ever needed:
            # ast.Tuple, ast.List, ast.Dict, ast.keyword
        )
        
        # Whitelist of allowed operators
        self.allowed_bin_ops = {
            ast.Add: "addition/concatenation",
            ast.Sub: "subtraction",
            ast.Mult: "multiplication",
            ast.Div: "division",
            ast.FloorDiv: "floor division",
            ast.Mod: "modulo",
            ast.Pow: "power",
        }
        self.allowed_unary_ops = {
            ast.USub: "unary subtraction",
            ast.UAdd: "unary addition",
            ast.Not: "logical not",
        }
        self.allowed_compare_ops = {
            ast.Eq: "equality",
            ast.NotEq: "inequality",
            ast.Lt: "less than",
            ast.LtE: "less than or equal",
            ast.Gt: "greater than",
            ast.GtE: "greater than or equal",
            # ast.Is / ast.IsNot could be added if needed
            # ast.In / ast.NotIn could be added if needed
        }
        self.allowed_bool_ops = {
            ast.And: "logical and",
            ast.Or: "logical or",
        }

    def visit(self, node):
        """
        Generic visit method to check if the node type is allowed.
        If not, it raises a ValueError.
        """
        if not isinstance(node, self.allowed_nodes):
            # Specific check for Call nodes, as they are a common source of vulnerabilities
            if isinstance(node, ast.Call):
                func_name = "unknown function"
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Attempt to reconstruct like 'obj.method'
                    # This can be complex; for now, a simple representation
                    value_name = "object"
                    if isinstance(node.func.value, ast.Name):
                        value_name = node.func.value.id
                    func_name = f"{value_name}.{node.func.attr}"
                
                raise ValueError(f"Disallowed operation: Function call to '{func_name}'")
            raise ValueError(f"Disallowed AST node type: {type(node).__name__}")
        super().visit(node) # Continue traversal for allowed nodes

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type in self.allowed_bin_ops:
            self.operations.append(self.allowed_bin_ops[op_type])
        else:
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type in self.allowed_unary_ops:
            self.operations.append(self.allowed_unary_ops[op_type])
        else:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            op_type = type(op)
            if op_type in self.allowed_compare_ops:
                self.operations.append(self.allowed_compare_ops[op_type])
            else:
                raise ValueError(f"Disallowed comparison operator: {op_type.__name__}")
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        op_type = type(node.op)
        if op_type in self.allowed_bool_ops:
            self.operations.append(self.allowed_bool_ops[op_type])
        else:
            raise ValueError(f"Disallowed boolean operator: {op_type.__name__}")
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Ensure assignment targets are simple names, not attributes or subscripts
        for target in node.targets:
            if not isinstance(target, ast.Name): # Implicitly checks for ast.Store via allowed_nodes
                raise ValueError(f"Disallowed assignment target type: {type(target).__name__}")
        # "Assignment" itself is not an "operation" in the sense of arithmetic/string ops,
        # but we must visit its children (value being assigned) to find operations.
        self.generic_visit(node)


def parse_script_operations(script_string: str) -> list:
    """
    Parses a script string, identifies basic arithmetic and string operations,
    and returns a list of these operations.
    Ensures no potentially harmful operations are present by whitelisting AST nodes
    and specific operator types.

    Args:
        script_string: The Python script content as a string.

    Returns:
        A list of strings, where each string describes an identified operation.

    Raises:
        ValueError: If the script contains syntax errors, disallowed AST node types,
                    disallowed operators, or performs disallowed actions like
                    unsafe assignments or function calls.
    """
    if not isinstance(script_string, str):
        raise TypeError("Input script must be a string.")

    try:
        tree = ast.parse(script_string)
    except SyntaxError as e:
        # Provide more context for syntax errors
        raise ValueError(f"Invalid Python script syntax: {e.msg} at line {e.lineno}, offset {e.offset}")

    visitor = OperationVisitor()
    
    # The visitor's methods will raise ValueError if disallowed operations are found.
    # This exception will propagate up.
    visitor.visit(tree)
    
    return visitor.operations


def execute_operations(script_string: str) -> dict | str:
    """
    Executes a script string after validating it for safety.
    The script is executed in a restricted environment.

    Args:
        script_string: The Python script content as a string.

    Returns:
        A dictionary containing the local variables from the executed script
        on success, or a string error message on failure (validation, compilation,
        or runtime).
    """
    if not isinstance(script_string, str):
        # parse_script_operations would also catch this, but good to be explicit.
        return "Error: Input script must be a string."

    try:
        # Step 1: Validate the script using parse_script_operations.
        # This will raise ValueError if the script contains disallowed operations or syntax.
        parse_script_operations(script_string)

        # Step 2: If validation passed, parse the AST again for compilation.
        # (Alternatively, parse_script_operations could return the AST tree on success
        # to avoid re-parsing, but that would change its current contract.)
        tree = ast.parse(script_string)
        
        # Step 3: Compile the AST into a code object.
        # filename='<user_script>' is a common convention for dynamically generated code.
        code_obj = compile(tree, filename='<user_script>', mode='exec')
        
        # Step 4: Prepare a restricted environment for execution.
        # By default, no builtins or global variables are accessible.
        # This is a strong safety measure. If specific safe builtins are ever needed
        # (e.g., abs, round), they must be explicitly added to `safe_globals["__builtins__"]`.
        # The OperationVisitor already disallows ast.Call, so calls to unknown builtins
        # would be caught during parsing.
        safe_globals = {"__builtins__": {}}
        
        locals_dict = {}  # This dictionary will store variables created by the script.
        
        # Step 5: Execute the compiled code.
        exec(code_obj, safe_globals, locals_dict)
        
        return locals_dict  # Return the namespace with resulting variables.

    except ValueError as e:
        # Catches errors from parse_script_operations (validation) or ast.parse/compile.
        return f"Error during parsing, validation, or compilation: {e}"
    except Exception as e:
        # Catches runtime errors from the exec() call (e.g., ZeroDivisionError, TypeError).
        return f"Error during execution: {type(e).__name__}: {e}"

# Example Usage (can be commented out or removed if not needed as part of the file)
# if __name__ == '__main__':
#     # Safe examples
#     safe_script_1 = "x = 1 + 2\ny = x * (5 - 3)\nz = 'hello' + ' ' + 'world'"
#     safe_script_2 = "a = 10 / 2\nb = 7 // 3\nc = 5 % 2\nd = 2 ** 3"
#     safe_script_3 = "val = -5\nis_true = not False\ncompare = (1 < 2) and (3 >= 3)"

#     print(f"Script 1: {safe_script_1}")
#     try:
#         ops1 = parse_script_operations(safe_script_1)
#         print(f"Operations: {ops1}\n") # Expected: ['addition/concatenation', 'subtraction', 'multiplication', 'addition/concatenation', 'addition/concatenation'] (order may vary slightly based on AST traversal details)
#     except ValueError as e:
#         print(f"Error: {e}\n")

#     print(f"Script 2: {safe_script_2}")
#     try:
#         ops2 = parse_script_operations(safe_script_2)
#         print(f"Operations: {ops2}\n")
#     except ValueError as e:
#         print(f"Error: {e}\n")

#     print(f"Script 3: {safe_script_3}")
#     try:
#         ops3 = parse_script_operations(safe_script_3)
#         print(f"Operations: {ops3}\n")
#     except ValueError as e:
#         print(f"Error: {e}\n")

#     # Unsafe or disallowed examples
#     unsafe_script_1 = "import os"
#     unsafe_script_2 = "print('hello')"
#     unsafe_script_3 = "eval('1+1')"
#     unsafe_script_4 = "x = lambda: os.system('clear')" # Lambda involves ast.Lambda, not in allowed_nodes
#     unsafe_script_5 = "my_list = [1,2,3]\nmy_list[0] = 0" # Subscript store
#     unsafe_script_6 = "class Foo: pass"
#     unsafe_script_7 = "def bar(): pass"
#     unsafe_script_8 = "a.b = 10" # Attribute assignment

#     unsafe_scripts = {
#         "Import OS": unsafe_script_1,
#         "Print Call": unsafe_script_2,
#         "Eval Call": unsafe_script_3,
#         "Lambda with OS Call": unsafe_script_4,
#         "Subscript Assignment": unsafe_script_5,
#         "Class Definition": unsafe_script_6,
#         "Function Definition": unsafe_script_7,
#         "Attribute Assignment": unsafe_script_8,
#     }

#     for name, script in unsafe_scripts.items():
#         print(f"Unsafe Script ({name}): {script}")
#         try:
#             parse_script_operations(script)
#             print("Operations: No error raised, but expected one.\n")
#         except ValueError as e:
#             print(f"Caught expected error: {e}\n")

#     # Example with syntax error
#     syntax_error_script = "x = 1 + "
#     print(f"Syntax Error Script: {syntax_error_script}")
#     try:
#         parse_script_operations(syntax_error_script)
#     except ValueError as e:
#         print(f"Caught expected error: {e}\n")

#     # Examples for execute_operations
#     print("--- Testing execute_operations ---")
#     script_to_execute_ok = "a = 10 + 5\nb = a * 2\nc = 'test' + ' ' + 'run'"
#     print(f"Executing script: \"{script_to_execute_ok}\"")
#     result = execute_operations(script_to_execute_ok)
#     print(f"Result: {result}\n") # Expected: {'a': 15, 'b': 30, 'c': 'test run'}

#     script_with_runtime_error = "x = 1 / 0"
#     print(f"Executing script: \"{script_with_runtime_error}\"")
#     result = execute_operations(script_with_runtime_error)
#     print(f"Result: {result}\n") # Expected: Error message about ZeroDivisionError

#     script_with_unsafe_op_import = "import os" # This will be caught by parse_script_operations
#     print(f"Executing script: \"{script_with_unsafe_op_import}\"")
#     result = execute_operations(script_with_unsafe_op_import)
#     print(f"Result: {result}\n") # Expected: Error message from parse_script_operations

#     script_calling_builtin = "x = len('hello')" # Also caught by parse_script_operations (ast.Call)
#     print(f"Executing script: \"{script_calling_builtin}\"")
#     result = execute_operations(script_calling_builtin)
#     print(f"Result: {result}\n") # Expected: Error message about Disallowed operation: Function call

#     empty_script = ""
#     print(f"Executing script: \"{empty_script}\"")
#     result = execute_operations(empty_script)
#     print(f"Result: {result}\n") # Expected: {} or error depending on ast.parse behavior for empty string

#     script_with_only_comments = "# This is just a comment"
#     print(f"Executing script: \"{script_with_only_comments}\"")
#     result = execute_operations(script_with_only_comments)
#     print(f"Result: {result}\n") # Expected: {}
    
#     none_string_input = 123
#     print(f"Executing script: {none_string_input} (as non-string)")
#     result = execute_operations(none_string_input)
#     print(f"Result: {result}\n") # Expected: Error: Input script must be a string.
