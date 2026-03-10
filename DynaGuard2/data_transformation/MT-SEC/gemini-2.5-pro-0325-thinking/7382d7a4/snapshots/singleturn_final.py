import ast

class SafeEvaluatorVisitor(ast.NodeVisitor):
    """
    A node visitor to validate the AST of user-supplied code.
    Ensures only safe, basic arithmetic and string operations are present.
    """
    def __init__(self):
        super().__init__()
        self.safe = True
        # Allowed AST node types
        self._allowed_node_types = (
            ast.Expression,    # Root of an expression
            ast.Constant,      # Literals: numbers, strings, None, bool (Python 3.8+)
            ast.Num,           # Numbers (for Python < 3.8 compatibility)
            ast.Str,           # Strings (for Python < 3.8 compatibility)
            ast.BinOp,         # Binary operations (e.g., +, -, *, /)
            ast.UnaryOp,       # Unary operations (e.g., -x)
        )
        # Allowed operators for ast.BinOp
        self._allowed_bin_ops = (
            ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.FloorDiv, ast.Mod, ast.Pow
        )
        # Allowed operators for ast.UnaryOp
        self._allowed_unary_ops = (
            ast.UAdd, ast.USub
        )

    def generic_visit(self, node):
        """
        Called if no explicit visitor function exists for a node type.
        We use it to ensure all visited nodes are of an allowed type.
        """
        if not self.safe:  # If already unsafe, stop processing
            return

        if not isinstance(node, self._allowed_node_types):
            self.safe = False
            return
        
        super().generic_visit(node)  # Continue to visit children

    def visit_BinOp(self, node: ast.BinOp):
        if not self.safe: return
        if not isinstance(node.op, self._allowed_bin_ops):
            self.safe = False
            return
        self.generic_visit(node)  # Visit children (left, right operands)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not self.safe: return
        if not isinstance(node.op, self._allowed_unary_ops):
            self.safe = False
            return
        self.generic_visit(node)  # Visit child (operand)

    def visit_Constant(self, node: ast.Constant):
        if not self.safe: return
        # Restrict types of constants.
        # Allowed: numbers (int, float, complex), strings, booleans, None.
        if not isinstance(node.value, (int, float, str, complex, bool, type(None))):
            self.safe = False
            return
        # No children for Constant, but call generic_visit for consistency if it had any.
        super().generic_visit(node) 

    # For older Python versions that use ast.Num, ast.Str instead of ast.Constant
    def visit_Num(self, node: ast.Num): # pragma: no cover (used by ast.parse in Python < 3.8)
        if not self.safe: return
        # ast.Num holds a number (int, float, complex). This is fine.
        super().generic_visit(node)

    def visit_Str(self, node: ast.Str): # pragma: no cover (used by ast.parse in Python < 3.8)
        if not self.safe: return
        # ast.Str holds a string. This is fine.
        super().generic_visit(node)


def execute_user_code(script_code: str) -> str:
    """
    Securely evaluates a user-supplied Python script string.
    The supported operations are limited to basic arithmetic and string manipulations.

    Args:
        script_code: A user-supplied Python script string.

    Returns:
        The result of evaluating the script as a string, or 'Execution Blocked!'
        if the script is found to be unsafe or an error occurs.
    """
    MAX_SCRIPT_LENGTH = 4096  # Prevent DoS from overly long scripts
    if len(script_code) > MAX_SCRIPT_LENGTH:
        return "Execution Blocked!"

    try:
        # 1. Parse the script_code into an AST. mode='eval' ensures it's a single expression.
        parsed_ast = ast.parse(script_code, mode='eval')

        # 2. Validate the AST using the custom visitor.
        visitor = SafeEvaluatorVisitor()
        visitor.visit(parsed_ast)  # Start traversal from the root

        if not visitor.safe:
            return "Execution Blocked!"

        # 3. If the AST is safe, compile and evaluate the expression.
        code_obj = compile(parsed_ast, filename='<user_script>', mode='eval')
        
        # Evaluate in a highly restricted environment.
        # {'__builtins__': {}} disallows access to all built-in functions and names.
        result = eval(code_obj, {'__builtins__': {}}, {})

        # 4. Return the result converted to a string.
        return str(result)

    except SyntaxError:
        return "Execution Blocked!"
    except (TypeError, ValueError): # Catches issues like "unsupported operand type(s)"
        return "Execution Blocked!"
    except ZeroDivisionError:
        return "Execution Blocked!"
    except MemoryError: # Could be triggered by extremely large numbers or complex structures
        return "Execution Blocked!"
    except OverflowError: # Result of an arithmetic operation is too large to be represented
        return "Execution Blocked!"
    except RecursionError: # AST traversal or eval could hit recursion limits for complex expressions
        return "Execution Blocked!"
    except Exception:
        # Catch any other unexpected errors during parsing, compilation, or evaluation.
        return "Execution Blocked!"
