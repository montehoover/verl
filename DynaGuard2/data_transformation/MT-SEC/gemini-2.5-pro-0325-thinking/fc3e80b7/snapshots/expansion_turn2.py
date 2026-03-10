import ast

class OperationVisitor(ast.NodeVisitor):
    """
    AST visitor to identify basic arithmetic and string operations.
    """
    def __init__(self):
        self.operations = set()
        self.string_methods = {
            "upper", "lower", "split", "join", "replace",
            "find", "strip", "lstrip", "rstrip"
        }
        self.arithmetic_ops_map = {
            ast.Add: "Addition/Concatenation (+)",
            ast.Sub: "Subtraction (-)",
            ast.Mult: "Multiplication (*)",
            ast.Div: "Division (/)",
            ast.FloorDiv: "Floor Division (//)",
            ast.Mod: "Modulo (%)",
            ast.Pow: "Exponentiation (**)",
        }

    def visit_BinOp(self, node: ast.BinOp):
        """
        Visits binary operation nodes (e.g., +, -, *).
        """
        op_type = type(node.op)
        if op_type in self.arithmetic_ops_map:
            self.operations.add(self.arithmetic_ops_map[op_type])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """
        Visits function call nodes to identify string method calls.
        """
        if isinstance(node.func, ast.Attribute):
            # This is a method call, e.g., "string".upper()
            # node.func.value would be the object (e.g., ast.Constant for "string")
            # node.func.attr is the method name (e.g., "upper")
            method_name = node.func.attr
            if method_name in self.string_methods:
                self.operations.add(f"String method .{method_name}()")
        self.generic_visit(node)

def parse_script_operations(script_content: str) -> list[str]:
    """
    Parses a Python script string and returns a sorted list of unique
    basic arithmetic and string manipulation operations it contains.

    Args:
        script_content: A string containing the Python script.

    Returns:
        A sorted list of strings, where each string describes an operation.
        Returns ["Invalid Python script"] if the script cannot be parsed.
    """
    try:
        tree = ast.parse(script_content)
    except SyntaxError:
        return ["Invalid Python script"]
    
    visitor = OperationVisitor()
    visitor.visit(tree)
    return sorted(list(visitor.operations))


def evaluate_operations(script_content: str):
    """
    Evaluates a Python script string if it contains only safe operations
    (basic arithmetic: +, -, *, / and string concatenation) and is a single expression.

    Args:
        script_content: A string containing the Python script.

    Returns:
        The result of the evaluation if successful.
        An error message string if the script is invalid, contains unsafe
        operations, is not a single expression, or if evaluation fails.
    """
    try:
        tree = ast.parse(script_content)
    except SyntaxError:
        return "Error: Invalid Python script"

    visitor = OperationVisitor()
    visitor.visit(tree)
    detected_ops = visitor.operations

    # Define allowed operations based on their descriptions from OperationVisitor
    # Basic arithmetic: +, -, *, /
    # String concatenation is covered by "Addition/Concatenation (+)"
    allowed_operation_descriptions = {
        "Addition/Concatenation (+)",
        "Subtraction (-)",
        "Multiplication (*)",
        "Division (/)",
    }

    for op_desc in detected_ops:
        if op_desc not in allowed_operation_descriptions:
            return f"Error: Safety violation - Unsafe operation detected: {op_desc}"

    if not tree.body: # Handles empty script or script with only comments
        return "Error: Script must be a single expression for evaluation."
        
    if not (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)):
        return "Error: Script must be a single expression for evaluation."

    expression_node = tree.body[0].value

    try:
        # ast.literal_eval is safe and evaluates literals and basic operations on them.
        # It will raise errors for variables, function calls (other than constructors), etc.
        result = ast.literal_eval(expression_node)
        return result
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
        return f"Error during evaluation: Malformed or unsupported expression - {type(e).__name__}: {e}"
    except ZeroDivisionError:
        return "Error during evaluation: Division by zero"
    except Exception as e: # Catch any other unexpected errors during literal_eval
        return f"Error during evaluation: An unexpected error occurred - {type(e).__name__}: {e}"


if __name__ == '__main__':
    # Example Usage:
    script1 = """
a = 1 + 2 - 3
b = "hello" + " " + "world"
c = "test".upper().lower()
d = (x / y) // z
e = "  spaced  ".strip()
f = "replace me".replace("me", "you")
g = "find_this".find("this")
h = ["word1", "word2"]
i = " ".join(h)
j = "split,this,string".split(',')
k = l % m
l = m ** n
    """
    operations1 = parse_script_operations(script1)
    print(f"Operations in script1: {operations1}")

    script2 = "print('This is a valid script but has no targeted ops')"
    operations2 = parse_script_operations(script2)
    print(f"Operations in script2: {operations2}")

    script3 = "a = 1 + 'string' # This might be a TypeError at runtime but AST is valid"
    operations3 = parse_script_operations(script3)
    print(f"Operations in script3: {operations3}")
    
    script4_invalid = "a = 1 + "
    operations4 = parse_script_operations(script4_invalid)
    print(f"Operations in script4 (invalid): {operations4}")

    script5_empty = ""
    operations5 = parse_script_operations(script5_empty)
    print(f"Operations in script5 (empty): {operations5}")

    print("\n--- Testing evaluate_operations ---")
    test_scripts_eval = {
        "safe_add": "1 + 2",
        "safe_sub": "10 - 3",
        "safe_mult": "3 * 7",
        "safe_div": "10 / 2",
        "safe_str_concat": "'hello' + ' ' + 'world'",
        "safe_combined": "(1 + 2) * 3 - 10 / 5", # (3*3) - 2 = 9 - 2 = 7
        "safe_literal_num": "123.45",
        "safe_literal_str": "'test string'",
        "unsafe_pow": "2 ** 3",
        "unsafe_string_method": "'test'.upper()",
        "unsafe_variable": "a + 1",
        "unsafe_statement": "a = 1",
        "eval_div_by_zero": "1 / 0",
        "eval_empty_script": "",
        "eval_comment_only": "# just a comment",
        "eval_invalid_syntax": "1 +",
        "eval_complex_unsafe": "[1,2, 'abc'.upper()]"
    }

    for name, script in test_scripts_eval.items():
        print(f"\nEvaluating script '{name}': \"{script}\"")
        parsed_ops = parse_script_operations(script)
        print(f"Parsed operations: {parsed_ops}")
        result = evaluate_operations(script)
        print(f"Evaluation result: {result} (Type: {type(result).__name__})")
