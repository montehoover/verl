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
