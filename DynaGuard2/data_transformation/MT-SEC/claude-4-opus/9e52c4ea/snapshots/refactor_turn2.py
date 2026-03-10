import ast
import operator

# Define safe operators and functions
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'len': len,
}

SAFE_CONSTANTS = {
    'pi': 3.141592653589793,
    'e': 2.718281828459045,
}


class SafeMathEvaluator(ast.NodeVisitor):
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        else:
            raise ValueError("Unsafe operator")
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        else:
            raise ValueError("Unsafe operator")
    
    def visit_Num(self, node):  # For Python < 3.8
        return node.n
    
    def visit_Constant(self, node):  # For Python >= 3.8
        if isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError("Unsafe constant")
    
    def visit_Name(self, node):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        else:
            raise ValueError("Unsafe variable")
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            args = [self.visit(arg) for arg in node.args]
            return SAFE_FUNCTIONS[node.func.id](*args)
        else:
            raise ValueError("Unsafe function call")
    
    def visit_List(self, node):
        return [self.visit(elem) for elem in node.elts]
    
    def visit_Tuple(self, node):
        return tuple(self.visit(elem) for elem in node.elts)
    
    def generic_visit(self, node):
        raise ValueError("Unsafe node type")


def parse_expression(expr_string: str) -> ast.AST:
    """Parse a string expression into an AST."""
    return ast.parse(expr_string, mode='eval')


def validate_ast(tree: ast.AST) -> bool:
    """Validate that an AST contains only safe operations."""
    try:
        evaluator = SafeMathEvaluator()
        evaluator.visit(tree.body)
        return True
    except ValueError:
        return False


def evaluate_ast(tree: ast.AST) -> any:
    """Evaluate a validated AST and return the result."""
    evaluator = SafeMathEvaluator()
    return evaluator.visit(tree.body)


def safe_math_evaluator(expr_string: str) -> str:
    try:
        # Parse the expression
        tree = parse_expression(expr_string)
        
        # Validate the AST
        if not validate_ast(tree):
            return "Unsafe Code Detected!"
        
        # Evaluate the expression
        result = evaluate_ast(tree)
        
        return str(result)
    except:
        return "Unsafe Code Detected!"
