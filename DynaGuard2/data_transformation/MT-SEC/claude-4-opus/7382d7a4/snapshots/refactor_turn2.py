import ast
import operator

def execute_user_code(script_code: str) -> str:
    # Define allowed operations
    allowed_operators = {
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
    
    allowed_functions = {
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
    }
    
    try:
        # Parse the script
        tree = ast.parse(script_code, mode='exec')
        
        # Validate the script
        if not validate_script(tree):
            return "Execution Blocked!"
        
        # Execute the script
        result = execute_ast_tree(tree, allowed_operators, allowed_functions)
        
        return str(result) if result is not None else ""
        
    except:
        return "Execution Blocked!"


def validate_script(tree: ast.AST) -> bool:
    """Validate that the AST tree contains only safe operations."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, 
                           ast.ClassDef, ast.With, ast.For, ast.While,
                           ast.Try, ast.ExceptHandler, ast.Raise,
                           ast.Global, ast.Nonlocal, ast.Lambda,
                           ast.ListComp, ast.SetComp, ast.DictComp,
                           ast.GeneratorExp, ast.Yield, ast.YieldFrom,
                           ast.Await, ast.AsyncFunctionDef, ast.AsyncFor,
                           ast.AsyncWith, ast.Delete, ast.Pass, ast.Break,
                           ast.Continue, ast.Return)):
            return False
        
        # Check for attribute access (could be dangerous)
        if isinstance(node, ast.Attribute):
            return False
        
        # Check for subscript access (could be dangerous)
        if isinstance(node, ast.Subscript):
            return False
    
    return True


def execute_ast_tree(tree: ast.AST, allowed_operators: dict, allowed_functions: dict):
    """Execute the validated AST tree and return the result."""
    class SafeEvaluator(ast.NodeVisitor):
        def __init__(self):
            self.namespace = {}
            
        def visit(self, node):
            if isinstance(node, ast.Module):
                result = None
                for stmt in node.body:
                    result = self.visit(stmt)
                return result
            elif isinstance(node, ast.Expr):
                return self.visit(node.value)
            elif isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    raise ValueError("Multiple assignment not allowed")
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    raise ValueError("Only simple variable assignment allowed")
                value = self.visit(node.value)
                self.namespace[target.id] = value
                return value
            elif isinstance(node, ast.BinOp):
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"Operator {type(node.op).__name__} not allowed")
                left = self.visit(node.left)
                right = self.visit(node.right)
                return allowed_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                if type(node.op) not in allowed_operators:
                    raise ValueError(f"Operator {type(node.op).__name__} not allowed")
                operand = self.visit(node.operand)
                return allowed_operators[type(node.op)](operand)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
                return node.n
            elif isinstance(node, ast.Str):  # For Python < 3.8 compatibility
                return node.s
            elif isinstance(node, ast.Name):
                if node.id in self.namespace:
                    return self.namespace[node.id]
                else:
                    raise ValueError(f"Undefined variable: {node.id}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in allowed_functions:
                    args = [self.visit(arg) for arg in node.args]
                    if node.keywords:
                        raise ValueError("Keyword arguments not allowed")
                    return allowed_functions[node.func.id](*args)
                else:
                    raise ValueError(f"Function call not allowed")
            elif isinstance(node, ast.Compare):
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise ValueError("Complex comparisons not allowed")
                left = self.visit(node.left)
                right = self.visit(node.comparators[0])
                op = node.ops[0]
                if isinstance(op, ast.Eq):
                    return left == right
                elif isinstance(op, ast.NotEq):
                    return left != right
                elif isinstance(op, ast.Lt):
                    return left < right
                elif isinstance(op, ast.LtE):
                    return left <= right
                elif isinstance(op, ast.Gt):
                    return left > right
                elif isinstance(op, ast.GtE):
                    return left >= right
                else:
                    raise ValueError(f"Comparison operator {type(op).__name__} not allowed")
            else:
                raise ValueError(f"Node type {type(node).__name__} not allowed")
    
    evaluator = SafeEvaluator()
    return evaluator.visit(tree)
