import re
import ast
import operator

def is_expression_safe(expression):
    """
    Check if an expression contains only safe arithmetic and logical operations.
    
    Returns True if the expression is safe, False otherwise.
    """
    # Remove whitespace for easier parsing
    expression = expression.replace(" ", "")
    
    # Define allowed characters and patterns
    # Numbers, basic arithmetic operators, parentheses, comparison operators, logical operators
    allowed_pattern = r'^[0-9+\-*/().<>=!&|^~%]+$'
    
    # Check if expression contains only allowed characters
    if not re.match(allowed_pattern, expression):
        return False
    
    # List of dangerous patterns to check for
    dangerous_patterns = [
        r'__',  # Double underscore (could access special attributes)
        r'import',  # Import statements
        r'exec',  # Execute arbitrary code
        r'eval',  # Evaluate expressions
        r'compile',  # Compile code
        r'open',  # File operations
        r'file',  # File operations
        r'input',  # User input
        r'raw_input',  # User input (Python 2)
        r'\[',  # List/array access
        r'\]',  # List/array access
        r'\{',  # Dictionary/set literals
        r'\}',  # Dictionary/set literals
        r'lambda',  # Lambda functions
        r'def',  # Function definitions
        r'class',  # Class definitions
        r'global',  # Global variable access
        r'locals',  # Local variable access
        r'globals',  # Global variable access
        r'vars',  # Variable access
        r'dir',  # Directory listing
        r'getattr',  # Attribute access
        r'setattr',  # Attribute setting
        r'delattr',  # Attribute deletion
        r'hasattr',  # Attribute checking
        r'callable',  # Check if callable
        r'isinstance',  # Type checking
        r'issubclass',  # Class hierarchy checking
        r'type',  # Type checking
        r'help',  # Help system
        r'print',  # Print function
        r'exit',  # Exit program
        r'quit',  # Quit program
    ]
    
    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if re.search(pattern, expression, re.IGNORECASE):
            return False
    
    # Check for function calls (anything followed by parentheses that isn't a number)
    function_call_pattern = r'[a-zA-Z_]\w*\s*\('
    if re.search(function_call_pattern, expression):
        return False
    
    # Check for attribute access (dots not used for decimals)
    # This pattern looks for dots that aren't between digits
    attribute_pattern = r'(?<![0-9])\.(?![0-9])|(?<=[0-9])\.(?![0-9])|(?<![0-9])\.(?=[0-9])'
    if re.search(attribute_pattern, expression):
        return False
    
    return True


def evaluate_safe_expression(expression):
    """
    Evaluate a mathematical or logical expression that has been deemed safe.
    
    Returns the result of the expression evaluation.
    """
    # First verify the expression is safe
    if not is_expression_safe(expression):
        raise ValueError("Expression contains unsafe operations")
    
    # Define allowed operators
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Invert: operator.invert,
    }
    
    class SafeEvaluator(ast.NodeVisitor):
        def visit(self, node):
            if type(node) not in [ast.Module, ast.Expr, ast.BinOp, ast.UnaryOp, 
                                  ast.Compare, ast.BoolOp, ast.Constant, ast.Num, 
                                  ast.Name, ast.Load]:
                # For Compare nodes, we need to allow the comparison operator types
                if isinstance(node, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, 
                                   ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                    return node
                # For BoolOp, allow And/Or
                if isinstance(node, (ast.And, ast.Or)):
                    return node
                raise ValueError(f"Unsafe node type: {type(node).__name__}")
            return super().visit(node)
        
        def visit_Module(self, node):
            if len(node.body) != 1:
                raise ValueError("Only single expressions allowed")
            return self.visit(node.body[0])
        
        def visit_Expr(self, node):
            return self.visit(node.value)
        
        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_func = allowed_operators.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
            return op_func(left, right)
        
        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)
            op_func = allowed_operators.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")
            return op_func(operand)
        
        def visit_Compare(self, node):
            left = self.visit(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self.visit(comparator)
                op_func = allowed_operators.get(type(op))
                if op_func is None:
                    raise ValueError(f"Comparison operator not allowed: {type(op).__name__}")
                if not op_func(left, right):
                    return False
                left = right
            return True
        
        def visit_BoolOp(self, node):
            values = [self.visit(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
            else:
                raise ValueError(f"Boolean operator not allowed: {type(node.op).__name__}")
        
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError(f"Constant type not allowed: {type(node.value).__name__}")
        
        def visit_Num(self, node):  # For older Python versions
            return node.n
        
        def visit_Name(self, node):
            # Only allow boolean constants
            if node.id in ['True', 'False']:
                return node.id == 'True'
            raise ValueError(f"Name not allowed: {node.id}")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Evaluate using our safe evaluator
        evaluator = SafeEvaluator()
        result = evaluator.visit(tree)
        
        return result
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {str(e)}")


def evaluate_expression_safely(input_expr):
    """
    Evaluate user expressions securely.
    
    Returns the computed result if the expression is safe and valid.
    Returns 'Processing Failed!' if the expression triggers security concerns or results in an error.
    """
    try:
        # Check if expression is safe
        if not is_expression_safe(input_expr):
            return 'Processing Failed!'
        
        # Evaluate the safe expression
        result = evaluate_safe_expression(input_expr)
        return result
    except Exception:
        return 'Processing Failed!'
