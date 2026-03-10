import re
import ast
import operator
import math

def parse_expression(expression):
    """
    Parse a mathematical expression and determine if it's valid and safe.
    
    Args:
        expression (str): The mathematical expression to parse
        
    Returns:
        bool: True if the expression is valid and safe, False otherwise
    """
    if not expression or not isinstance(expression, str):
        return False
    
    # Remove whitespace
    expression = expression.strip()
    
    if not expression:
        return False
    
    # Check for forbidden characters and patterns
    # Allow only numbers, operators, parentheses, and decimal points
    allowed_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(allowed_pattern, expression):
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'__',  # Double underscore (could be trying to access special attributes)
        r'import',  # Import statements
        r'exec',  # Exec function
        r'eval',  # Eval function
        r'compile',  # Compile function
        r'open',  # File operations
        r'file',  # File operations
        r'input',  # Input function
        r'raw_input',  # Raw input function
        r'\[',  # List comprehensions
        r'\]',  # List comprehensions
        r'{',  # Dictionary/set comprehensions
        r'}',  # Dictionary/set comprehensions
        r'lambda',  # Lambda functions
        r'def',  # Function definitions
        r'class',  # Class definitions
        r'for',  # Loops
        r'while',  # Loops
        r'if',  # Conditionals
        r'else',  # Conditionals
        r'elif',  # Conditionals
        r'try',  # Exception handling
        r'except',  # Exception handling
        r'raise',  # Exception handling
        r'assert',  # Assertions
        r'global',  # Global declarations
        r'nonlocal',  # Nonlocal declarations
        r'del',  # Delete statements
        r'with',  # Context managers
        r'yield',  # Generators
        r'return',  # Return statements
        r'pass',  # Pass statements
        r'break',  # Break statements
        r'continue',  # Continue statements
    ]
    
    expression_lower = expression.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, expression_lower):
            return False
    
    # Check for valid parentheses matching
    paren_count = 0
    for char in expression:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                return False
    
    if paren_count != 0:
        return False
    
    # Check for empty parentheses
    if '()' in expression:
        return False
    
    # Check for multiple consecutive operators
    if re.search(r'[+\-*/]{2,}', expression):
        # Allow -- and ++ as they could be double negation or positive
        if not re.match(r'^[+\-]+$', re.search(r'[+\-*/]{2,}', expression).group()):
            return False
    
    # Try to parse as AST to check structure
    try:
        tree = ast.parse(expression, mode='eval')
        
        # Check that the AST only contains allowed node types
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Num,  # For Python < 3.8
            ast.Constant,  # For Python >= 3.8
        )
        
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False
    except:
        return False
    
    return True


def evaluate_safe_expression(expression):
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        float or str: The result of the computation, or an error message
    """
    # First validate the expression
    if not parse_expression(expression):
        return "Error: Invalid or unsafe expression"
    
    # Define allowed operations
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def safe_power(a, b):
        # Prevent excessively large exponents
        if isinstance(b, (int, float)) and abs(b) > 100:
            raise ValueError("Exponent too large")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(a) > 1000 and b > 10:
                raise ValueError("Power operation too large")
        return operator.pow(a, b)
    
    # Override power operation with safe version
    operators[ast.Pow] = safe_power
    
    class SafeEvaluator(ast.NodeVisitor):
        def visit(self, node):
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            return visitor(node)
        
        def generic_visit(self, node):
            raise ValueError(f"Unsafe node type: {type(node).__name__}")
        
        def visit_Expression(self, node):
            return self.visit(node.body)
        
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)):
                # Check for excessively large numbers
                if abs(node.value) > 1e308:
                    raise ValueError("Number too large")
                return node.value
            raise ValueError("Only numeric constants allowed")
        
        def visit_Num(self, node):  # For Python < 3.8
            if isinstance(node.n, (int, float)):
                # Check for excessively large numbers
                if abs(node.n) > 1e308:
                    raise ValueError("Number too large")
                return node.n
            raise ValueError("Only numeric constants allowed")
        
        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            
            # Additional safety checks for division
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")
            
            # Additional safety checks for modulo
            if isinstance(node.op, ast.Mod) and right == 0:
                raise ValueError("Modulo by zero")
            
            return operators[type(node.op)](left, right)
        
        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)
            return operators[type(node.op)](operand)
    
    try:
        # Parse the expression
        tree = ast.parse(expression.strip(), mode='eval')
        
        # Evaluate using our safe evaluator
        evaluator = SafeEvaluator()
        result = evaluator.visit(tree)
        
        # Final safety check on result
        if isinstance(result, (int, float)):
            if math.isnan(result):
                return "Error: Result is NaN"
            if math.isinf(result):
                return "Error: Result is infinity"
            if abs(result) > 1e308:
                return "Error: Result too large"
            return result
        else:
            return "Error: Invalid result type"
            
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: {str(e)}"
    except OverflowError:
        return "Error: Arithmetic overflow"
    except Exception as e:
        return f"Error: Safety violation - {type(e).__name__}"
