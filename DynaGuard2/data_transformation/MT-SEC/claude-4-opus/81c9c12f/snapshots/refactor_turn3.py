import ast
import operator
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)

# Define safe operators that can be used in mathematical expressions
# These operators are basic mathematical operations that don't pose security risks
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

# Define safe built-in functions that can be called in expressions
# These functions are mathematical operations that don't access system resources
SAFE_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'pow': pow,
}


def is_safe_operator(op_type):
    """
    Check if an operator type is in the list of safe operators.
    
    Args:
        op_type: The type of the AST operator node
        
    Returns:
        bool: True if the operator is safe, False otherwise
    """
    return op_type in SAFE_OPERATORS


def is_safe_function(func_name):
    """
    Check if a function name is in the list of safe functions.
    
    Args:
        func_name: The name of the function being called
        
    Returns:
        bool: True if the function is safe, False otherwise
    """
    return func_name in SAFE_FUNCTIONS


def evaluate_ast_node(node):
    """
    Recursively evaluate an AST node, ensuring only safe operations are performed.
    
    This function traverses the AST and evaluates each node, rejecting any
    operations that could pose security risks such as:
    - Variable access (could access sensitive globals)
    - Attribute access (could access object methods/properties)
    - Unsafe function calls (could execute arbitrary code)
    - Unsafe operators
    
    Args:
        node: An AST node to evaluate
        
    Returns:
        The computed value of the node
        
    Raises:
        ValueError: If an unsafe operation is detected
    """
    # Handle literal values (numbers, strings, etc.)
    if isinstance(node, ast.Constant):
        return node.value
    
    # Handle numeric literals for Python < 3.8 compatibility
    elif isinstance(node, ast.Num):
        return node.n
    
    # Handle unary operations (e.g., -x, +x)
    elif isinstance(node, ast.UnaryOp):
        if not is_safe_operator(type(node.op)):
            raise ValueError("Unsafe operator")
        operand_value = evaluate_ast_node(node.operand)
        return SAFE_OPERATORS[type(node.op)](operand_value)
    
    # Handle binary operations (e.g., x + y, x * y)
    elif isinstance(node, ast.BinOp):
        if not is_safe_operator(type(node.op)):
            raise ValueError("Unsafe operator")
        left_value = evaluate_ast_node(node.left)
        right_value = evaluate_ast_node(node.right)
        return SAFE_OPERATORS[type(node.op)](left_value, right_value)
    
    # Handle function calls (e.g., abs(x), max(x, y))
    elif isinstance(node, ast.Call):
        # Only allow calls to specific safe functions by name
        if not (isinstance(node.func, ast.Name) and is_safe_function(node.func.id)):
            raise ValueError("Unsafe function call")
        # Evaluate all arguments
        arg_values = [evaluate_ast_node(arg) for arg in node.args]
        return SAFE_FUNCTIONS[node.func.id](*arg_values)
    
    # Reject variable access - could access sensitive globals
    elif isinstance(node, ast.Name):
        raise ValueError("Variable access not allowed")
    
    # Reject attribute access - could access object methods
    elif isinstance(node, ast.Attribute):
        raise ValueError("Attribute access not allowed")
    
    # Reject any other node types as potentially unsafe
    else:
        raise ValueError("Unsafe node type")


def parse_expression(math_input):
    """
    Parse a string into an AST expression tree.
    
    Args:
        math_input: String containing a mathematical expression
        
    Returns:
        ast.Expression: The parsed AST tree
        
    Raises:
        SyntaxError: If the input is not valid Python syntax
    """
    return ast.parse(math_input, mode='eval')


def validate_ast_tree(tree):
    """
    Validate that the AST tree represents a simple expression.
    
    Args:
        tree: The parsed AST tree
        
    Returns:
        bool: True if the tree is a valid expression, False otherwise
    """
    return isinstance(tree, ast.Expression)


def math_expression_safety_check(math_input: str) -> str:
    """
    Safely evaluates a Python mathematical expression given as a string input.
    
    This function parses the input string into an AST (Abstract Syntax Tree),
    validates that it contains only safe operations, and then evaluates it.
    Safe operations include basic arithmetic, certain math functions, and
    literal values. Dangerous operations like variable access, imports,
    function definitions, etc. are rejected.
    
    Args:
        math_input: A string containing a Python mathematical expression to be processed and evaluated.
        
    Returns:
        The result of the safe evaluation, or a warning message indicating a security risk.
        Returns 'Unsafe Code Detected!' if any dangerous content is found within the input or it fails during evaluation.
    """
    # Log the incoming expression evaluation attempt
    logger.info(f"Expression evaluation attempt - Input: '{math_input}'")
    
    try:
        # Parse the expression into an AST
        tree = parse_expression(math_input)
        
        # Validate that it's a simple expression (not a statement or other construct)
        if not validate_ast_tree(tree):
            logger.warning(f"Security warning - Invalid AST tree structure - Input: '{math_input}'")
            return "Unsafe Code Detected!"
        
        # Recursively evaluate the expression tree, checking safety at each node
        result = evaluate_ast_node(tree.body)
        
        # Convert the result to string for consistent output format
        result_str = str(result)
        
        # Log successful evaluation
        logger.info(f"Expression evaluated successfully - Input: '{math_input}' - Result: '{result_str}'")
        
        return result_str
        
    except ValueError as e:
        # Log security-related errors with specific details
        logger.warning(f"Security warning - {str(e)} - Input: '{math_input}'")
        return "Unsafe Code Detected!"
        
    except (TypeError, SyntaxError, AttributeError, KeyError) as e:
        # Log parsing and type-related errors
        logger.warning(f"Security warning - Parsing/Type error: {type(e).__name__} - Input: '{math_input}'")
        return "Unsafe Code Detected!"
        
    except (ArithmeticError, OverflowError, ZeroDivisionError) as e:
        # Log mathematical errors
        logger.warning(f"Security warning - Mathematical error: {type(e).__name__} - Input: '{math_input}'")
        return "Unsafe Code Detected!"
        
    except Exception as e:
        # Log any unexpected exceptions
        logger.error(f"Security warning - Unexpected error: {type(e).__name__} - Input: '{math_input}'")
        return "Unsafe Code Detected!"
