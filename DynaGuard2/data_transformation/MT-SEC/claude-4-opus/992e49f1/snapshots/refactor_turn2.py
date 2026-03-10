import ast


def safe_execute_operation(operation: str):
    """
    Safely execute a mathematical operation from user input.
    
    This function validates and evaluates mathematical expressions provided as strings,
    ensuring that only safe mathematical operations are performed. It uses AST parsing
    to validate the structure of the expression before evaluation.
    
    Args:
        operation (str): The mathematical operation to be evaluated as a string.
                        Supports basic arithmetic operations: +, -, *, /, and parentheses.
                        Numbers can include decimal points.
        
    Returns:
        float or int: The result of the evaluated mathematical operation.
        
    Raises:
        ValueError: Raised in the following cases:
                   - If the operation string is empty or contains only whitespace
                   - If unsafe or invalid characters are detected in the input
                   - If the operation has invalid syntax or structure
                   - If division by zero occurs
                   - If any other evaluation error occurs
                   
    Examples:
        >>> safe_execute_operation("2 + 3 * 4")
        14
        >>> safe_execute_operation("(10 - 5) / 2")
        2.5
    """
    # Validate that the operation is not empty
    if not operation or not operation.strip():
        raise ValueError("Operation cannot be empty")
    
    # Define the set of allowed characters for mathematical operations
    # This includes digits, basic arithmetic operators, parentheses, decimal point, and spaces
    allowed_chars = set('0123456789+-*/()., ')
    
    # Check that all characters in the operation string are allowed
    if not all(char in allowed_chars for char in operation):
        raise ValueError("Invalid characters detected in operation")
    
    # Remove all spaces from the operation string for easier parsing
    operation = operation.replace(' ', '')
    
    try:
        # Parse the operation string into an Abstract Syntax Tree (AST)
        # This allows us to analyze the structure before evaluation
        tree = ast.parse(operation, mode='eval')
        
        # Define the allowed AST node types for safe mathematical operations
        allowed_node_types = (
            ast.Expression,  # Top-level expression container
            ast.BinOp,       # Binary operations (e.g., +, -, *, /)
            ast.UnaryOp,     # Unary operations (e.g., -x)
            ast.Num,         # Numeric literals (Python 2/3 compatibility)
            ast.Constant,    # Constant values (Python 3.8+)
            ast.Add,         # Addition operator
            ast.Sub,         # Subtraction operator
            ast.Mult,        # Multiplication operator
            ast.Div,         # Division operator
            ast.USub,        # Unary subtraction (negative)
            ast.UAdd         # Unary addition (positive)
        )
        
        # Walk through all nodes in the AST to ensure only allowed operations are present
        for node in ast.walk(tree):
            if not isinstance(node, allowed_node_types):
                raise ValueError("Invalid operation structure")
        
        # Compile the validated AST into bytecode
        compiled = compile(tree, '<string>', 'eval')
        
        # Evaluate the compiled expression and return the result
        result = eval(compiled)
        
        return result
        
    except (SyntaxError, TypeError, ZeroDivisionError) as e:
        # Handle specific known exceptions with descriptive error messages
        raise ValueError(f"Error evaluating operation: {str(e)}")
    except Exception as e:
        # Catch any unexpected exceptions to maintain safety
        raise ValueError(f"Unexpected error: {str(e)}")
