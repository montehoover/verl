import re
import ast
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('expression_calculator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_and_calculate(expression: str):
    """Parse and calculate a mathematical expression safely.
    
    This function takes a string containing a mathematical expression,
    validates it for safety, and returns the calculated result.
    
    Args:
        expression: A string containing the arithmetic expression to evaluate.
                   Supports basic operators (+, -, *, /), parentheses, and decimals.
    
    Returns:
        The numeric result of the calculated expression.
    
    Raises:
        ValueError: If the expression is empty, contains unsupported characters,
                   potentially malicious commands, or cannot be computed.
    """
    # Log the incoming expression
    logger.info(f"Processing expression: {expression}")
    
    # Input validation
    cleaned_expression = expression.strip()
    
    if not cleaned_expression:
        logger.error("Empty expression provided")
        raise ValueError("Empty expression")
    
    # Regex pattern explanation:
    # ^ - start of string
    # [0-9+\-*/().\s]+ - one or more of: digits, operators (+,-,*,/), parentheses, decimal points, whitespace
    # $ - end of string
    allowed_characters_pattern = r'^[0-9+\-*/().\s]+$'
    if not re.match(allowed_characters_pattern, cleaned_expression):
        logger.error(f"Expression contains unsupported characters: {cleaned_expression}")
        raise ValueError("Expression contains unsupported characters")
    
    # Security checks for potentially malicious patterns
    suspicious_keywords = ['__', 'import', 'eval', 'exec']
    for keyword in suspicious_keywords:
        if keyword in cleaned_expression:
            logger.warning(f"Potentially malicious keyword '{keyword}' detected in expression: {cleaned_expression}")
            raise ValueError("Expression contains potentially malicious commands")
    
    try:
        # Parse and validate AST structure
        syntax_tree = ast.parse(cleaned_expression, mode='eval')
        logger.debug(f"Successfully parsed expression into AST")
        
        # Extract operations from the AST for logging
        operations = []
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.BinOp):
                op_name = type(node.op).__name__
                operations.append(op_name)
            elif isinstance(node, ast.UnaryOp):
                op_name = type(node.op).__name__
                operations.append(f"Unary{op_name}")
        
        if operations:
            logger.info(f"Operations detected: {', '.join(operations)}")
        
        # Ensure only arithmetic operations are present in the AST
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Call):
                logger.error("Function call detected in expression")
                raise ValueError("Function calls are not allowed")
            elif isinstance(node, ast.Attribute):
                logger.error("Attribute access detected in expression")
                raise ValueError("Attribute access is not allowed")
            elif isinstance(node, ast.Name):
                logger.error("Variable name detected in expression")
                raise ValueError("Variable names are not allowed")
        
        # Compile and execute the validated expression
        compiled_code = compile(syntax_tree, '<string>', 'eval')
        result = eval(compiled_code)
        
        # Validate the result for mathematical errors
        if isinstance(result, float):
            if result == float('inf') or result == float('-inf'):
                logger.error(f"Result is infinity for expression: {cleaned_expression}")
                raise ValueError("Cannot compute: division by zero or invalid operation")
            if result != result:  # Check for NaN
                logger.error(f"Result is NaN for expression: {cleaned_expression}")
                raise ValueError("Cannot compute: division by zero or invalid operation")
        
        # Log successful calculation
        logger.info(f"Successfully calculated: {cleaned_expression} = {result}")
        return result
        
    except SyntaxError as e:
        logger.error(f"Syntax error in expression '{cleaned_expression}': {str(e)}")
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError as e:
        logger.error(f"Division by zero in expression '{cleaned_expression}': {str(e)}")
        raise ValueError("Cannot compute: division by zero")
    except Exception as e:
        logger.error(f"Error computing expression '{cleaned_expression}': {str(e)}")
        raise ValueError("Cannot compute the expression")
