import re
import ast

def parse_and_calculate(expression: str):
    # Remove whitespace
    expression = expression.strip()
    
    # Check for empty expression
    if not expression:
        raise ValueError("Empty expression")
    
    # Check for allowed characters (digits, operators, parentheses, decimal points, and whitespace)
    if not re.match(r'^[0-9+\-*/().\s]+$', expression):
        raise ValueError("Expression contains unsupported characters")
    
    # Check for suspicious patterns that might indicate malicious input
    if '__' in expression or 'import' in expression or 'eval' in expression or 'exec' in expression:
        raise ValueError("Expression contains potentially malicious commands")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Check that the AST only contains allowed operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                raise ValueError("Function calls are not allowed")
            elif isinstance(node, ast.Attribute):
                raise ValueError("Attribute access is not allowed")
            elif isinstance(node, ast.Name):
                raise ValueError("Variable names are not allowed")
        
        # Compile and evaluate the expression
        code = compile(tree, '<string>', 'eval')
        result = eval(code)
        
        # Check for division by zero or other math errors that result in inf/nan
        if isinstance(result, float) and (result == float('inf') or result == float('-inf') or result != result):
            raise ValueError("Cannot compute: division by zero or invalid operation")
        
        return result
        
    except SyntaxError:
        raise ValueError("Invalid expression syntax")
    except ZeroDivisionError:
        raise ValueError("Cannot compute: division by zero")
    except Exception:
        raise ValueError("Cannot compute the expression")
