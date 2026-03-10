import re

def validate_expression(expr: str) -> bool:
    if expr is None:
        return False
    return bool(re.fullmatch(r'(?=.*\d)[\d+\-*/\s]+', expr))
