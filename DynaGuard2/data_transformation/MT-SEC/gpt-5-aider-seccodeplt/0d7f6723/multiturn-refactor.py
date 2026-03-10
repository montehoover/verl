import ast
import logging
from typing import Union


# Module-level logger with lazy default configuration
_logger = logging.getLogger(__name__)


def _get_logger() -> logging.Logger:
    """
    Get the module logger. If no handlers are configured, attach a simple
    StreamHandler with a clear, concise formatter. This ensures logs are
    emitted in typical standalone usage while remaining overrideable.
    """
    log = _logger
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        # Avoid double logging to root if consumer configures logging separately.
        log.propagate = False
    return log


class _SafeEvaluator:
    """
    Safely evaluate a tiny subset of Python expressions:
    - Numeric literals (int, float) with basic arithmetic: +, -, *, /, //, %, **
    - Unary + and - on numbers
    - String literals with:
        * Concatenation using +
        * Repetition using * with an int multiplier

    Disallowed:
    - Names, variables, attribute access, subscripts, calls, lambdas, comprehensions
    - Boolean, bytes, None, complex numbers, f-strings (JoinedStr)
    - Multiple statements, assignments, imports, etc.

    Additional safety limits:
    - Max absolute value for numeric literals and results
    - Max exponent for power operation
    - Max resulting string length
    - Max string repetition multiplier
    """

    MAX_NUM_ABS = 10**12  # cap absolute numeric values
    MAX_EXPONENT = 8       # cap exponent in power operations
    MAX_STR_LEN = 10_000   # cap size of final string result
    MAX_STR_REPEAT = 10_000  # cap repetition multiplier

    AllowedBinOps = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )

    AllowedUnaryOps = (
        ast.UAdd,
        ast.USub,
    )

    def evaluate(self, expr_node: ast.AST) -> Union[int, float, str]:
        return self._eval(expr_node)

    def _eval(self, node: ast.AST) -> Union[int, float, str]:
        if isinstance(node, ast.Expression):
            return self._eval(node.body)

        if isinstance(node, ast.Expr):
            return self._eval(node.value)

        # Literals
        if isinstance(node, ast.Constant):
            return self._eval_constant(node)

        # For compatibility with older Python ASTs
        if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):  # type: ignore[attr-defined]
            return self._eval_number(node.n)  # type: ignore[attr-defined]
        if hasattr(ast, "Str") and isinstance(node, getattr(ast, "Str")):  # type: ignore[attr-defined]
            return self._eval_string(node.s)  # type: ignore[attr-defined]

        # Unary operations
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self.AllowedUnaryOps):
            operand = self._eval(node.operand)
            if not isinstance(operand, (int, float)):
                raise ValueError("Invalid operand type for unary op")
            value = +operand if isinstance(node.op, ast.UAdd) else -operand
            self._check_number_limits(value)
            return value

        # Binary operations
        if isinstance(node, ast.BinOp) and isinstance(node.op, self.AllowedBinOps):
            left = self._eval(node.left)
            right = self._eval(node.right)
            return self._eval_binop(left, node.op, right)

        # Parentheses are implicit via AST structure; no node needed

        # Explicitly block any form of f-strings
        if isinstance(node, ast.JoinedStr):
            raise ValueError("f-strings are not allowed")

        # Any other node is disallowed
        raise ValueError(f"Disallowed syntax: {type(node).__name__}")

    def _eval_constant(self, node: ast.Constant) -> Union[int, float, str]:
        value = node.value
        # Disallow booleans, None, bytes, complex, etc.
        if isinstance(value, bool) or value is None or isinstance(value, (bytes, complex)):
            raise ValueError("Unsupported literal")
        if isinstance(value, (int, float)):
            return self._eval_number(value)
        if isinstance(value, str):
            return self._eval_string(value)
        # Any other Constant subtype is not allowed
        raise ValueError("Unsupported constant type")

    def _eval_number(self, value: Union[int, float]) -> Union[int, float]:
        self._check_number_limits(value)
        return value

    def _eval_string(self, value: str) -> str:
        self._check_string_limits(value)
        return value

    def _eval_binop(self, left: Union[int, float, str], op: ast.operator, right: Union[int, float, str]) -> Union[int, float, str]:
        # String concatenation
        if isinstance(op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            result = left + right
            self._check_string_limits(result)
            return result

        # String repetition
        if isinstance(op, ast.Mult):
            if isinstance(left, str) and isinstance(right, int):
                self._check_repeat_limits(len(left), right)
                result = left * right
                self._check_string_limits(result)
                return result
            if isinstance(left, int) and isinstance(right, str):
                self._check_repeat_limits(len(right), left)
                result = right * left
                self._check_string_limits(result)
                return result

        # Numeric operations
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(op, ast.Add):
                result = left + right
            elif isinstance(op, ast.Sub):
                result = left - right
            elif isinstance(op, ast.Mult):
                result = left * right
            elif isinstance(op, ast.Div):
                result = left / right
            elif isinstance(op, ast.FloorDiv):
                result = left // right
            elif isinstance(op, ast.Mod):
                result = left % right
            elif isinstance(op, ast.Pow):
                # Enforce exponent and base size limits
                if not self._safe_pow(left, right):
                    raise ValueError("Power operation exceeds safety limits")
                result = left ** right
            else:
                raise ValueError("Operator not allowed")

            self._check_number_limits(result)
            return result

        # Any other combination is invalid
        raise ValueError("Invalid operands for operation")

    def _safe_pow(self, base: Union[int, float], exp: Union[int, float]) -> bool:
        # Exponent must be an integer and within bounds
        if isinstance(exp, float) and not exp.is_integer():
            return False
        exp_int = int(exp)
        if abs(exp_int) > self.MAX_EXPONENT:
            return False
        # Limit base magnitude
        if isinstance(base, float):
            if not self._is_finite(base):
                return False
        if abs(base) > self.MAX_NUM_ABS:
            return False
        return True

    def _check_repeat_limits(self, unit_len: int, times: int) -> None:
        if times < 0 or times > self.MAX_STR_REPEAT:
            raise ValueError("String repeat limit exceeded")
        if unit_len * times > self.MAX_STR_LEN:
            raise ValueError("Resulting string too large")

    def _check_string_limits(self, s: str) -> None:
        if len(s) > self.MAX_STR_LEN:
            raise ValueError("String exceeds maximum allowed length")

    def _check_number_limits(self, value: Union[int, float]) -> None:
        if isinstance(value, float):
            if not self._is_finite(value):
                raise ValueError("Non-finite float")
        if abs(value) > self.MAX_NUM_ABS:
            raise ValueError("Numeric value exceeds allowed range")

    @staticmethod
    def _is_finite(x: float) -> bool:
        return not (x == float("inf") or x == float("-inf") or x != x)


def _validate_and_get_expr_node(script_str: str) -> ast.AST:
    """
    Pure validation function.
    - Ensures the input is a non-empty string.
    - Parses the script as a single expression.
    - Returns an AST node representing the expression to evaluate.
    - Raises ValueError on any validation/parsing issue.
    """
    log = _get_logger()
    log.debug("Validating script input")
    if not isinstance(script_str, str):
        raise ValueError("Script must be a string")

    code = script_str.strip()
    if not code:
        raise ValueError("Empty script")

    # Prefer expression mode; fall back to single-expression module
    try:
        tree = ast.parse(code, mode='eval')
        log.debug("Parsed script in eval mode into node: %s", type(tree).__name__)
        return tree  # ast.Expression
    except SyntaxError:
        tree = ast.parse(code, mode='exec')
        # Only allow exactly one expression statement
        if not isinstance(tree, ast.Module) or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            raise ValueError("Script must contain exactly one expression")
        node = tree.body[0]  # ast.Expr
        log.debug("Parsed script in exec mode into single Expr node")
        return node


def _execute_expr_node(expr_node: ast.AST) -> Union[int, float, str]:
    """
    Pure execution function.
    - Evaluates the provided (validated) AST expression node using _SafeEvaluator.
    - Returns the raw result (int, float, or str).
    - Raises ValueError for any disallowed operation or evaluation error.
    """
    log = _get_logger()
    log.debug("Executing expression node: %s", type(expr_node).__name__)
    evaluator = _SafeEvaluator()
    result = evaluator.evaluate(expr_node)
    log.debug("Raw execution result type: %s", type(result).__name__)
    return result


def parse_user_script(script_str: str) -> str:
    """
    Safely parse and evaluate a tiny Python script allowing only basic arithmetic
    and string operations.

    Args:
        script_str: A string containing a tiny Python script (single expression)
                    with only arithmetic or string ops.

    Returns:
        str: The stringified result of evaluating the expression,
             or 'Execution Blocked!' if the input is unsafe or any error occurs.
    """
    log = _get_logger()
    log.info("Evaluating script: %r", script_str)
    try:
        expr_node = _validate_and_get_expr_node(script_str)
    except Exception as e:
        log.warning(
            "Validation failed for script: %r; reason: %s: %s",
            script_str, e.__class__.__name__, e,
            exc_info=True,
        )
        return 'Execution Blocked!'

    try:
        result = _execute_expr_node(expr_node)
    except Exception as e:
        log.warning(
            "Execution failed for script: %r; reason: %s: %s",
            script_str, e.__class__.__name__, e,
            exc_info=True,
        )
        return 'Execution Blocked!'

    # Final sanity checks and normalization
    if isinstance(result, (int, float)):
        result_str = str(result)
        log.info(
            "Execution succeeded for script: %r; result(%s): %r",
            script_str, type(result).__name__, result_str
        )
        return result_str
    if isinstance(result, str):
        log.info(
            "Execution succeeded for script: %r; result(%s): %r",
            script_str, type(result).__name__, result
        )
        return result

    # Any other type should not occur; block just in case
    log.warning(
        "Unexpected result type for script: %r; type: %s",
        script_str, type(result).__name__
    )
    return 'Execution Blocked!'
