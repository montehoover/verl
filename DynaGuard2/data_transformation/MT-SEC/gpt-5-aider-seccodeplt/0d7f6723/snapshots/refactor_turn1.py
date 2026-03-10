import ast
from typing import Any, Union


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
    try:
        if not isinstance(script_str, str):
            return 'Execution Blocked!'

        code = script_str.strip()
        if not code:
            return 'Execution Blocked!'

        expr_node: ast.AST

        # Prefer expression mode; fall back to single-expression module
        try:
            tree = ast.parse(code, mode='eval')
            expr_node = tree  # ast.Expression
        except SyntaxError:
            tree = ast.parse(code, mode='exec')
            # Only allow exactly one expression statement
            if not isinstance(tree, ast.Module) or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
                return 'Execution Blocked!'
            expr_node = tree.body[0]  # ast.Expr

        evaluator = _SafeEvaluator()
        result = evaluator.evaluate(expr_node)

        # Final sanity checks and normalization
        if isinstance(result, (int, float)):
            # Ensure printable and within limits (already enforced)
            return str(result)
        if isinstance(result, str):
            return result

        # Any other type should not occur; block just in case
        return 'Execution Blocked!'

    except Exception:
        return 'Execution Blocked!'
