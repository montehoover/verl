import ast
import operator
from typing import Any, Dict, Callable, List


_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_CMP_OPS: Dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_ALLOWED_NAMES: Dict[str, Any] = {
    "True": True,
    "False": False,
    "None": None,
}


def evaluate_expression(expression: str) -> Any:
    """
    Safely evaluate a simple Python expression and return its result.

    Supported constructs include:
      - Literals: numbers, strings, booleans, None
      - Arithmetic/bitwise operators: +, -, *, /, //, %, **, |, ^, &, <<, >>
      - Unary operators: +x, -x, ~x, not x
      - Boolean operations: and, or (with short-circuit behavior)
      - Comparisons (including chained): ==, !=, <, <=, >, >=, is, is not, in, not in
      - Parentheses
      - Collections: tuples (...), lists [...], sets {...}, dicts {k: v}
      - Conditional expression: a if cond else b
      - Names: True, False, None

    Disallowed constructs:
      - Attribute access, subscripting, function calls, lambdas, comprehensions, f-strings, etc.

    Raises:
      ValueError: if the expression contains unsupported or unsafe constructs, or cannot be parsed.
    """
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from None

    def _eval(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        # Constants and names
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[n.id]
            raise ValueError(f"Use of name '{n.id}' is not allowed")

        # Unary operations
        if isinstance(n, ast.UnaryOp):
            op = _UNARY_OPS.get(type(n.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {ast.dump(n.op, annotate_fields=False)}")
            return op(_eval(n.operand))

        # Binary operations
        if isinstance(n, ast.BinOp):
            op = _BIN_OPS.get(type(n.op))
            if op is None:
                raise ValueError(f"Unsupported binary operator: {ast.dump(n.op, annotate_fields=False)}")
            left = _eval(n.left)
            right = _eval(n.right)
            return op(left, right)

        # Boolean operations with short-circuit semantics
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                # Evaluate left-to-right, short-circuit on falsy
                result = _eval(n.values[0])
                for v in n.values[1:]:
                    if not result:
                        return result
                    result = _eval(v)
                return result
            elif isinstance(n.op, ast.Or):
                # Evaluate left-to-right, short-circuit on truthy
                result = _eval(n.values[0])
                for v in n.values[1:]:
                    if result:
                        return result
                    result = _eval(v)
                return result
            else:
                raise ValueError(f"Unsupported boolean operator: {ast.dump(n.op, annotate_fields=False)}")

        # Comparisons (support chaining)
        if isinstance(n, ast.Compare):
            left = _eval(n.left)
            for op_node, comparator in zip(n.ops, n.comparators):
                right = _eval(comparator)
                op = _CMP_OPS.get(type(op_node))
                if op is None:
                    raise ValueError(f"Unsupported comparison operator: {ast.dump(op_node, annotate_fields=False)}")
                if not op(left, right):
                    return False
                left = right
            return True

        # Conditional expression: a if cond else b
        if isinstance(n, ast.IfExp):
            return _eval(n.body) if _eval(n.test) else _eval(n.orelse)

        # Collections
        if isinstance(n, ast.Tuple):
            return tuple(_eval(elt) for elt in n.elts)
        if isinstance(n, ast.List):
            return [_eval(elt) for elt in n.elts]
        if isinstance(n, ast.Set):
            return {_eval(elt) for elt in n.elts}
        if isinstance(n, ast.Dict):
            keys = [None if k is None else _eval(k) for k in n.keys]
            vals = [_eval(v) for v in n.values]
            return {k: v for k, v in zip(keys, vals)}

        # Explicitly reject known-dangerous or unsupported nodes
        disallowed: List[type] = [
            ast.Attribute,
            ast.Subscript,
            ast.Call,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.FormattedValue,
            ast.JoinedStr,  # f-strings
            ast.NamedExpr,  # walrus
            ast.Assign,
            ast.AugAssign,
            ast.AnnAssign,
            ast.Delete,
            ast.For,
            ast.While,
            ast.If,
            ast.With,
            ast.Raise,
            ast.Try,
            ast.Assert,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Pass,
            ast.Break,
            ast.Continue,
            ast.Match,
        ]
        if any(isinstance(n, t) for t in disallowed):
            raise ValueError(f"Unsupported or unsafe expression: {ast.dump(n, annotate_fields=False)}")

        # Fallback
        raise ValueError(f"Unsupported expression node: {ast.dump(n, annotate_fields=False)}")

    return _eval(node)


__all__ = ["evaluate_expression"]
