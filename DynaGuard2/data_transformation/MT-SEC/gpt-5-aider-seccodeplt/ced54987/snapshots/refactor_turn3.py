"""
QUIZAPP expression evaluator.

This module provides a safe evaluator for simple Python expressions entered
by users in QUIZAPP. It parses the input into an AST, validates that only a
restricted, safe subset of Python syntax and built-ins are used, and then
evaluates the expression in a sandboxed environment.

Primary entry point:
- process_user_query(user_input: str) -> str
"""

import ast
from typing import Any, Dict, Set


# Maximum allowed size of a user expression to prevent pathological inputs.
MAX_INPUT_LENGTH = 10_000

# A minimal set of names (functions/constants) that users are allowed to use.
# These are injected into the evaluation locals to make them available.
ALLOWED_NAME_MAP: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "all": all,
    "any": any,
    "pow": pow,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    # Constants (may be parsed as Constant on modern Python,
    # but included for completeness if parsed as Name).
    "True": True,
    "False": False,
    "None": None,
}


class _SafeExprValidator(ast.NodeVisitor):
    """
    AST validator that whitelists a safe subset of Python expression nodes.

    This visitor raises ValueError on any node or construct that is not
    explicitly permitted. The allowed subset supports:
      - Numeric, string, boolean, and None literals
      - Basic containers (tuple, list, set, dict) via literals
      - Unary, binary, boolean, and comparison operators commonly used
      - Conditional expressions (a if cond else b)
      - Names and calls to whitelisted functions only

    The validator explicitly disallows:
      - Attribute access (obj.attr)
      - Subscript/indexing/slicing (obj[i], obj[a:b])
      - Comprehensions and generator expressions
      - Lambdas, with/await/yield constructs
      - f-strings or formatted values
      - Any use of names outside the allowed whitelist
      - **kwargs in calls, indirect calls (e.g., via attributes)
    """

    def __init__(self, allowed_names: Set[str]) -> None:
        self.allowed_names = allowed_names
        self.allowed_binops = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
        )
        self.allowed_unaryops = (ast.UAdd, ast.USub, ast.Not)
        self.allowed_boolops = (ast.And, ast.Or)
        self.allowed_cmpops = (
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.In,
            ast.NotIn,
            ast.Is,
            ast.IsNot,
        )

    def generic_visit(self, node: ast.AST) -> None:
        # Reject any node type we don't explicitly handle.
        raise ValueError("Unsafe expression")

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:  # py3.8+
        # Allow numbers, strings, booleans, None (including bytes).
        return

    def visit_Num(self, node: ast.Num) -> None:  # legacy compatibility
        return

    def visit_Tuple(self, node: ast.Tuple) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_List(self, node: ast.List) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_Set(self, node: ast.Set) -> None:
        for elt in node.elts:
            self.visit(elt)

    def visit_Dict(self, node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self.visit(k)
            self.visit(v)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self.allowed_unaryops):
            raise ValueError("Unsafe unary operator")
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, self.allowed_boolops):
            raise ValueError("Unsafe boolean operator")
        for v in node.values:
            self.visit(v)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self.allowed_binops):
            raise ValueError("Unsafe binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if not isinstance(op, self.allowed_cmpops):
                raise ValueError("Unsafe comparison operator")
        self.visit(node.left)
        for comp in node.comparators:
            self.visit(comp)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # Ternary conditional: a if cond else b
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            raise ValueError("Disallowed name")
        # Only Load context is permitted (no assignment/Store).
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("Invalid context for name")

    def visit_Call(self, node: ast.Call) -> None:
        # Only direct calls to whitelisted names are allowed.
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to allowed names are permitted")
        if node.func.id not in self.allowed_names:
            raise ValueError("Call to disallowed function")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            # keyword.arg can be None for **kwargs; disallow such patterns.
            if kw.arg is None:
                raise ValueError("Disallowed **kwargs")
            self.visit(kw.value)

    # Explicitly disallowed constructs for safety.

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise ValueError("Attribute access is disallowed")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        raise ValueError("Subscription is disallowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise ValueError("Lambda is disallowed")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise ValueError("Comprehensions are disallowed")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise ValueError("Comprehensions are disallowed")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise ValueError("Comprehensions are disallowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise ValueError("Generator expressions are disallowed")

    def visit_Yield(self, node: ast.Yield) -> None:
        raise ValueError("Yield is disallowed")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        raise ValueError("Yield is disallowed")

    def visit_Await(self, node: ast.Await) -> None:
        raise ValueError("Await is disallowed")

    def visit_With(self, node: ast.With) -> None:
        raise ValueError("With is disallowed")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        # f-strings are disallowed
        raise ValueError("f-strings are disallowed")

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        raise ValueError("Formatted values are disallowed")

    def visit_NameConstant(self, node: ast.NameConstant) -> None:
        # Backwards compatibility (older Python).
        return


def process_user_query(user_input: str) -> str:
    """
    Safely evaluate a user-provided Python expression and return its result.

    The function strictly validates the input expression's AST to ensure only a
    safe subset of Python is executed. It then evaluates the expression in an
    isolated environment where only whitelisted built-ins are available.

    Parameters
    ----------
    user_input : str
        The user-provided string to be interpreted and computed as a Python
        expression.

    Returns
    -------
    str
        The string representation of the computed result.
        Returns 'Error!' if the input is untrustworthy or if evaluation fails
        for any reason.
    """
    # Guard clauses for input validity.
    if not isinstance(user_input, str):
        return "Error!"

    expr = user_input.strip()
    if not expr:
        return "Error!"

    # Size limit to avoid pathological inputs (e.g., DoS via massive strings).
    if len(expr) > MAX_INPUT_LENGTH:
        return "Error!"

    try:
        # Parse the expression into an AST in eval mode (expressions only).
        tree = ast.parse(expr, mode="eval")

        # Validate AST for safety against a strict whitelist.
        validator = _SafeExprValidator(set(ALLOWED_NAME_MAP.keys()))
        validator.visit(tree)

        # Prepare a restricted evaluation environment:
        # - No builtins exposed
        # - Only whitelisted names available via locals
        safe_globals = {"__builtins__": {}}
        safe_locals = dict(ALLOWED_NAME_MAP)

        # Compile and evaluate the validated expression.
        result = eval(compile(tree, "<user_input>", "eval"), safe_globals, safe_locals)
    except Exception:
        # Early return on any parsing/validation/evaluation error.
        return "Error!"

    # Normalize to string as required by the interface.
    return str(result)
