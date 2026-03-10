import ast
from typing import List


def parse_script_operations(script: str) -> List[str]:
    """
    Parse a Python script and return a list of operations it contains, limited to
    basic arithmetic and string manipulations.

    The returned list contains unique operation names in the order they are first encountered.

    Arithmetic operations detected:
    - add, subtract, multiply, divide, floor_divide, modulo, power
    - unary_plus, unary_minus

    String operations detected:
    - string_concat (e.g., "a" + "b")
    - string_repeat (e.g., "a" * 3)
    - string_slice (e.g., "abc"[1:2], "abc"[0])
    - string_format (e.g., f"{}", "{}".format(), "%s" % val)
    - string_join, string_split, string_replace, string_strip, string_lstrip, string_rstrip
    - string_lower, string_upper, string_title, string_capitalize, string_casefold, string_swapcase
    - string_startswith, string_endswith, string_find, string_index, string_count
    - string_center, string_ljust, string_rjust, string_zfill, string_partition, string_translate
    - string_splitlines

    Note:
    - Type inference is heuristic. When it's ambiguous whether an operation is on strings,
      the operation is categorized under arithmetic (e.g., add, multiply) rather than string-specific.
    """
    if not isinstance(script, str):
        raise TypeError("script must be a string")

    tree = ast.parse(script)
    collector = _OperationCollector()
    collector.visit(tree)
    return collector.operations


class _OperationCollector(ast.NodeVisitor):
    STRING_METHODS_MAP = {
        "lower": "string_lower",
        "upper": "string_upper",
        "title": "string_title",
        "capitalize": "string_capitalize",
        "strip": "string_strip",
        "lstrip": "string_lstrip",
        "rstrip": "string_rstrip",
        "split": "string_split",
        "rsplit": "string_split",
        "splitlines": "string_splitlines",
        "join": "string_join",
        "replace": "string_replace",
        "startswith": "string_startswith",
        "endswith": "string_endswith",
        "find": "string_find",
        "rfind": "string_find",
        "index": "string_index",
        "rindex": "string_index",
        "count": "string_count",
        "format": "string_format",
        "casefold": "string_casefold",
        "center": "string_center",
        "ljust": "string_ljust",
        "rjust": "string_rjust",
        "zfill": "string_zfill",
        "partition": "string_partition",
        "rpartition": "string_partition",
        "translate": "string_translate",
        "swapcase": "string_swapcase",
    }

    def __init__(self) -> None:
        self.operations: List[str] = []
        self._seen = set()

    def _add(self, op: str) -> None:
        if op not in self._seen:
            self._seen.add(op)
            self.operations.append(op)

    def _is_stringy(self, node: ast.AST) -> bool:
        # Heuristic detection of a string-producing node
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return True
        if isinstance(node, ast.JoinedStr):
            return True  # f-strings
        return False

    # Handle binary operators
    def visit_BinOp(self, node: ast.BinOp) -> None:
        op = node.op
        left, right = node.left, node.right

        if isinstance(op, ast.Add):
            if self._is_stringy(left) or self._is_stringy(right):
                self._add("string_concat")
            else:
                self._add("add")
        elif isinstance(op, ast.Sub):
            self._add("subtract")
        elif isinstance(op, ast.Mult):
            if self._is_stringy(left) or self._is_stringy(right):
                self._add("string_repeat")
            else:
                self._add("multiply")
        elif isinstance(op, ast.Div):
            self._add("divide")
        elif isinstance(op, ast.FloorDiv):
            self._add("floor_divide")
        elif isinstance(op, ast.Mod):
            # Percent string formatting if left side is string-like
            if self._is_stringy(left):
                self._add("string_format")
            else:
                self._add("modulo")
        elif isinstance(op, ast.Pow):
            self._add("power")
        # Other binary operators (bitwise, matmul) are ignored

        self.generic_visit(node)

    # Handle augmented assignment (e.g., +=, *=)
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        op = node.op
        target, value = node.target, node.value

        if isinstance(op, ast.Add):
            if self._is_stringy(target) or self._is_stringy(value):
                self._add("string_concat")
            else:
                self._add("add")
        elif isinstance(op, ast.Sub):
            self._add("subtract")
        elif isinstance(op, ast.Mult):
            if self._is_stringy(target) or self._is_stringy(value):
                self._add("string_repeat")
            else:
                self._add("multiply")
        elif isinstance(op, ast.Div):
            self._add("divide")
        elif isinstance(op, ast.FloorDiv):
            self._add("floor_divide")
        elif isinstance(op, ast.Mod):
            if self._is_stringy(target):
                self._add("string_format")
            else:
                self._add("modulo")
        elif isinstance(op, ast.Pow):
            self._add("power")

        self.generic_visit(node)

    # Handle unary operators
    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.UAdd):
            self._add("unary_plus")
        elif isinstance(node.op, ast.USub):
            self._add("unary_minus")
        self.generic_visit(node)

    # f-strings
    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._add("string_format")
        self.generic_visit(node)

    # String-related calls via attribute methods
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            op_name = self.STRING_METHODS_MAP.get(func.attr)
            if op_name:
                self._add(op_name)
        elif isinstance(func, ast.Name):
            # format(value, spec) builtin can be used for strings as well
            if func.id == "format":
                self._add("string_format")
        self.generic_visit(node)

    # Slicing / indexing
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self._is_stringy(node.value):
            self._add("string_slice")
        else:
            # Keep generic slice for non-string where type is unknown
            self._add("slice")
        self.generic_visit(node)


def evaluate_operations(operations: List[str]):
    """
    Evaluate a list of operation names previously identified by parse_script_operations.

    Safe operations executed:
    - Arithmetic: add, subtract, multiply, divide, floor_divide, modulo, power, unary_plus, unary_minus
    - String: string_concat

    Rules:
    - If an operation outside the safe set is encountered, return a safety violation message.
    - Mixing numeric operations with string operations is not allowed and results in a safety violation.
    - The evaluation uses a deterministic accumulator:
        * Numeric accumulator starts at 0 (int).
          add:+1, subtract:-1, multiply:*2, divide:/2, floor_divide://2, modulo:%2, power:**2,
          unary_plus:+acc, unary_minus:-acc
        * String accumulator starts as "" and string_concat appends "x".

    Returns:
        - A numeric or string result if all operations are safe and type-consistent.
        - A string message starting with "Safety violation:" if an unsafe operation or invalid mix is detected.
    """
    if not isinstance(operations, list) or not all(isinstance(o, str) for o in operations):
        return "Safety violation: invalid operations input"

    safe_arith = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "modulo",
        "power",
        "unary_plus",
        "unary_minus",
    }
    safe_string = {"string_concat"}

    value = None  # type: ignore[assignment]
    mode = None  # "number" or "string"

    for op in operations:
        if op in safe_arith:
            if mode is None:
                mode = "number"
                value = 0
            elif mode != "number":
                return "Safety violation: mixed string and numeric operations"
            if op == "add":
                value += 1
            elif op == "subtract":
                value -= 1
            elif op == "multiply":
                value *= 2
            elif op == "divide":
                value /= 2
            elif op == "floor_divide":
                value //= 2
            elif op == "modulo":
                value %= 2
            elif op == "power":
                value **= 2
            elif op == "unary_plus":
                value = +value
            elif op == "unary_minus":
                value = -value
        elif op in safe_string:
            if mode is None:
                mode = "string"
                value = ""
            elif mode != "string":
                return "Safety violation: mixed string and numeric operations"
            if op == "string_concat":
                value += "x"
        else:
            return f"Safety violation: unsafe operation '{op}'"

    if value is None:
        # No operations: return neutral numeric result
        return 0
    return value


def run_user_code(python_code: str):
    """
    Securely evaluate user-supplied Python code by parsing for allowed operations
    and computing a deterministic result. If unsafe operations are detected or parsing
    fails, return 'Execution Blocked!'.
    """
    try:
        operations = parse_script_operations(python_code)
    except Exception:
        return "Execution Blocked!"
    result = evaluate_operations(operations)
    if isinstance(result, str) and result.startswith("Safety violation:"):
        return "Execution Blocked!"
    return result


__all__ = ["parse_script_operations", "evaluate_operations", "run_user_code"]
