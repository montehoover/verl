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


__all__ = ["parse_script_operations"]
