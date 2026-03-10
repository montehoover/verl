"""
Utilities for safely evaluating limited mathematical expressions provided by
users. The evaluation is performed by parsing the input into a Python AST and
then interpreting only a strict subset of node types and operators.

This approach avoids using eval while still supporting basic arithmetic.

Logging
-------
This module emits human-readable log messages that trace:
  - The raw operation input
  - Validation and parsing steps
  - Each arithmetic step during evaluation
  - The final result (or errors if encountered)

By default, a NullHandler is attached to the module logger to avoid
'No handler found' warnings in library contexts. Applications can configure
logging as desired, for example:

    import logging
    logging.basicConfig(level=logging.INFO)

"""

import ast
import logging
from typing import Union


# Module-level logger for this utility.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def safe_execute_operation(operation: str) -> Union[int, float]:
    """
    Safely evaluate a simple mathematical expression provided as a string.

    Supported constructs:
      - Numeric literals: integers and floats
      - Binary operators: +, -, *, /, //, %, **
      - Unary operators: +, -
      - Parentheses (implicitly handled by AST structure)

    Disallowed constructs:
      - Names/variables, attribute access, function calls
      - Subscripts (e.g., a[0]), slicing
      - Collections (lists, tuples, dicts, sets) and comprehensions
      - Boolean operations, comparisons, conditionals, lambdas, f-strings, etc.

    Logging
    -------
    INFO-level logs provide a readable trace of:
      - Input operation
      - Validation success
      - Each arithmetic step with intermediate results
      - The final result

    DEBUG-level logs include extra details like AST dumps.

    Parameters
    ----------
    operation : str
        The mathematical expression to evaluate.

    Returns
    -------
    int | float
        The numeric result of the evaluated expression.

    Raises
    ------
    ValueError
        If the input is not a string, is empty, contains disallowed syntax,
        or if an error occurs during evaluation (e.g., division by zero).
    """
    # Basic input validation.
    if not isinstance(operation, str):
        logger.error("Operation type invalid: %r", type(operation))
        raise ValueError("Operation must be a string.")

    logger.info("Evaluating operation: %r", operation)
    expr = operation.strip()
    logger.debug("Normalized expression: %s", expr)

    if not expr:
        logger.error("Empty operation provided.")
        raise ValueError("Empty operation.")

    # Parse the string into an AST in expression mode.
    try:
        tree = ast.parse(expr, mode="eval")
        logger.debug("AST parsed successfully.")
        # AST dump can be verbose; keep at DEBUG level.
        logger.debug("AST dump: %s", ast.dump(tree))
    except SyntaxError as exc:
        # Normalize to ValueError for the API contract.
        logger.error("Invalid expression during parsing: %s", exc)
        raise ValueError(f"Invalid expression: {exc}") from None

    # Whitelists of operator node types we allow.
    allowed_binops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    allowed_unaryops = (ast.UAdd, ast.USub)

    def _is_allowed_number_node(node: ast.AST) -> bool:
        """
        Check whether a node represents a numeric literal we allow.

        Supports:
          - ast.Constant with int/float
          - ast.Num (for older Python versions)
        """
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return isinstance(node.n, (int, float))
        return False

    def _op_symbol(op: ast.AST) -> str:
        """
        Return a human-friendly symbol for a given operator node.
        """
        mapping = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
        }
        for cls, symbol in mapping.items():
            if isinstance(op, cls):
                return symbol
        return "?"

    def _validate(node: ast.AST) -> None:
        """
        Recursively validate that the AST contains only allowed constructs.

        Raises ValueError if any disallowed or unknown node type is found.
        """
        if isinstance(node, ast.Expression):
            _validate(node.body)

        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_binops):
                raise ValueError("Disallowed binary operator.")
            _validate(node.left)
            _validate(node.right)

        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unaryops):
                raise ValueError("Disallowed unary operator.")
            _validate(node.operand)

        elif _is_allowed_number_node(node):
            # Numeric literals are allowed.
            return

        elif isinstance(node, ast.Expr):
            # Wrapper node sometimes appears; validate its value.
            _validate(node.value)

        # Explicitly disallow common dangerous/irrelevant nodes.
        elif isinstance(
            node,
            (
                ast.Call,
                ast.Name,
                ast.Attribute,
                ast.Subscript,
                ast.List,
                ast.Tuple,
                ast.Dict,
                ast.Set,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp if hasattr(ast, "DictComp") else tuple(),
                ast.GeneratorExp,
                ast.Compare,
                ast.BoolOp,
                ast.Lambda,
                ast.IfExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.FormattedValue,
                ast.JoinedStr,
                ast.Bytes,
                ast.Slice,
                ast.ExtSlice,
                ast.Index if hasattr(ast, "Index") else tuple(),
            ),
        ):
            raise ValueError("Disallowed expression element.")

        else:
            # Any node type not explicitly allowed is rejected.
            raise ValueError(
                f"Unsupported expression element: {type(node).__name__}"
            )

    def _eval(node: ast.AST) -> Union[int, float]:
        """
        Interpret the validated AST to compute the numeric result.

        This function assumes the AST has already been validated by _validate.
        """
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if _is_allowed_number_node(node):
            # Return the numeric value for constants/nums.
            value = node.value if isinstance(node, ast.Constant) else node.n
            logger.info("Number literal: %s", value)
            return value

        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                result = +operand
                logger.info("Unary + applied to %s -> %s", operand, result)
                return result
            if isinstance(node.op, ast.USub):
                result = -operand
                logger.info("Unary - applied to %s -> %s", operand, result)
                return result
            raise ValueError("Disallowed unary operator during evaluation.")

        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op = node.op
            symbol = _op_symbol(op)

            try:
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
                    result = left ** right
                else:
                    raise ValueError("Disallowed binary operator during evaluation.")
            except Exception as exc:
                logger.error(
                    "Error during evaluation step: %s %s %s -> %s",
                    left,
                    symbol,
                    right,
                    exc,
                )
                # Normalize runtime errors to ValueError to meet the contract.
                raise ValueError(f"Error during evaluation: {exc}") from None

            logger.info("Step result: %s %s %s = %s", left, symbol, right, result)
            return result

        # Should never be reached if validation worked correctly.
        raise ValueError(
            f"Unsupported node during evaluation: {type(node).__name__}"
        )

    # Validate, then evaluate.
    try:
        _validate(tree)
        logger.info("Expression validated as safe.")
    except ValueError as exc:
        logger.error("Validation failed for expression %r: %s", expr, exc)
        raise

    try:
        result = _eval(tree)
        logger.info("Final result: %s", result)
        return result
    except ValueError:
        # Re-raise ValueError unchanged (already normalized).
        logger.error("Evaluation error for %r.", expr)
        raise
    except Exception as exc:
        # Any unexpected error becomes a ValueError for the API contract.
        logger.error("Unexpected error during evaluation of %r: %s", expr, exc)
        raise ValueError(f"Evaluation failed: {exc}") from None
