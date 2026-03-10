import re
from typing import Union, Any

# Allowed arithmetic operators
ALLOWED_ARITHMETIC_OPERATORS = ['+', '-', '*', '/', '%', '**']
# Allowed string operators (Note: '*' is for repetition, '+' for concatenation)
# For simplicity, we are not including string methods like .upper(), .split() etc.

# Regex for a number (integer or float)
REGEX_NUMBER = r"-?\d+(\.\d+)?"
# Regex for an integer (used for string repetition count)
REGEX_INTEGER = r"-?\d+"
# Regex for a string literal (single or double quotes)
# This basic regex doesn't handle escaped quotes within strings.
REGEX_STRING = r"('[^']*'|\"[^\"]*\")"

# Compiled regex patterns for valid operations:

# 1. Arithmetic operation: NUMBER OPERATOR NUMBER
#    e.g., "1 + 2", "3.14 * 4", "-5 / 2"
REGEX_ARITHMETIC_OP = re.compile(
    rf"^\s*({REGEX_NUMBER})\s*({'|'.join(re.escape(op) for op in ALLOWED_ARITHMETIC_OPERATORS)})\s*({REGEX_NUMBER})\s*$"
)

# 2. String concatenation: STRING + STRING
#    e.g., "'hello' + 'world'", "\"abc\" + \"def\""
REGEX_STRING_CONCAT_OP = re.compile(
    rf"^\s*({REGEX_STRING})\s*(\+)\s*({REGEX_STRING})\s*$"
)

# 3. String repetition: STRING * INTEGER or INTEGER * STRING
#    e.g., "'a' * 3", "3 * 'a'"
REGEX_STRING_REPETITION_OP_STR_FIRST = re.compile(
    rf"^\s*({REGEX_STRING})\s*(\*)\s*({REGEX_INTEGER})\s*$"
)
REGEX_STRING_REPETITION_OP_INT_FIRST = re.compile(
    rf"^\s*({REGEX_INTEGER})\s*(\*)\s*({REGEX_STRING})\s*$"
)

# List of all valid operation patterns to check against
VALID_OPERATION_PATTERNS = [
    REGEX_ARITHMETIC_OP,
    REGEX_STRING_CONCAT_OP,
    REGEX_STRING_REPETITION_OP_STR_FIRST,
    REGEX_STRING_REPETITION_OP_INT_FIRST,
]

def _strip_quotes(s: str) -> str:
    """Helper to remove one layer of single or double quotes."""
    if len(s) >= 2:
        if (s.startswith("'") and s.endswith("'")) or \
           (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
    return s

def parse_script_operations(script_text: str) -> list[str]:
    """
    Parses a user-supplied script for valid arithmetic or string operations.

    The function splits the script into lines and checks each line against a
    predefined set of safe operation patterns (e.g., "number operator number",
    "string + string", "string * integer"). Only lines that exactly match
    one of these patterns are considered valid operations.

    This approach avoids using `eval()` or `exec()` and relies on strict
    pattern matching to prevent the execution of potentially harmful commands.

    Args:
        script_text: The script content as a multi-line string.

    Returns:
        A list of strings, where each string is a line from the input
        script that has been identified as a valid and safe operation.
        Lines that do not match any safe pattern are ignored.
    """
    valid_operations: list[str] = []
    lines = script_text.splitlines()

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:  # Skip empty or whitespace-only lines
            continue

        for pattern in VALID_OPERATION_PATTERNS:
            if pattern.match(stripped_line):
                valid_operations.append(stripped_line)
                break  # Line matched, move to the next line
        # If no pattern matched, the line is considered invalid or not an operation
        # and is implicitly skipped.

    return valid_operations

def evaluate_operations(operations: list[str]) -> Union[list[Any], str]:
    """
    Evaluates a list of validated operation strings.

    Args:
        operations: A list of strings, where each string is a pre-validated
                    operation (e.g., "1 + 2", "'a' * 3").

    Returns:
        A list of results if all operations are successful.
        An error message string if any operation fails (e.g., division by zero,
        malformed number, or unrecognized operation format).
    """
    results: list[Any] = []
    for op_str in operations:
        # Ensure we are working with a clean, stripped string
        # parse_script_operations should already provide this.
        current_op_str = op_str.strip()

        # Attempt Arithmetic Operation
        match = REGEX_ARITHMETIC_OP.match(current_op_str)
        if match:
            val1_str, op_symbol, val2_str = match.groups()
            try:
                # Use float for general arithmetic, allows for decimal results
                val1 = float(val1_str)
                val2 = float(val2_str)
            except ValueError:
                return f"Error: Invalid number format in operation '{current_op_str}'"

            if op_symbol == '+': result = val1 + val2
            elif op_symbol == '-': result = val1 - val2
            elif op_symbol == '*': result = val1 * val2
            elif op_symbol == '/':
                if val2 == 0: return f"Error: Division by zero in operation '{current_op_str}'"
                result = val1 / val2
            elif op_symbol == '%':
                if val2 == 0: return f"Error: Modulo by zero in operation '{current_op_str}'"
                result = val1 % val2
            elif op_symbol == '**': result = val1 ** val2
            else:
                # This case should ideally not be reached if regex is comprehensive
                return f"Error: Unknown arithmetic operator '{op_symbol}' in '{current_op_str}' (safety concern)"
            results.append(result)
            continue

        # Attempt String Concatenation
        match = REGEX_STRING_CONCAT_OP.match(current_op_str)
        if match:
            str1_raw, _, str2_raw = match.groups() # Operator is known to be '+'
            str1 = _strip_quotes(str1_raw)
            str2 = _strip_quotes(str2_raw)
            results.append(str1 + str2)
            continue

        # Attempt String Repetition (string * integer)
        match = REGEX_STRING_REPETITION_OP_STR_FIRST.match(current_op_str)
        if match:
            str_raw, _, count_str = match.groups() # Operator is known to be '*'
            str_val = _strip_quotes(str_raw)
            try:
                count = int(count_str)
            except ValueError:
                return f"Error: Invalid integer for string repetition count in '{current_op_str}'"
            results.append(str_val * count)
            continue

        # Attempt String Repetition (integer * string)
        match = REGEX_STRING_REPETITION_OP_INT_FIRST.match(current_op_str)
        if match:
            count_str, _, str_raw = match.groups() # Operator is known to be '*'
            str_val = _strip_quotes(str_raw)
            try:
                count = int(count_str)
            except ValueError:
                return f"Error: Invalid integer for string repetition count in '{current_op_str}'"
            results.append(count * str_val) # Python handles int * str correctly
            continue

        # If no pattern matched the operation string
        # This indicates a discrepancy between parser and evaluator, or a manually passed invalid op
        return f"Error: Unrecognized operation format (safety concern): '{current_op_str}'"

    return results

if __name__ == '__main__':
    def run_test_case(script_name: str, script_content: str):
        print(f"\n--- Testing: {script_name} ---")
        print("Script content:")
        print(script_content.strip())

        parsed_ops = parse_script_operations(script_content)
        print("\nParsed valid operations:")
        if parsed_ops:
            for op in parsed_ops:
                print(f"- \"{op}\"")

            print("\nEvaluating operations:")
            eval_results = evaluate_operations(parsed_ops)

            if isinstance(eval_results, str):  # Error message returned
                print(f"Evaluation Error: {eval_results}")
            else:
                print("Evaluation Results:")
                for i, res in enumerate(eval_results):
                    # Show original operation alongside its result for clarity
                    original_op = parsed_ops[i]
                    print(f"- \"{original_op}\" -> {res} (type: {type(res).__name__})")
        else:
            print("(No valid operations found to parse or evaluate)")
        print("--- End Test Case ---")

    # Test Case 1: Sample script with various operations
    sample_script = """
    # This is a valid arithmetic operation
    10 + 20
    # This is a string concatenation
    'hello' + " world"
    # This is string repetition
    "abc" * 3
    3 * "abc" 
    # Invalid operations or comments (will be filtered by parser)
    # x = 10 + 5
    # print("Hello")
    # Division by zero (syntactically valid for parser, error for evaluator)
    1 / 0
    # Another operation after division by zero (will not be reached if error occurs)
    5 + 5
    # Harmful command (will be ignored by parser)
    __import__('os').system('echo harmful')
    # Malformed operations (will be ignored by parser)
    1 +
    'hello' +
    'a' + 'b' + 'c' 
    5.5 * 2.1 # Valid arithmetic
    -10 - 5   # Valid arithmetic
    "test" % 2 # Invalid for parser as "test" is not a number.
    """
    run_test_case("Sample Script", sample_script)

    # Test Case 2: Script with only invalid lines or comments
    empty_script = """
    variable = 10
    another_var = "test"
    # just comments
    """
    run_test_case("Empty/Invalid Script", empty_script)

    # Test Case 3: Script with mixed valid and invalid lines
    mixed_script = """
    100 * 2
    # comment line
    'valid' + ' string'
    invalid_operation() 
    4 / 2
    """
    run_test_case("Mixed Script", mixed_script)

    # Test Case 4: Script that should produce a specific evaluation error (division by zero)
    error_script_division_by_zero = """
    10 + 5
    20 / 0 
    30 + 3
    """
    run_test_case("Error Script (Division by Zero)", error_script_division_by_zero)

    # Test Case 5: Script with string repetition count that is not an integer
    # This should be caught by the parser (REGEX_INTEGER for count).
    # If it somehow passed parsing, evaluate_operations has a check.
    error_script_bad_repetition_count = """
    "text" * 2.5 
    """
    run_test_case("Error Script (Bad Repetition Count - Parser should catch)", error_script_bad_repetition_count)

    # Test an unrecognized operation format by manually feeding evaluate_operations
    # This simulates a scenario where an operation string bypasses the parser's validation
    # or there's a mismatch between parser regex and evaluator logic.
    print("\n--- Testing: Manual Unrecognized Operation ---")
    manual_ops_unrecognized = ["10 + 20", "this is not a valid op format"]
    print(f"Manually providing operations: {manual_ops_unrecognized}")
    eval_results_manual_unrec = evaluate_operations(manual_ops_unrecognized)
    if isinstance(eval_results_manual_unrec, str):
        print(f"Evaluation Error: {eval_results_manual_unrec}")
    else:
        print("Evaluation Results (should not be reached for this test):")
        for res in eval_results_manual_unrec:
            print(f"- {res} (type: {type(res).__name__})")
    print("--- End Test Case ---")
