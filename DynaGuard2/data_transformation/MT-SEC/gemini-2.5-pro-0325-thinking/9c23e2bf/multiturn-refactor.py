import re
import operator
import logging
from functools import partial # Included as per setup, though not actively used in this impl.

# Setup a module-level logger. Configuration will be done in run_custom_query if needed.
logger = logging.getLogger(__name__)

# Operator mapping for WHERE clause
_OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

# Main query parsing regex: SELECT <cols> [WHERE <cond>] [ORDER BY <sort_cols>]
# FROM is implicit (the dataset argument)
_QUERY_REGEX = re.compile(
    r"^SELECT\s+(?P<select_cols>.+?)"
    r"(?:\s+WHERE\s+(?P<where_cond>.+?))?"
    r"(?:\s+ORDER BY\s+(?P<orderby_cols>.+?))?$",
    re.IGNORECASE # Make keywords SELECT, WHERE, ORDER BY case-insensitive
)

def _parse_query_string(query_str):
    """
    Parses the SQL-like query string into its components.

    Args:
        query_str: The raw SQL-like query string.

    Returns:
        A dictionary with keys 'select_cols', 'where_cond', 'orderby_cols',
        containing the respective parts of the query. 'where_cond' and
        'orderby_cols' can be None if not present in the query.

    Raises:
        ValueError: If the query string is empty or has an invalid format.
    """
    query_str = query_str.strip()
    if not query_str:
        raise ValueError("Query cannot be empty.")

    match = _QUERY_REGEX.match(query_str)
    if not match:
        raise ValueError(f"Invalid query format: '{query_str}'. Expected 'SELECT ... [WHERE ...] [ORDER BY ...]'")
    
    return match.groupdict()

def _parse_value(value_str):
    """Converts a string value from a query into a Python type."""
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # String literal
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            # If not a recognized type, raise error.
            # Unquoted strings are not supported as values to avoid ambiguity with column names.
            raise ValueError(f"Unrecognized value format or unquoted string: {value_str}")

def _apply_where_clause(dataset, where_clause_str):
    """Filters the dataset based on the WHERE clause."""
    if not where_clause_str:
        return dataset

    # Simple WHERE condition parser: "column operator value"
    # Does not support AND/OR, parentheses, etc.
    match = re.fullmatch(r"\s*(\w+)\s*([=<>!]+)\s*(.*)", where_clause_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid WHERE clause format: {where_clause_str}")

    column, op_str, value_str = match.groups()

    if op_str not in _OPERATORS:
        raise ValueError(f"Unsupported operator in WHERE clause: {op_str}")
    
    op_func = _OPERATORS[op_str]
    
    try:
        value = _parse_value(value_str)
    except ValueError as e:
        # Propagate error from _parse_value, which already has a good message
        raise ValueError(f"Invalid value in WHERE clause '{value_str}': {e}")

    filtered_dataset = []
    if not dataset: # No data to filter
        return []

    # Check if column exists using the first row (assuming homogeneous dataset structure)
    if column not in dataset[0]:
        raise ValueError(f"Column '{column}' not found in dataset for WHERE clause.")

    for row in dataset:
        row_value = row[column]
        try:
            if op_func(row_value, value):
                filtered_dataset.append(row)
        except TypeError:
            # This error occurs for incompatible type comparisons, e.g., int > str
            raise ValueError(
                f"Type mismatch in WHERE clause: cannot compare '{row_value}' (type {type(row_value).__name__}) "
                f"with '{value}' (type {type(value).__name__}) for column '{column}'."
            )
            
    return filtered_dataset

def _apply_select_clause(dataset, select_clause_str):
    """Projects columns from the dataset based on the SELECT clause."""
    select_clause_str = select_clause_str.strip()
    if not select_clause_str:
        # This should ideally be caught by the main query regex
        raise ValueError("SELECT clause cannot be empty.")

    if select_clause_str == '*':
        return [row.copy() for row in dataset] # Return copies

    select_columns = [col.strip() for col in select_clause_str.split(',')]
    if not all(select_columns): # Check for empty column names like "col1,,col2"
        raise ValueError("Invalid column name in SELECT clause: found empty or malformed column name.")

    projected_dataset = []
    if not dataset and select_columns: # No data to select from, but columns specified
        return []
    
    # Validate column existence using the first row (if dataset is not empty)
    if dataset:
        sample_row = dataset[0]
        for col in select_columns:
            if col not in sample_row:
                raise ValueError(f"Column '{col}' not found in dataset for SELECT clause.")

    for row in dataset:
        new_row = {}
        for col in select_columns:
            # This check is redundant if validated above, but good for safety per row
            # if col not in row: 
            #     raise ValueError(f"Column '{col}' not found in row: {row} for SELECT clause.")
            new_row[col] = row[col]
        projected_dataset.append(new_row)
    return projected_dataset

def _apply_order_by_clause(dataset, order_by_clause_str):
    """Sorts the dataset based on the ORDER BY clause."""
    if not order_by_clause_str:
        return dataset

    order_by_terms = [term.strip() for term in order_by_clause_str.split(',')]
    
    sort_criteria = []
    for term in order_by_terms:
        parts = term.strip().split()
        if not parts: continue # Should not happen with proper splitting

        column = parts[0]
        direction = 'ASC'
        if len(parts) > 2:
            raise ValueError(f"Invalid ORDER BY term format: '{term}'. Expected 'column [ASC|DESC]'.")
        if len(parts) == 2:
            direction = parts[1].upper()
            if direction not in ['ASC', 'DESC']:
                raise ValueError(f"Invalid sort direction '{parts[1]}' in ORDER BY term: '{term}'. Use ASC or DESC.")
        sort_criteria.append({'column': column, 'reverse': direction == 'DESC'})

    if not dataset and sort_criteria: # No data to sort
        return []

    # Validate columns using the first row (if dataset is not empty)
    # These columns must exist in the dataset *after* SELECT has been applied.
    if dataset:
        sample_row = dataset[0]
        for crit in sort_criteria:
            if crit['column'] not in sample_row:
                raise ValueError(f"Column '{crit['column']}' not found in dataset for ORDER BY clause. "
                                 "Ensure columns in ORDER BY are also in SELECT if not using SELECT *.")
    
    # Multi-level sort: Python's sort is stable. Apply sorts in reverse order of criteria.
    # (i.e., sort by last key, then second to last, etc.)
    for crit in reversed(sort_criteria):
        try:
            dataset.sort(key=operator.itemgetter(crit['column']), reverse=crit['reverse'])
        except TypeError as e:
            raise ValueError(f"Cannot sort by column '{crit['column']}' due to mixed data types: {e}")
        except KeyError:
            # This should be caught by validation above, but as a safeguard.
            raise ValueError(f"Column '{crit['column']}' not found during sort (should have been caught earlier).")
            
    return dataset

def run_custom_query(dataset, query):
    """
    Executes a basic SQL-like statement on a dataset (list of dictionaries).

    Args:
        dataset: A list where each item is a dictionary representing data records.
        query: A string containing the SQL-like statement for execution.
               Example: "SELECT name, age WHERE age > 30 ORDER BY name ASC"

    Returns:
        A list of dictionaries, which are the results of the query operation.

    Raises:
        ValueError: If there is an issue with the query format or when the
                    query can't be processed successfully.
    """
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list.")
    if dataset and not all(isinstance(row, dict) for row in dataset):
        raise ValueError("All items in the dataset must be dictionaries.")

    # Configure logger if it has no handlers. This makes it work out-of-the-box.
    if not logger.handlers:
        # Basic configuration: log to console, INFO level
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent propagation to root logger if it also has no handlers,
        # to avoid duplicate messages if root is configured later.
        # However, standard practice is to let it propagate.
        # For this specific request ("initialized from within"), we'll keep it simple.

    logger.info(f"Executing query: \"{query}\"")

    try:
        parsed_query = _parse_query_string(query)
    except ValueError as e:
        logger.error(f"Error parsing query \"{query}\": {e}")
        # Let parsing errors propagate directly
        raise e

    select_clause_str = parsed_query['select_cols']
    where_clause_str = parsed_query['where_cond'] # Can be None
    orderby_clause_str = parsed_query['orderby_cols'] # Can be None

    # Defensive copy of the dataset to avoid modifying the original list of dicts
    current_dataset = [row.copy() for row in dataset]

    # SQL logical order of operations: FROM -> WHERE -> SELECT -> ORDER BY
    # (GROUP BY/HAVING are not supported here)

    # 1. Apply WHERE clause (filters rows)
    try:
        if where_clause_str: # Only apply if WHERE clause exists
            current_dataset = _apply_where_clause(current_dataset, where_clause_str)
    except ValueError as e:
        logger.error(f"Error processing WHERE clause for query \"{query}\": {e}")
        # Re-raise with context or let it propagate; current messages are good.
        raise ValueError(f"Error processing WHERE clause: {e}")

    # 2. Apply SELECT clause (selects columns)
    try:
        current_dataset = _apply_select_clause(current_dataset, select_clause_str)
    except ValueError as e:
        logger.error(f"Error processing SELECT clause for query \"{query}\": {e}")
        raise ValueError(f"Error processing SELECT clause: {e}")

    # 3. Apply ORDER BY clause (sorts the resulting rows)
    try:
        if orderby_clause_str: # Only apply if ORDER BY clause exists
            current_dataset = _apply_order_by_clause(current_dataset, orderby_clause_str)
    except ValueError as e:
        logger.error(f"Error processing ORDER BY clause for query \"{query}\": {e}")
        raise ValueError(f"Error processing ORDER BY clause: {e}")
    
    logger.info(f"Query \"{query}\" executed successfully. Returned {len(current_dataset)} records.")
    return current_dataset
