import re
import operator
import functools # As per "Modules to consider"
import logging
import sys

# Global constant for comparison operators, mapping symbols to functions
# Get a logger instance for this module
logger = logging.getLogger(__name__)

_COMPARISON_OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str):
    """Converts a string value from a SQL query to a Python type."""
    value_str = value_str.strip()
    # Check for string literals
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]
    # Check for boolean literals
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    # Try converting to number (int, then float)
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            raise ValueError(f"Unsupported value literal in WHERE clause: {value_str}")

def parse_sql_query(sql_statement):
    """
    Parses the SQL query string into a structured representation.
    Returns a dictionary with 'select_fields', 'where_details', and 'orderby_fields'.
    """
    sql_statement = sql_statement.strip()
    
    # Regex to capture SELECT, optional WHERE, and optional ORDER BY clauses
    # Renamed capture groups for clarity (e.g., where_condition_str)
    query_pattern = re.compile(
        r"SELECT\s+(?P<select_fields_str>.+?)"
        r"(?:\s+WHERE\s+(?P<where_condition_str>.+?))?"
        r"(?:\s+ORDER BY\s+(?P<orderby_fields_str>.+?))?$", 
        re.IGNORECASE | re.DOTALL
    )
    
    match = query_pattern.match(sql_statement)
    if not match:
        raise ValueError("Invalid SQL query structure. Must follow SELECT ... [WHERE ...] [ORDER BY ...].")

    parts = match.groupdict()

    # Parse SELECT fields
    select_fields_str = parts['select_fields_str'].strip()
    parsed_select_fields = [f.strip() for f in select_fields_str.split(',')]
    if not parsed_select_fields or not all(f for f in parsed_select_fields): # Ensure no empty field names like 'name,,age'
        raise ValueError("SELECT clause requires at least one field or '*'. Empty field names are not allowed.")

    # Parse WHERE condition string into structured details
    parsed_where_details = None
    if parts['where_condition_str']:
        where_condition_str = parts['where_condition_str'].strip()
        # Regex for 'field_name operator value'
        condition_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.+)", where_condition_str, re.IGNORECASE)
        if not condition_match:
            raise ValueError(f"Malformed WHERE condition: '{where_condition_str}'. Expected 'field_name operator value'.")

        field_name, op_symbol, value_str = condition_match.groups()
        op_symbol = op_symbol.strip() # Ensure no extra spaces around operator
        
        if op_symbol not in _COMPARISON_OPERATORS:
            raise ValueError(f"Unsupported operator in WHERE clause: '{op_symbol}'.")
        
        op_func = _COMPARISON_OPERATORS[op_symbol]
        
        try:
            # _parse_value handles string, boolean, int, float literals
            condition_value = _parse_value(value_str)
        except ValueError as e:
            # Error from _parse_value is specific, e.g., "Unsupported value literal..."
            raise ValueError(f"Invalid value in WHERE condition ('{value_str}'): {e}")

        parsed_where_details = {
            'field': field_name,
            'operator_func': op_func, # Store the actual operator function
            'value': condition_value  # Store the typed value
        }

    # Parse ORDER BY fields
    parsed_orderby_fields = []
    if parts['orderby_fields_str']:
        orderby_fields_str = parts['orderby_fields_str'].strip()
        if not orderby_fields_str: # Handles "ORDER BY " with nothing after it.
             raise ValueError("ORDER BY clause is present but empty.")
        for item_str in orderby_fields_str.split(','):
            item_str = item_str.strip()
            if not item_str: # Handles "field1, , field2"
                raise ValueError("Empty field definition in ORDER BY clause (e.g., due to extraneous comma).")
            
            field_parts = item_str.split() # Splits by space, e.g., "age DESC"
            field_name = field_parts[0]
            direction = 'ASC' # Default direction
            if len(field_parts) > 2: # e.g. "age DESC extra"
                raise ValueError(f"Invalid ORDER BY item format: '{item_str}'. Expected 'field [ASC|DESC]'.")
            if len(field_parts) == 2:
                direction = field_parts[1].upper()
                if direction not in ['ASC', 'DESC']:
                    raise ValueError(f"Invalid ORDER BY direction: '{direction}'. Must be ASC or DESC.")
            parsed_orderby_fields.append({'field': field_name, 'direction': direction})
            
    return {
        'select_fields': parsed_select_fields,
        'where_details': parsed_where_details, # Contains structured WHERE info or None
        'orderby_fields': parsed_orderby_fields
    }

def _execute_where_clause(records, where_details):
    """Filters records based on the pre-parsed WHERE details."""
    if not where_details: # No WHERE clause to apply
        return list(records)

    field_name = where_details['field']
    op_func = where_details['operator_func']
    condition_value = where_details['value']

    filtered_records = []
    for record in records:
        if field_name not in record:
            # If field doesn't exist in record, row doesn't match condition.
            continue 

        record_value = record[field_name]

        if record_value is None: # SQL NULL comparison semantics (usually false, except for IS NULL)
            continue

        try:
            record_value_coerced = record_value
            # Attempt type coercion if record value type differs from condition value type
            if isinstance(condition_value, (int, float)) and not isinstance(record_value, (int, float)):
                try:
                    record_value_coerced = float(record_value) # Try float first
                    # If condition is int and coerced value is whole number, convert to int
                    if isinstance(condition_value, int) and record_value_coerced == int(record_value_coerced):
                         record_value_coerced = int(record_value_coerced)
                except (ValueError, TypeError): 
                    continue # Skip record if value cannot be coerced to a number for comparison
            elif isinstance(condition_value, str) and not isinstance(record_value, str):
                record_value_coerced = str(record_value)
            elif isinstance(condition_value, bool) and not isinstance(record_value, bool):
                if isinstance(record_value, str): # Handle 'true'/'false' strings for bool comparison
                    val_lower = record_value.lower()
                    if val_lower == 'true': record_value_coerced = True
                    elif val_lower == 'false': record_value_coerced = False
                    else: 
                        continue # Skip record if string is not 'true'/'false' for bool comparison
                else: # Try direct conversion to bool (e.g., 0/1 for int/float)
                    record_value_coerced = bool(record_value)
            
            # Perform the comparison using the operator function
            if op_func(record_value_coerced, condition_value):
                filtered_records.append(record)
        except (TypeError, ValueError):
            # This error occurs if op_func fails due to fundamentally incompatible types
            # even after coercion attempts (e.g. comparing "text" < 10).
            raise ValueError(
                f"Type mismatch in WHERE clause for field '{field_name}'. "
                f"Cannot compare record value '{record_value}' (type {type(record_value).__name__}) "
                f"with condition value '{condition_value}' (type {type(condition_value).__name__}) after coercion."
            )
    return filtered_records

def _execute_order_by_clause(records, orderby_items):
    """Sorts records based on the pre-parsed ORDER BY items."""
    if not orderby_items:
        return list(records)

    # cmp_to_key requires a comparison function
    def compare_records(rec1, rec2):
        for item in orderby_items:
            field_name = item['field']
            is_desc = item['direction'] == 'DESC'

            val1 = rec1.get(field_name)
            val2 = rec2.get(field_name)
            
            if val1 is None and val2 is None: cmp_res = 0
            elif val1 is None: cmp_res = -1
            elif val2 is None: cmp_res = 1
            else:
                try:
                    if val1 < val2: cmp_res = -1
                    elif val1 > val2: cmp_res = 1
                    else: cmp_res = 0
                except TypeError:
                    raise ValueError(
                        f"Type error during ORDER BY on field '{field_name}'. "
                        f"Cannot compare values '{val1}' (type {type(val1).__name__}) and "
                        f"'{val2}' (type {type(val2).__name__}). Ensure data in sortable columns is comparable."
                    )
            
            if cmp_res != 0:
                return cmp_res if not is_desc else -cmp_res
        return 0

    try:
        return sorted(records, key=functools.cmp_to_key(compare_records))
    except ValueError as e:
        raise e
    except Exception as e_sort: 
        raise ValueError(f"Unexpected error during sorting process: {e_sort}")

def _execute_select_clause(records, select_fields):
    """Projects records to include only the SELECTed fields, based on pre-parsed select_fields."""
    if not select_fields:
        raise ValueError("SELECT fields list cannot be empty.")

    if select_fields == ['*']:
        return [dict(r) for r in records]

    projected_records = []
    for record in records:
        new_record = {}
        for field_name in select_fields:
            if field_name not in record:
                raise ValueError(f"Selected field '{field_name}' not found in record: {record}. "
                                 "All selected fields must exist in records after WHERE and ORDER BY clauses.")
            new_record[field_name] = record[field_name]
        projected_records.append(new_record)
    return projected_records

def process_sql_request(dataset_records, sql_statement):
    """
    Processes a custom SQL-like query on data represented as a list of dictionaries.
    Handles SELECT, WHERE, and ORDER BY clauses.

    Args:
        dataset_records: A list of dictionaries, where each dictionary is a record.
        sql_statement: A string containing the SQL-like query.

    Returns:
        A list of dictionaries representing the results of the executed query.

    Raises:
        ValueError: If the query is malformed, execution fails due to type mismatches,
                    missing fields, or other operational errors.
    """
    if not isinstance(dataset_records, list):
        raise ValueError("Dataset must be a list.")
    if dataset_records and not all(isinstance(r, dict) for r in dataset_records):
        raise ValueError("All items in a non-empty dataset must be dictionaries.")
    if not isinstance(sql_statement, str):
        raise ValueError("SQL statement must be a string.")

    # Configure logger if it has no handlers (e.g., first call in an environment without prior logging setup)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Create a handler (e.g., StreamHandler to log to console/stdout)
        ch = logging.StreamHandler(sys.stdout)
        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(ch)
        # Prevent the log messages from being duplicated in the root logger's handlers
        logger.propagate = False

    logger.info("Processing SQL query: %s", sql_statement)

    # Pipeline Step 1: Parse the SQL query string into structured components
    try:
        parsed_query_components = parse_sql_query(sql_statement)
        logger.info("Parsed query components: %s", parsed_query_components)
    except ValueError as e:
        logger.error("Query parsing error: %s", e, exc_info=True)
        # Errors from parse_sql_query are already specific.
        # Prepend a general context for clarity if needed, but logger includes it.
        raise ValueError(f"Query parsing error: {e}")

    # Pipeline Step 2: Execute the parsed query components on the dataset
    try:
        result_records = execute_parsed_query(dataset_records, parsed_query_components)
        logger.info("Query successful. Returned %d records. Preview (first 3): %s", 
                    len(result_records), result_records[:3] if result_records else "[]")
    except ValueError as e:
        logger.error("Query execution error: %s", e, exc_info=True)
        # Errors from execute_parsed_query already include context.
        # Re-raise them directly.
        raise e 
        
    return result_records

# New function to execute the parsed query components in a pipeline
def execute_parsed_query(dataset_records, parsed_query_components):
    """
    Executes the parsed SQL query components (WHERE, ORDER BY, SELECT)
    on the dataset. This function forms the execution part of the pipeline.

    Args:
        dataset_records: The initial list of records (list of dicts).
        parsed_query_components: A dictionary from parse_sql_query, containing 
                                 'where_details', 'orderby_fields', 
                                 and 'select_fields'.

    Returns:
        A list of dictionaries representing the results of the query.
    
    Raises:
        ValueError: If execution fails at any step (e.g., type mismatches during
                    comparison, missing fields during selection).
    """
    current_records = list(dataset_records) # Work on a copy

    # Step 1: Apply WHERE clause (filter records)
    if parsed_query_components['where_details']:
        try:
            current_records = _execute_where_clause(current_records, parsed_query_components['where_details'])
        except ValueError as e:
            # Propagate error, potentially adding context if needed, but _execute_where_clause should be specific
            raise ValueError(f"WHERE clause execution error: {e}")

    # Step 2: Apply ORDER BY clause (sort records)
    if parsed_query_components['orderby_fields']:
        try:
            current_records = _execute_order_by_clause(current_records, parsed_query_components['orderby_fields'])
        except ValueError as e:
            raise ValueError(f"ORDER BY clause execution error: {e}")

    # Step 3: Apply SELECT clause (project fields)
    # This is always present as parse_sql_query ensures select_fields is populated.
    try:
        result_records = _execute_select_clause(current_records, parsed_query_components['select_fields'])
    except ValueError as e:
        raise ValueError(f"SELECT clause execution error: {e}")
        
    return result_records
