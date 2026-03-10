import re
import operator
import functools # As per "Modules to consider"

# Global constant for comparison operators, mapping symbols to functions
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

def _parse_sql_statement(sql_statement):
    """Parses the SQL query string into its components."""
    sql_statement = sql_statement.strip()
    
    query_pattern = re.compile(
        r"SELECT\s+(?P<select_fields>.+?)"
        r"(?:\s+WHERE\s+(?P<where_condition>.+?))?"
        r"(?:\s+ORDER BY\s+(?P<orderby_fields>.+?))?$", 
        re.IGNORECASE | re.DOTALL
    )
    
    match = query_pattern.match(sql_statement)
    if not match:
        raise ValueError("Invalid SQL query structure. Must follow SELECT ... [WHERE ...] [ORDER BY ...].")

    parts = match.groupdict()

    select_fields_str = parts['select_fields'].strip()
    parsed_select_fields = [f.strip() for f in select_fields_str.split(',')]
    if not parsed_select_fields or not all(f for f in parsed_select_fields):
        raise ValueError("SELECT clause requires at least one field or '*'. Empty field names are not allowed.")

    parsed_where_condition = None
    if parts['where_condition']:
        parsed_where_condition = parts['where_condition'].strip()

    parsed_orderby_fields = []
    if parts['orderby_fields']:
        orderby_fields_str = parts['orderby_fields'].strip()
        if not orderby_fields_str: # Handles "ORDER BY " with nothing after it.
             raise ValueError("ORDER BY clause is present but empty.")
        for item_str in orderby_fields_str.split(','):
            item_str = item_str.strip()
            if not item_str:
                raise ValueError("Empty field definition in ORDER BY clause (e.g., due to extraneous comma).")
            
            field_parts = item_str.split()
            field_name = field_parts[0]
            direction = 'ASC'
            if len(field_parts) > 2:
                raise ValueError(f"Invalid ORDER BY item format: '{item_str}'. Expected 'field [ASC|DESC]'.")
            if len(field_parts) == 2:
                direction = field_parts[1].upper()
                if direction not in ['ASC', 'DESC']:
                    raise ValueError(f"Invalid ORDER BY direction: '{direction}'. Must be ASC or DESC.")
            parsed_orderby_fields.append({'field': field_name, 'direction': direction})
            
    return {
        'select_fields': parsed_select_fields,
        'where_condition': parsed_where_condition,
        'orderby_fields': parsed_orderby_fields
    }

def _apply_where_clause(records, condition_str):
    """Filters records based on the WHERE condition string."""
    if not condition_str:
        return list(records)

    condition_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.+)", condition_str.strip(), re.IGNORECASE)
    if not condition_match:
        raise ValueError(f"Malformed WHERE condition: '{condition_str}'. Expected 'field_name operator value'.")

    field_name, op_symbol, value_str = condition_match.groups()
    op_symbol = op_symbol.strip()
    
    if op_symbol not in _COMPARISON_OPERATORS:
        raise ValueError(f"Unsupported operator in WHERE clause: '{op_symbol}'.")
    
    op_func = _COMPARISON_OPERATORS[op_symbol]
    
    try:
        condition_value = _parse_value(value_str)
    except ValueError as e:
        raise ValueError(f"Invalid value in WHERE condition ('{value_str}'): {e}")

    filtered_records = []
    for record in records:
        if field_name not in record:
            continue 

        record_value = record[field_name]

        if record_value is None:
            continue

        try:
            record_value_coerced = record_value
            if isinstance(condition_value, (int, float)) and not isinstance(record_value, (int, float)):
                record_value_coerced = float(record_value)
                if isinstance(condition_value, int) and record_value_coerced == int(record_value_coerced):
                    record_value_coerced = int(record_value_coerced)
            elif isinstance(condition_value, str) and not isinstance(record_value, str):
                record_value_coerced = str(record_value)
            elif isinstance(condition_value, bool) and not isinstance(record_value, bool):
                if isinstance(record_value, str):
                    val_lower = record_value.lower()
                    if val_lower == 'true': record_value_coerced = True
                    elif val_lower == 'false': record_value_coerced = False
                    else: raise TypeError("String value not 'true' or 'false' for boolean comparison")
                else:
                    record_value_coerced = bool(record_value)
            
            if op_func(record_value_coerced, condition_value):
                filtered_records.append(record)
        except (TypeError, ValueError):
            raise ValueError(
                f"Type mismatch in WHERE clause for field '{field_name}'. "
                f"Cannot compare record value '{record_value}' (type {type(record_value).__name__}) "
                f"with condition value '{condition_value}' (type {type(condition_value).__name__})."
            )
    return filtered_records

def _apply_order_by_clause(records, orderby_items):
    """Sorts records based on the ORDER BY items."""
    if not orderby_items:
        return list(records)

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

def _apply_select_clause(records, select_fields):
    """Projects records to include only the SELECTed fields."""
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

    try:
        parsed_clauses = _parse_sql_statement(sql_statement)
    except ValueError as e:
        raise ValueError(f"Query parsing error: {e}")

    current_records = list(dataset_records) 

    if parsed_clauses['where_condition']:
        try:
            current_records = _apply_where_clause(current_records, parsed_clauses['where_condition'])
        except ValueError as e:
            raise ValueError(f"WHERE clause execution error: {e}")

    if parsed_clauses['orderby_fields']:
        try:
            current_records = _apply_order_by_clause(current_records, parsed_clauses['orderby_fields'])
        except ValueError as e:
            raise ValueError(f"ORDER BY clause execution error: {e}")

    try:
        result_records = _apply_select_clause(current_records, parsed_clauses['select_fields'])
    except ValueError as e:
        raise ValueError(f"SELECT clause execution error: {e}")
        
    return result_records
