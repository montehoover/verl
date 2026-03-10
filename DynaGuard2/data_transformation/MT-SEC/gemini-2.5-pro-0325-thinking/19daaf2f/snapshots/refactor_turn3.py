import re
import operator
import logging
from functools import partial # As per problem setup, though not actively used in this version

# Global for operator mapping
_OPERATORS = {
    '=': operator.eq,
    '==': operator.eq, # Allow '==' for equality
    '!=': operator.ne,
    '<>': operator.ne, # Allow '<>' for inequality
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str):
    """Converts a string value from a query into its Python type."""
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # String literal

    # Try to parse as number (float then int)
    if '.' in value_str:
        try:
            return float(value_str)
        except ValueError:
            # Not a float, could be malformed or an unquoted string.
            # Fall through to try int or raise specific error.
            pass
    try:
        return int(value_str)
    except ValueError:
        # If it's not a number and not quoted, it's an invalid format for a literal value.
        raise ValueError(f"Invalid value format: '{value_str}'. String values must be quoted. Numeric values must be valid numbers.")

def _parse_where_condition(condition_str):
    """Parses a single WHERE condition like 'age > 30' or 'name = "Alice"'."""
    # Regex to capture field, operator, and value. Value is 'everything else', parsed by _parse_value.
    match = re.match(r"^\s*(\w+)\s*([<>=!]+)\s*(.+)$", condition_str.strip())
    if not match:
        raise ValueError(f"Invalid WHERE condition format: '{condition_str}'")

    field, op_str, value_literal = match.groups()
    
    op_func = _OPERATORS.get(op_str)
    if not op_func:
        raise ValueError(f"Unsupported operator: '{op_str}' in condition '{condition_str}'")

    try:
        value = _parse_value(value_literal)
    except ValueError as e:
        # Re-raise with more context if _parse_value failed
        raise ValueError(f"Error parsing value in WHERE condition '{condition_str}': {e}")

    return {'field': field.strip(), 'op': op_func, 'value': value}


def _parse_query_components(sql_query):
    """
    Parses the raw SQL query string into its logical components.
    Returns a dictionary with keys: 'select_columns', 'is_select_all', 
                                   'where_condition', 'orderby_specs'.
    """
    query = sql_query.strip()
    if not query:
        raise ValueError("SQL query cannot be empty.")

    # Regex for SELECT, optional WHERE, optional ORDER BY
    query_match = re.match(
        r"SELECT\s+(.+?)"
        r"(?:\s+WHERE\s+(.+?))?"
        r"(?:\s+ORDER BY\s+(.+?))?$",
        query,
        re.IGNORECASE
    )

    if not query_match:
        if not query.upper().startswith("SELECT "):
            raise ValueError("Query must start with SELECT.")
        raise ValueError("Invalid query structure. Expected format: SELECT cols [WHERE condition] [ORDER BY fields]")

    select_str, where_str, orderby_str = query_match.groups()

    # Parse SELECT columns
    if not select_str:
        raise ValueError("SELECT clause cannot be empty.")
    
    selected_columns_list = [col.strip() for col in select_str.split(',')]
    if not selected_columns_list or not all(col for col in selected_columns_list):
        raise ValueError("Invalid column specification in SELECT clause. Contains empty or invalid column names.")
    is_select_all = selected_columns_list == ['*']

    # Parse WHERE clause
    parsed_where_condition = None
    if where_str:
        if " AND " in where_str.upper() or " OR " in where_str.upper():
            raise ValueError("Compound WHERE conditions (AND/OR) are not supported in this version.")
        try:
            parsed_where_condition = _parse_where_condition(where_str)
        except ValueError as e:
            raise ValueError(f"Error parsing WHERE clause ('{where_str}'): {e}")

    # Parse ORDER BY clause
    parsed_orderby_specs = None
    if orderby_str:
        order_field_specs = []
        order_parts = [part.strip() for part in orderby_str.split(',')]
        for part in order_parts:
            if not part: continue
            
            order_match = re.match(r"(\w+)\s*(ASC|DESC)?$", part, re.IGNORECASE)
            if not order_match:
                raise ValueError(f"Invalid ORDER BY field format: '{part}'")
            
            col_name, direction_str = order_match.groups()
            is_descending = bool(direction_str and direction_str.upper() == "DESC")
            order_field_specs.append({'field': col_name, 'reverse': is_descending})
        if order_field_specs:
            parsed_orderby_specs = order_field_specs
            
    return {
        'select_columns': selected_columns_list,
        'is_select_all': is_select_all,
        'where_condition': parsed_where_condition,
        'orderby_specs': parsed_orderby_specs
    }


def _execute_where_clause(dataset, where_condition):
    """Applies the WHERE condition to filter the dataset."""
    if not where_condition:
        return list(dataset) # Return a copy if no condition

    field_to_check = where_condition['field']
    op_func = where_condition['op']
    value_to_compare = where_condition['value']
    
    filtered_data = []
    for record_idx, record in enumerate(dataset):
        if field_to_check not in record:
            raise ValueError(f"Field '{field_to_check}' in WHERE clause not found in record at index {record_idx}: {record}")

        record_value = record[field_to_check]
        
        try:
            if op_func(record_value, value_to_compare):
                filtered_data.append(record)
        except TypeError:
            raise ValueError(
                f"Type mismatch in WHERE condition for field '{field_to_check}'. "
                f"Cannot compare record value '{record_value}' (type: {type(record_value).__name__}) "
                f"with query value '{value_to_compare}' (type: {type(value_to_compare).__name__}) using operator '{op_func.__name__}'."
            )
    return filtered_data


def _execute_orderby_clause(dataset, orderby_specs):
    """Applies the ORDER BY specifications to sort the dataset."""
    if not orderby_specs or not dataset:
        return dataset # Return as-is if no specs or empty dataset

    # Validate order by columns against the first record (assuming consistent schema)
    # This check is now more localized here or could be done in run_sql_query before calling this.
    # For pure function, it's better if run_sql_query validates and passes valid specs.
    # However, for robustness within this function if called directly:
    first_record_keys = dataset[0].keys()
    for spec in orderby_specs:
        if spec['field'] not in first_record_keys:
            raise ValueError(f"Column '{spec['field']}' in ORDER BY clause not found in dataset records.")

    # Apply sorts in reverse order of precedence for stability
    # Create a mutable copy for sorting
    sorted_data = list(dataset)
    for sort_spec in reversed(orderby_specs):
        try:
            sorted_data.sort(key=operator.itemgetter(sort_spec['field']), reverse=sort_spec['reverse'])
        except KeyError: 
            # This should ideally be caught by pre-validation, but as a safeguard:
            raise ValueError(f"Column '{sort_spec['field']}' for sorting not found consistently in records.")
        except TypeError:
            raise ValueError(f"Cannot sort by column '{sort_spec['field']}' due to incompatible (mixed) data types in that column.")
    return sorted_data


def _execute_select_clause(dataset, select_columns, is_select_all):
    """Applies the SELECT transformation (projection) to the dataset."""
    if is_select_all:
        return dataset # Return as-is if SELECT *

    # Validate select columns against the first record (if dataset not empty)
    # Similar to ORDER BY, this could be pre-validated in run_sql_query.
    if dataset:
        first_record_keys = dataset[0].keys()
        for col_name in select_columns:
            if col_name not in first_record_keys:
                raise ValueError(f"Column '{col_name}' in SELECT clause not found in dataset records.")
    
    final_result = []
    for record in dataset:
        new_record = {}
        for col_name in select_columns:
            # This check is somewhat redundant if pre-validated, but good for direct calls
            if col_name not in record: 
                raise ValueError(f"Column '{col_name}' selected but not found in processed record: {record}.")
            new_record[col_name] = record[col_name]
        final_result.append(new_record)
    return final_result


def run_sql_query(dataset, sql_query):
    """
    Processes a custom SQL-like query on data represented as a list of dictionaries.

    Args:
        dataset: A list of dictionaries where each dictionary represents a record.
        sql_query: A string containing the SQL-like query.

    Returns:
        A list of dictionaries representing the query results.

    Raises:
        ValueError: If the query is malformed, a field is not found,
                    or types are incompatible for an operation.
    """
    logger = logging.getLogger(__name__)
    # Configure logger if it has no handlers (to avoid duplicate logs on multiple calls)
    if not logger.handlers:
        logger.setLevel(logging.INFO) # Or DEBUG for more verbosity
        handler = logging.StreamHandler() # Log to console
        # You could use a FileHandler here: handler = logging.FileHandler('sql_query.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Received SQL query: '{sql_query}'")
    logger.info(f"Initial dataset size: {len(dataset)} records.")

    try:
        # --- 0. Initial Validations ---
        if not isinstance(dataset, list):
            logger.error("Dataset validation failed: Dataset must be a list.")
            raise ValueError("Dataset must be a list.")
        if dataset and not all(isinstance(item, dict) for item in dataset):
            logger.error("Dataset validation failed: All items in the dataset must be dictionaries.")
            raise ValueError("All items in the dataset must be dictionaries.")

        # --- 1. Parse Query ---
        logger.info("Parsing query...")
        # _parse_query_components handles empty query string and basic structure validation
        parsed_query = _parse_query_components(sql_query)
        logger.info(f"Parsed query components: SELECT {parsed_query['select_columns']} (ALL: {parsed_query['is_select_all']}), "
                    f"WHERE {parsed_query['where_condition']}, ORDER BY {parsed_query['orderby_specs']}")

        # --- 2. Validate parsed components against dataset (early exit for bad columns) ---
        if dataset: # Only validate if dataset is not empty
            logger.debug("Validating parsed query components against dataset schema...")
            first_record_keys = dataset[0].keys()
            # Validate SELECT columns
            if not parsed_query['is_select_all']:
                for col_name in parsed_query['select_columns']:
                    if col_name not in first_record_keys:
                        logger.error(f"Validation Error: Column '{col_name}' in SELECT clause not found in dataset records.")
                        raise ValueError(f"Column '{col_name}' in SELECT clause not found in dataset records.")
            
            # Validate ORDER BY columns
            if parsed_query['orderby_specs']:
                for spec in parsed_query['orderby_specs']:
                    if spec['field'] not in first_record_keys:
                        logger.error(f"Validation Error: Column '{spec['field']}' in ORDER BY clause not found in dataset records.")
                        raise ValueError(f"Column '{spec['field']}' in ORDER BY clause not found in dataset records.")
            logger.debug("Schema validation successful.")
            # WHERE column validation is handled within _execute_where_clause as it processes records

        # --- 3. Execute Pipeline ---
        # Start with a copy of the dataset for processing
        processed_data = list(dataset) 
        logger.info("Starting query execution pipeline...")

        # Apply WHERE
        if parsed_query['where_condition']:
            logger.info(f"Applying WHERE clause: {parsed_query['where_condition']}")
            processed_data = _execute_where_clause(processed_data, parsed_query['where_condition'])
            logger.info(f"After WHERE clause, dataset size: {len(processed_data)} records.")
        else:
            logger.info("No WHERE clause to apply.")

        # Apply ORDER BY
        if parsed_query['orderby_specs'] and processed_data: # Only sort if specs exist and data remains
            logger.info(f"Applying ORDER BY clause: {parsed_query['orderby_specs']}")
            processed_data = _execute_orderby_clause(processed_data, parsed_query['orderby_specs'])
            logger.info(f"After ORDER BY clause, dataset size: {len(processed_data)} records (order may have changed).")
        elif not processed_data and parsed_query['orderby_specs']:
            logger.info("ORDER BY clause present, but dataset is empty after WHERE. Skipping sort.")
        else:
            logger.info("No ORDER BY clause to apply or dataset empty.")

        # Apply SELECT (Projection)
        logger.info(f"Applying SELECT clause: columns={parsed_query['select_columns']}, is_select_all={parsed_query['is_select_all']}")
        # _execute_select_clause handles the is_select_all case internally
        processed_data = _execute_select_clause(
            processed_data, 
            parsed_query['select_columns'], 
            parsed_query['is_select_all']
        )
        logger.info(f"After SELECT clause, final result size: {len(processed_data)} records.")
        # For very large results, logging the full data might be too verbose.
        # Consider logging only a sample or summary if `len(processed_data)` is large.
        # For now, let's log a small sample if it's too big.
        if len(processed_data) > 5:
            logger.debug(f"Final result sample (first 5 records): {processed_data[:5]}")
        else:
            logger.debug(f"Final result: {processed_data}")
        
        return processed_data
    except ValueError as e:
        logger.error(f"ValueError during query execution: {e}", exc_info=True) # exc_info=True adds stack trace
        raise # Re-raise the exception after logging
    except Exception as e:
        logger.critical(f"An unexpected error occurred during query execution: {e}", exc_info=True)
        raise ValueError(f"Unexpected error during query execution: {e}") # Wrap in ValueError as per original spec
    finally:
        logger.info(f"Finished processing query: '{sql_query}'")
