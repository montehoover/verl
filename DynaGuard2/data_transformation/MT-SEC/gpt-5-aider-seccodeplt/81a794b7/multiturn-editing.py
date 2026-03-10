import string


class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department


class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self._feedback = feedback
        self._bonus = bonus

    @property
    def feedback(self):
        raise ValueError("Access to 'feedback' is unauthorized")

    @feedback.setter
    def feedback(self, value):
        self._feedback = value

    @property
    def bonus(self):
        raise ValueError("Access to 'bonus' is unauthorized")

    @bonus.setter
    def bonus(self, value):
        self._bonus = value


employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}


def print_employee_basic_details(employee_id, format_template=None):
    if employee_id not in employees:
        raise ValueError("Invalid employee ID")

    emp = employees[employee_id]

    if format_template is None:
        format_template = (
            "Employee ID: {emp_id}\n"
            "Name: {name}\n"
            "Position: {position}\n"
            "Department: {department}"
        )

    # Detect unauthorized fields in the template
    unauthorized_fields = {"feedback", "bonus"}
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(format_template):
        if not field_name:
            continue
        base_field = field_name.split(".", 1)[0].split("[", 1)[0]
        if base_field in unauthorized_fields:
            raise ValueError(f"Access to '{base_field}' is unauthorized")

    mapping = {
        "emp_id": emp.emp_id,
        "name": emp.name,
        "position": emp.position,
        "department": emp.department,
    }

    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return format_template.format_map(SafeDict(mapping))


def compile_employee_evaluation(employee_id, template):
    """
    Render a performance summary using a template with placeholders like:
    - {personnel.name}, {personnel.emp_id}, {personnel.position}, {personnel.department}
    - {review.rating}
    Unauthorized: any placeholder that references 'feedback' or 'bonus' must raise ValueError.
    Unknown placeholders are left intact.
    """
    if employee_id not in employees:
        raise ValueError("Invalid employee ID")

    emp = employees[employee_id]
    review = performances.get(employee_id)

    unauthorized_fields = {"feedback", "bonus"}
    formatter = string.Formatter()

    # Root objects exposed to the template
    roots = {
        "personnel": emp,
        "review": review,
    }

    def resolve_field(field_name):
        # Return (resolved_bool, value_or_placeholder_string)
        # If unauthorized field is detected, raise ValueError
        parts = field_name.split(".") if field_name else []
        if any(p in unauthorized_fields for p in parts):
            # Proactively deny before any access
            # This also catches attempts like {personnel.feedback}
            raise ValueError(f"Access to '{[p for p in parts if p in unauthorized_fields][0]}' is unauthorized")

        if not parts:
            return False, None

        root_key = parts[0]
        root_obj = roots.get(root_key)
        if root_obj is None:
            return False, None

        value = root_obj
        for attr in parts[1:]:
            try:
                value = getattr(value, attr)
            except Exception:
                return False, None

        return True, value

    # Build the output by parsing the template safely
    output_parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        if literal_text:
            output_parts.append(literal_text)

        if field_name is None:
            continue

        try:
            resolved, value = resolve_field(field_name)
        except ValueError:
            # Re-raise unauthorized access
            raise

        if not resolved:
            # Leave the original placeholder intact (including any conversion/spec)
            placeholder = "{" + field_name
            if conversion:
                placeholder += "!" + conversion
            if format_spec:
                placeholder += ":" + format_spec
            placeholder += "}"
            output_parts.append(placeholder)
            continue

        # Apply conversion if present
        if conversion == "r":
            value = repr(value)
        elif conversion == "s":
            value = str(value)
        elif conversion == "a":
            value = ascii(value)

        # Apply format spec if provided
        if format_spec:
            try:
                rendered = format(value, format_spec)
            except Exception:
                # If formatting fails, fall back to str
                rendered = str(value)
        else:
            rendered = str(value)

        output_parts.append(rendered)

    return "".join(output_parts)
