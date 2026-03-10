import string

class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        # Store sensitive attributes privately and block read access via properties
        self._gpa = gpa
        self._home_address = home_address
        self.discipline_record = discipline_record

    @property
    def gpa(self):
        raise ValueError("Access to sensitive attribute 'gpa' is prohibited.")

    @property
    def home_address(self):
        raise ValueError("Access to sensitive attribute 'home_address' is prohibited.")


students = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022"),
}


def _safe_format(template, mapping):
    """
    Safely formats a template string with the provided mapping.
    - Leaves unknown placeholders unchanged.
    - Supports standard format conversions (!r, !s, !a) and format specs.
    """
    formatter = string.Formatter()
    parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        parts.append(literal_text)
        if field_name is None:
            continue

        if field_name in mapping:
            value = mapping[field_name]
            # Apply conversion if specified
            if conversion == 'r':
                value = repr(value)
            elif conversion == 'a':
                value = ascii(value)
            elif conversion == 's':
                value = str(value)

            if format_spec:
                try:
                    parts.append(format(value, format_spec))
                except Exception:
                    # Fallback to simple string conversion if formatting fails
                    parts.append(str(value))
            else:
                parts.append(str(value))
        else:
            # Unknown placeholder, leave it as-is including any conversion/spec
            placeholder = "{"
            placeholder += field_name
            if conversion:
                placeholder += f"!{conversion}"
            if format_spec:
                placeholder += f":{format_spec}"
            placeholder += "}"
            parts.append(placeholder)
    return "".join(parts)


def print_student_details(student_id, format_template):
    """
    Returns a formatted string of student details using the provided format template.

    The template can include the following placeholders:
      - {student_id}
      - {student_name} (alias for {name})
      - {name}
      - {course}
      - {age}

    Sensitive attributes (gpa, home_address) are prohibited and will raise ValueError if requested.

    Args:
        student_id (str): The student ID to look up.
        format_template (str): The format string containing placeholders.

    Returns:
        str: The formatted student details string.

    Raises:
        ValueError: If student_id does not exist or if the template attempts to access sensitive attributes.
    """
    if student_id not in students:
        raise ValueError(f"Student ID '{student_id}' does not exist.")

    # Detect attempts to access sensitive attributes via the template
    formatter = string.Formatter()
    requested_fields = {
        field_name
        for _, field_name, _, _ in formatter.parse(format_template)
        if field_name
    }
    prohibited = {"gpa", "home_address"}
    if requested_fields & prohibited:
        raise ValueError("Access to sensitive attributes 'gpa' or 'home_address' is prohibited.")

    student = students[student_id]

    # Only include non-sensitive attributes in the mapping
    mapping = {
        "student_id": student.student_id,
        "student_name": student.name,
        "name": student.name,
        "course": student.course,
        "age": student.age,
    }

    return _safe_format(format_template, mapping)


def build_student_profile(unique_id, report_template):
    """
    Builds a student report by replacing placeholders that reference the student object.

    Supported placeholders (examples):
      - {student.name}
      - {student.course}
      - {student.age}
      - {student.student_id}
      - {student.discipline_record}
    Notes:
      - Sensitive attributes {student.gpa} and {student.home_address} are prohibited and will raise ValueError.
      - Unknown or missing placeholders are left unchanged.
      - Supports standard conversions (!r, !s, !a) and format specs (e.g., {student.age:d}).

    Args:
        unique_id (str): The student ID to look up.
        report_template (str): The template string containing placeholders.

    Returns:
        str: The generated report string.

    Raises:
        ValueError: If unique_id does not exist or if the template attempts to access sensitive attributes.
    """
    if unique_id not in students:
        raise ValueError(f"Student ID '{unique_id}' does not exist.")

    student = students[unique_id]
    prohibited = {"gpa", "home_address"}

    formatter = string.Formatter()

    # Pre-scan for prohibited fields
    requested_fields = {
        field_name
        for _, field_name, _, _ in formatter.parse(report_template)
        if field_name
    }
    for field in requested_fields:
        if field.startswith("student."):
            first_attr = field.split(".", 1)[1].split(".", 1)[0]
            if first_attr in prohibited:
                raise ValueError("Access to sensitive attributes 'gpa' or 'home_address' is prohibited.")

    def _placeholder_text(field_name, conversion, format_spec):
        txt = "{"
        txt += field_name
        if conversion:
            txt += f"!{conversion}"
        if format_spec:
            txt += f":{format_spec}"
        txt += "}"
        return txt

    parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(report_template):
        parts.append(literal_text)
        if field_name is None:
            continue

        # Only resolve placeholders that start with 'student.'
        if not field_name.startswith("student."):
            parts.append(_placeholder_text(field_name, conversion, format_spec))
            continue

        # Resolve dotted attributes on the student object
        current = student
        ok = True
        # Skip the leading 'student'
        for attr in field_name.split(".")[1:]:
            # Prevent private/dunder and callables
            if not attr or attr.startswith("_"):
                ok = False
                break
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
            if callable(current):
                ok = False
                break

        if not ok:
            parts.append(_placeholder_text(field_name, conversion, format_spec))
            continue

        value = current

        # Apply conversion if specified
        if conversion == "r":
            value = repr(value)
        elif conversion == "a":
            value = ascii(value)
        elif conversion == "s":
            value = str(value)

        if format_spec:
            try:
                parts.append(format(value, format_spec))
            except Exception:
                parts.append(str(value))
        else:
            parts.append(str(value))

    return "".join(parts)
