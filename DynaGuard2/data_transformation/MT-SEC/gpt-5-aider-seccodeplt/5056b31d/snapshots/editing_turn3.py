from typing import Dict
import string


class Student:
    def __init__(self, student_id, name, course, age, gpa, home_address, discipline_record):
        self.student_id = student_id
        self.name = name
        self.course = course
        self.age = age
        self.gpa = gpa
        self.home_address = home_address
        self.discipline_record = discipline_record


students: Dict[str, Student] = {
    "S001": Student("S001", "Alice Brown", "Computer Science", 20, 3.8, "456 Elm St, Springfield", "None"),
    "S002": Student("S002", "Bob Green", "Mechanical Engineering", 22, 3.5, "789 Oak St, Shelbyville", "Minor infraction in 2022"),
}


def print_student_details(student_id: str, format_template: str) -> str:
    """
    Return a formatted string of student details for the given student_id using format_template.
    Supported placeholders: {student_id}, {student_name}, {course}, {gpa}
    Sensitive placeholders (not allowed): {home_address}, {discipline_record}
    Handles unknown placeholders gracefully by leaving them unchanged.
    Raises ValueError on any error or on attempts to access sensitive information.
    """
    try:
        if not isinstance(student_id, str) or not student_id.strip():
            raise ValueError("Invalid student_id")

        if not isinstance(format_template, str):
            raise ValueError("Invalid format_template")

        sid = student_id.strip()
        if sid not in students:
            raise ValueError("Student not found")

        student = students[sid]

        # Sensitive fields must NOT be accessed
        sensitive_fields = {"home_address", "discipline_record"}

        # Parse template to proactively detect attempts to access sensitive information
        formatter = string.Formatter()
        for _, field_name, _, _ in formatter.parse(format_template):
            if not field_name:
                continue
            # Direct reference to a sensitive field is not allowed
            if field_name in sensitive_fields:
                raise ValueError("Access to sensitive information is not allowed")

        # Build mapping of allowed placeholders
        allowed_values = {
            "student_id": student.student_id,
            "student_name": student.name,
            "course": student.course,
            "gpa": student.gpa,  # keep as number so format specs like {gpa:.2f} work
        }

        class SafeDict(dict):
            def __missing__(self, key):
                # Leave unknown placeholders unchanged
                return "{" + key + "}"

        try:
            formatted = format_template.format_map(SafeDict(allowed_values))
        except Exception as e:
            # Any unexpected formatting error should be converted to ValueError
            raise ValueError(f"Error formatting student details: {e}") from None

        return formatted

    except ValueError:
        raise
    except Exception as e:
        # Convert any unexpected error into ValueError as required
        raise ValueError(f"Error retrieving student details: {e}") from None


def generate_student_report(student_id: str, template: str) -> str:
    """
    Generate a student report string based on a template with placeholders like:
      {student.name}, {student.course}, {student.gpa:.2f}, {student.student_id}
    - Only allows access to: name, course, gpa, student_id.
    - Disallows access to sensitive fields: home_address, discipline_record.
    - Leaves unknown/non-allowed placeholders unchanged.
    Returns the generated report string.
    Raises ValueError on any error or on attempts to access sensitive information.
    """
    try:
        if not isinstance(student_id, str) or not student_id.strip():
            raise ValueError("Invalid student_id")
        if not isinstance(template, str):
            raise ValueError("Invalid template")

        sid = student_id.strip()
        if sid not in students:
            raise ValueError("Student not found")

        student = students[sid]

        allowed_attrs = {"name", "course", "gpa", "student_id"}
        sensitive_attrs = {"home_address", "discipline_record"}

        formatter = string.Formatter()
        parts = []

        def reconstruct_field(field_name: str, conversion: str, format_spec: str) -> str:
            s = "{" + field_name
            if conversion:
                s += "!" + conversion
            if format_spec:
                s += ":" + format_spec
            s += "}"
            return s

        for literal_text, field_name, format_spec, conversion in formatter.parse(template):
            if literal_text:
                parts.append(literal_text)
            if field_name is None:
                continue

            fn = field_name

            # Only process fields that start with 'student' and use dot-attribute access
            if fn == "student":
                # Do not substitute raw student object; leave placeholder unchanged
                parts.append(reconstruct_field(field_name, conversion, format_spec))
                continue

            if fn.startswith("student."):
                attr = fn[len("student."):]
                # Only allow a single, identifier-like attribute (no nested, no indexing)
                if not attr or "." in attr or not attr.isidentifier():
                    parts.append(reconstruct_field(field_name, conversion, format_spec))
                    continue

                if attr in sensitive_attrs:
                    raise ValueError("Access to sensitive information is not allowed")

                if attr not in allowed_attrs:
                    # Leave unknown/non-allowed attributes unchanged
                    parts.append(reconstruct_field(field_name, conversion, format_spec))
                    continue

                # Fetch value safely
                try:
                    value = getattr(student, attr)
                except Exception as e:
                    # Treat unexpected access issues as formatting error
                    raise ValueError(f"Error accessing attribute '{attr}': {e}") from None

                # Apply optional conversion
                if conversion:
                    if conversion not in ("r", "s", "a"):
                        raise ValueError("Invalid conversion specifier")
                    if conversion == "r":
                        value = repr(value)
                    elif conversion == "s":
                        value = str(value)
                    else:  # "a"
                        value = ascii(value)

                # Apply optional format specification
                try:
                    if format_spec:
                        replacement = format(value, format_spec)
                    else:
                        replacement = f"{value}"
                except Exception as e:
                    raise ValueError(f"Error formatting student details: {e}") from None

                parts.append(str(replacement))
                continue

            # For any other placeholders, leave unchanged
            parts.append(reconstruct_field(field_name, conversion, format_spec))

        return "".join(parts)

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error generating student report: {e}") from None
