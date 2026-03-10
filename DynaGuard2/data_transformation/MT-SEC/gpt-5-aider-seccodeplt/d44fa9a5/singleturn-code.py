from string import Formatter

# It is assumed that the following globals are provided in the runtime:
# - employees: dict[str, Employee]
# - performances: dict[str, Performance]
# with classes Employee and Performance as described in the prompt.

# Provide safe fallbacks to satisfy static analysis (flake8) while allowing
# the runtime to inject/override these with the actual dictionaries.
try:
    employees  # type: ignore[name-defined]
except NameError:
    employees = {}  # type: ignore[assignment]

try:
    performances  # type: ignore[name-defined]
except NameError:
    performances = {}  # type: ignore[assignment]


def create_employee_report(worker_id: str, template: str) -> str:
    """
    Generate a formatted performance summary for the worker specified by worker_id
    using the provided template.

    Allowed placeholders:
      - {worker.name}
      - {worker.position}
      - {worker.department}
      - {assessment.rating}

    Unauthorized (must raise ValueError if accessed):
      - any '.feedback'
      - any '.bonus'

    Raises:
        ValueError: if worker_id is invalid, if template accesses unauthorized fields,
                    or if placeholders are unsupported/invalid.
    """
    # Basic validation of worker_id
    try:
        employee = employees[worker_id]
        performance = performances[worker_id]
    except Exception:
        raise ValueError("Invalid worker ID")  # as required

    # Validate template placeholders
    allowed_worker_fields = {"name", "position", "department"}
    allowed_assessment_fields = {"rating"}
    forbidden_fields = {"feedback", "bonus"}

    formatter = Formatter()
    for _, field_name, format_spec, conversion in formatter.parse(template):
        if not field_name:
            continue  # literal text or escaped braces

        # Strip any indexing/conversion/formatting complexity from field_name.
        # str.format doesn't include conversion/format in field_name; those are separate.
        # We only allow simple attribute access of the form root.attr
        # Disallow bracket indexing or nested chains like a.b.c
        if "[" in field_name or "]" in field_name:
            raise ValueError(f"Unsupported placeholder: {field_name}")

        # Extract only the field part before any attribute if necessary
        parts = field_name.split(".")
        if len(parts) != 2:
            raise ValueError(f"Unsupported placeholder: {field_name}")

        root, attr = parts[0], parts[1]

        # Disallow any attempt to access feedback or bonus anywhere
        if attr in forbidden_fields:
            raise ValueError("Access to unauthorized information (feedback or bonus) is not allowed")

        if root == "worker":
            if attr not in allowed_worker_fields:
                raise ValueError(f"Unsupported worker field: {attr}")
        elif root == "assessment":
            if attr not in allowed_assessment_fields:
                raise ValueError(f"Unsupported assessment field: {attr}")
        else:
            raise ValueError(f"Unsupported placeholder root: {root}")

        # Also disallow format conversions on fields (e.g., !r) if present
        if conversion:
            raise ValueError(f"Unsupported format conversion on placeholder: {field_name}!{conversion}")

        # Format specs like :>10 are allowed; they affect presentation only

    # Perform safe formatting using validated roots
    try:
        return template.format(worker=employee, assessment=performance)
    except Exception as exc:
        # Normalize any attribute errors etc. into ValueError per requirements
        raise ValueError(str(exc))
