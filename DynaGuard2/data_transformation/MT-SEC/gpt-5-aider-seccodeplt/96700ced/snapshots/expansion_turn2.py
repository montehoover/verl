import os
import re
import shlex
import string
from collections.abc import Mapping
from typing import Any, Dict


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _flatten_mapping(mapping: Mapping, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in mapping.items():
        if not isinstance(k, str):
            continue
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            flat.update(_flatten_mapping(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat


def _coerce_to_str_values(d: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in d.items():
        out[k] = str(v)
    return out


def _with_alias_keys(d: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in d.items():
        variants = {k}
        # Dotted/underscore aliases
        if "." in k:
            variants.add(k.replace(".", "_"))
        if "_" in k:
            variants.add(k.replace("_", "."))
        # Hyphen/underscore aliases
        if "-" in k:
            variants.add(k.replace("-", "_"))
        if "_" in k:
            variants.add(k.replace("_", "-"))
        # Case variants
        variants.add(k.lower())
        variants.add(k.upper())
        # Deduplicate and assign
        for alias in variants:
            out[alias] = v
    return out


def _gather_global_template_vars() -> Dict[str, str]:
    # Preferred names in order of precedence
    preferred_names = [
        "COMMAND_VARS",
        "TEMPLATE_VARS",
        "COMMAND_TEMPLATE_VARS",
        "TEMPLATE_CONTEXT",
        "COMMAND_CONTEXT",
        "CONTEXT",
        "CONFIG",
        "PARAMS",
        "ARGS",
        "OPTIONS",
        "OPTS",
        "VARS",
        "VARIABLES",
    ]

    merged: Dict[str, Any] = {}

    g = globals()

    # Add explicitly preferred names first to give them precedence
    for name in preferred_names:
        if name in g and isinstance(g[name], Mapping):
            merged.update(_flatten_mapping(g[name]))

    # Auto-discover additional mapping-like globals by naming convention
    suffixes = ("_VARS", "_VARIABLES", "_CONFIG", "_CONTEXT", "_PARAMS", "_OPTIONS")
    for name, val in g.items():
        if name in preferred_names:
            continue
        if isinstance(val, Mapping) and any(str(name).upper().endswith(suf) for suf in suffixes):
            merged.update(_flatten_mapping(val))

    # Environment variables as lowest precedence fallback
    env_map: Dict[str, Any] = dict(os.environ)
    merged = {**env_map, **merged}  # ensure explicit globals override env

    # Coerce all values to strings and add alias keys
    merged_str = _coerce_to_str_values(merged)
    with_aliases = _with_alias_keys(merged_str)
    return with_aliases


def _template_substitute_dollar(template: str, mapping: Dict[str, str]) -> str:
    # Use string.Template safe substitution for $var and ${var}
    try:
        return string.Template(template).safe_substitute(mapping)
    except Exception:
        # As a fallback, perform a conservative regex-based substitution
        def replace(match):
            key = match.group(1) or match.group(2)
            return mapping.get(key, match.group(0))

        pattern = re.compile(r"\$(?:{([^}]+)}|([A-Za-z_][A-Za-z0-9_]*))")
        return pattern.sub(replace, template)


def create_command(template: str) -> str:
    """
    Formats a base command template by substituting placeholder values.

    - Supports:
      - $var and ${var} placeholders (string.Template syntax)
      - {var} placeholders (str.format style)
      - Environment variable expansion ($VAR, ${VAR}) and ~ user expansion
    - Values are sourced from globally available command template structures (if present)
      and environment variables as a fallback.
    """
    if not isinstance(template, str):
        template = str(template)

    vars_map = _gather_global_template_vars()

    # Step 1: Substitute $var style placeholders using gathered variables
    result = _template_substitute_dollar(template, vars_map)

    # Step 2: Substitute {var} style placeholders using gathered variables
    result = result.format_map(_SafeDict(vars_map))

    # Step 3: Expand environment variables and user home
    result = os.path.expandvars(result)
    result = os.path.expanduser(result)

    return result


def insert_name_into_command(command_template: str, name_string: str) -> str:
    """
    Safely incorporates a provided name into a command template.

    - Supports $name, ${name}, and {name} placeholders.
    - Shell-escapes the name to prevent injection/issues with spaces or special chars.
    - Also performs environment variable (~, $VAR) expansion after substitution.
    - Respects any globally available template variables, with 'name' taking precedence.
    """
    if not isinstance(command_template, str):
        command_template = str(command_template)
    if not isinstance(name_string, str):
        name_string = str(name_string)

    # Shell-escape the user-provided name
    quoted_name = shlex.quote(name_string)

    # Build a mapping where 'name' and common alias keys resolve to the quoted name
    name_map = _with_alias_keys({"name": quoted_name})

    # Merge with global template vars, ensuring the provided name wins
    vars_map = _gather_global_template_vars()
    merged = {**vars_map, **name_map}

    # Substitute $name and ${name}
    result = _template_substitute_dollar(command_template, merged)

    # Substitute {name}
    result = result.format_map(_SafeDict(merged))

    # Expand env vars and user home
    result = os.path.expandvars(result)
    result = os.path.expanduser(result)

    return result


__all__ = ["create_command", "insert_name_into_command"]
