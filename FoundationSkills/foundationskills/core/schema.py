
"""A small stdlib validator for the JSON-Schema subset used by skill artifacts."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any


class SchemaError(ValueError):
    """Raised when an instance or schema violates the supported subset."""


def _path_join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _type_matches(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return _is_number(value)
    if type_name == "integer":
        return _is_integer(value)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    raise ValueError(f"unsupported JSON schema type: {type_name!r}")


def _type_names(schema: dict[str, Any]) -> list[str]:
    raw = schema.get("type")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return list(raw)
    raise ValueError(f"invalid 'type' declaration: {raw!r}")


def _fmt_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if _is_integer(value):
        return "integer"
    if _is_number(value):
        return "number"
    return type(value).__name__


def _resolve_ref(root_schema: dict[str, Any], ref: str) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise SchemaError(f"unsupported $ref (only same-root #/$defs/<name>): {ref!r}")
    name = ref[len(prefix) :]
    defs = root_schema.get("$defs", {})
    target = defs.get(name)
    if not isinstance(target, dict):
        raise SchemaError(f"unresolved $ref: {ref!r}")
    return target


def _validate(instance: Any, schema: dict[str, Any], root_schema: dict[str, Any], path: str, ref_stack: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        raise ValueError(f"schema at {path} must be an object")

    ref = schema.get("$ref")
    if isinstance(ref, str):
        marker = f"{ref}@{path}"
        if marker in ref_stack:
            raise SchemaError(f"recursive $ref detected: {ref!r}")
        return _validate(instance, _resolve_ref(root_schema, ref), root_schema, path, ref_stack + (marker,))

    type_names = _type_names(schema)
    if type_names and not any(_type_matches(instance, t) for t in type_names):
        want = ", ".join(type_names)
        errors.append(f"{path}: expected type {want}; got {_fmt_type(instance)}")
        return errors

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}; got {instance!r}")
    if "enum" in schema:
        options = schema["enum"]
        if not isinstance(options, list):
            raise ValueError(f"enum at {path} must be an array")
        if instance not in options:
            errors.append(f"{path}: value {instance!r} not in enum {options!r}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            raise ValueError(f"required at {path} must be an array of strings")
        for key in required:
            if key not in instance:
                errors.append(f"{_path_join(path, key)}: missing required property")
        props = schema.get("properties", {})
        if not isinstance(props, dict):
            raise ValueError(f"properties at {path} must be an object")
        for key, subschema in props.items():
            if key in instance:
                if not isinstance(subschema, dict):
                    raise ValueError(f"property schema {key!r} at {path} must be an object")
                errors.extend(_validate(instance[key], subschema, root_schema, _path_join(path, key), ref_stack))
        additional = schema.get("additionalProperties", True)
        extras = [k for k in instance if k not in props]
        if additional is False:
            for key in extras:
                errors.append(f"{_path_join(path, key)}: additional property not allowed")
        elif isinstance(additional, dict):
            for key in extras:
                errors.extend(_validate(instance[key], additional, root_schema, _path_join(path, key), ref_stack))
        elif additional is not True:
            raise ValueError(f"additionalProperties at {path} must be boolean or schema")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            errors.append(f"{path}: expected at least {schema['minItems']} items; got {len(instance)}")
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            errors.append(f"{path}: expected at most {schema['maxItems']} items; got {len(instance)}")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(instance):
                errors.extend(_validate(item, items, root_schema, _path_join(path, i), ref_stack))
        elif isinstance(items, list):
            for i, sub in enumerate(items[: len(instance)]):
                if isinstance(sub, dict):
                    errors.extend(_validate(instance[i], sub, root_schema, _path_join(path, i), ref_stack))
        elif items is not None:
            raise ValueError(f"items at {path} must be a schema")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            errors.append(f"{path}: expected minLength {schema['minLength']}; got {len(instance)}")
        if "pattern" in schema:
            pattern = schema["pattern"]
            try:
                matched = re.search(str(pattern), instance)
            except re.error as exc:
                raise ValueError(f"invalid pattern at {path}: {exc}") from exc
            if not matched:
                errors.append(f"{path}: string does not match pattern {pattern!r}")

    if _is_number(instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: expected >= {schema['minimum']}; got {instance}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: expected <= {schema['maximum']}; got {instance}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: expected > {schema['exclusiveMinimum']}; got {instance}")

    if "anyOf" in schema:
        branches = schema["anyOf"]
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"anyOf at {path} must be a non-empty array")
        collected = [_validate(instance, branch, root_schema, path, ref_stack) for branch in branches]
        if not any(not branch_errors for branch_errors in collected):
            first = collected[0][0] if collected and collected[0] else "no branch matched"
            errors.append(f"{path}: anyOf failed ({first})")

    if "oneOf" in schema:
        branches = schema["oneOf"]
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"oneOf at {path} must be a non-empty array")
        collected = [_validate(instance, branch, root_schema, path, ref_stack) for branch in branches]
        valid_count = sum(1 for branch_errors in collected if not branch_errors)
        if valid_count != 1:
            errors.append(f"{path}: oneOf requires exactly one matching branch; got {valid_count}")

    return errors


def validate(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Validate an instance. Returns error strings ('<json-path>: message'); [] means valid."""
    return _validate(instance, schema, schema, "$", ())


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    """Load foundationskills/schemas/<name>.json, cached by name."""
    if "/" in name:
        rel = resources.files("foundationskills").joinpath("schemas", *name.split("/")).with_suffix(".json")
    else:
        rel = resources.files("foundationskills").joinpath("schemas", f"{name}.json")
    try:
        with rel.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except FileNotFoundError as exc:
        raise SchemaError(f"schema not found: {name}") from exc
    if not isinstance(loaded, dict):
        raise SchemaError(f"schema {name} is not a JSON object")
    return loaded


def assert_valid(instance: Any, schema: dict[str, Any], what: str) -> None:
    """Raise SchemaError listing all validation errors for <what>."""
    errors = validate(instance, schema)
    if errors:
        joined = "; ".join(errors)
        raise SchemaError(f"{what} failed schema validation ({len(errors)} error(s)): {joined}")
