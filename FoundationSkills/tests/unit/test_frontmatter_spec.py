"""Unit tests for ``routing_eval._parse_frontmatter``.

The Agent Skills spec (agentskills.io) keeps ``when_to_use`` inside ``metadata`` as a JSON
array string; the legacy top-level list stays readable for one release behind a
DeprecationWarning. Every malformed variant must be REFUSED -- never silently healed.
"""
from __future__ import annotations

import json

import pytest
import yaml

from foundationskills.agent import routing_eval

PHRASES = ["Run an auto research campaign", "Validate my campaign spec", "Close the campaign"]
NAME = "fskills-unit-probe"
DESCRIPTION = "A probe package used to unit-test frontmatter parsing."

BAD_METADATA_WHEN = [
    "{not json",            # not JSON at all
    '"a phrase"',           # JSON string, not a list
    '{"when_to_use": []}',  # JSON object, not a list
    "[]",                   # empty list
    '["", "b"]',            # empty string phrase
    '["a", 2]',             # non-string phrase
    '["a", null]',          # null phrase
    '["a", ["b"]]',         # nested list phrase
    "null",                 # JSON null, not a list
]


def _skill_md(data: dict) -> str:
    """Render *data* as the SKILL.md text ``_parse_frontmatter`` consumes."""
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "---\n\n## Purpose\n"


def _base() -> dict:
    return {"name": NAME, "description": DESCRIPTION}


def test_metadata_json_path_reads_the_phrases():
    data = _base()
    data["metadata"] = {"when_to_use": json.dumps(PHRASES, ensure_ascii=False)}
    parsed = routing_eval._parse_frontmatter(_skill_md(data), "pkg")
    assert parsed == {"name": NAME, "description": DESCRIPTION, "when_to_use": list(PHRASES)}


def test_metadata_when_to_use_keeps_non_ascii_phrases():
    phrases = ["Ferme la campagne — rapport final"]
    data = _base()
    data["metadata"] = {"when_to_use": json.dumps(phrases, ensure_ascii=False)}
    parsed = routing_eval._parse_frontmatter(_skill_md(data), "pkg")
    assert parsed["when_to_use"] == phrases


def test_legacy_top_level_path_still_reads_and_warns():
    data = _base()
    data["when_to_use"] = list(PHRASES)
    with pytest.warns(DeprecationWarning, match="pkg"):
        parsed = routing_eval._parse_frontmatter(_skill_md(data), "pkg")
    assert parsed["when_to_use"] == list(PHRASES)


def test_both_when_to_use_locations_are_refused():
    data = _base()
    data["when_to_use"] = list(PHRASES)
    data["metadata"] = {"when_to_use": json.dumps(PHRASES)}
    with pytest.raises(routing_eval.RoutingEvalRefused, match="both top-level and metadata when_to_use"):
        routing_eval._parse_frontmatter(_skill_md(data), "pkg")


def test_missing_when_to_use_is_refused():
    with pytest.raises(routing_eval.RoutingEvalRefused, match="no when_to_use list of strings"):
        routing_eval._parse_frontmatter(_skill_md(_base()), "pkg")


def test_legacy_top_level_must_still_be_a_list_of_strings():
    for bad in (["a phrase", 7], "not a list"):
        data = _base()
        data["when_to_use"] = bad
        with pytest.warns(DeprecationWarning), \
                pytest.raises(routing_eval.RoutingEvalRefused, match="no when_to_use list of strings"):
            routing_eval._parse_frontmatter(_skill_md(data), "pkg")


@pytest.mark.parametrize("raw", BAD_METADATA_WHEN, ids=range(len(BAD_METADATA_WHEN)))
def test_metadata_when_to_use_must_be_a_json_array_of_non_empty_strings(raw):
    data = _base()
    data["metadata"] = {"when_to_use": raw}
    with pytest.raises(routing_eval.RoutingEvalRefused,
                       match=r"metadata\.when_to_use is not a JSON array of non-empty strings"):
        routing_eval._parse_frontmatter(_skill_md(data), "pkg")


def test_metadata_when_to_use_must_be_a_string_not_a_yaml_list():
    data = _base()
    data["metadata"] = {"when_to_use": list(PHRASES)}
    with pytest.raises(routing_eval.RoutingEvalRefused,
                       match=r"metadata\.when_to_use is not a JSON array of non-empty strings"):
        routing_eval._parse_frontmatter(_skill_md(data), "pkg")


@pytest.mark.parametrize("metadata", [5, "when", ["when"], False, 3.5])
def test_metadata_must_be_a_mapping(metadata):
    data = _base()
    data["metadata"] = metadata
    data["when_to_use"] = list(PHRASES)
    with pytest.raises(routing_eval.RoutingEvalRefused, match="metadata must map str keys to str values"):
        routing_eval._parse_frontmatter(_skill_md(data), "pkg")


def test_metadata_values_must_be_strings():
    data = _base()
    data["metadata"] = {"version": 3, "when_to_use": json.dumps(PHRASES)}
    with pytest.raises(routing_eval.RoutingEvalRefused, match="metadata must map str keys to str values"):
        routing_eval._parse_frontmatter(_skill_md(data), "pkg")


def test_metadata_keys_must_be_strings():
    data = _base()
    data["metadata"] = {1: "one", "when_to_use": json.dumps(PHRASES)}
    with pytest.raises(routing_eval.RoutingEvalRefused, match="metadata must map str keys to str values"):
        routing_eval._parse_frontmatter(_skill_md(data), "pkg")


def test_both_paths_return_the_same_shape_given_the_same_phrases():
    spec = _base()
    spec["metadata"] = {"when_to_use": json.dumps(PHRASES, ensure_ascii=False)}
    legacy = _base()
    legacy["when_to_use"] = list(PHRASES)
    with pytest.warns(DeprecationWarning):
        legacy_parsed = routing_eval._parse_frontmatter(_skill_md(legacy), "pkg")
    spec_parsed = routing_eval._parse_frontmatter(_skill_md(spec), "pkg")
    assert spec_parsed == legacy_parsed
    assert set(spec_parsed) == {"name", "description", "when_to_use"}
    assert all(isinstance(phrase, str) for phrase in spec_parsed["when_to_use"])
