from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from foundationskills.core import activation_governor as ag
from foundationskills.core import ledger


NOW = datetime(2030, 1, 15, 12, 0, tzinfo=timezone.utc)
FUTURE = "2030-01-16T00:00:00+00:00"


def make_record(**changes):
    values = {
        "trigger_id": "trigger-1",
        "skill": "example-skill",
        "owner": "test-owner",
        "scope": "test",
        "condition": "an operator-approved condition",
        "expires_at": FUTURE,
        "budget_cap": 3,
        "daily_cap": 3,
        "action_class": "proposing",
        "confirm_policy": "spec-only",
        "provenance": "unit-test",
        "quiet_hours": None,
    }
    values.update(changes)
    return ag.TriggerRecord(**values)


def trigger_rows(directory):
    path = directory / "ledger.jsonl"
    if not path.exists():
        return []
    return [row for row in ledger.read(path) if row.get("kind") == "trigger"]


def call(governor, trigger_id="trigger-1", source="test-source", payload=None,
         now=NOW, condition_holds=True):
    return governor.on_event(
        trigger_id,
        source,
        {"value": 1} if payload is None else payload,
        now,
        condition_holds,
    )


def test_payload_key_is_stable_canonical_and_16_hex():
    first = ag.payload_key("source", {"b": 2, "a": {"d": 4, "c": 3}})
    reordered = ag.payload_key("source", {"a": {"c": 3, "d": 4}, "b": 2})

    assert first == reordered
    assert len(first) == 16
    int(first, 16)
    assert first != ag.payload_key("other-source", {"a": {"c": 3, "d": 4}, "b": 2})
    assert first != ag.payload_key("source", {"a": {"c": 3, "d": 4}, "b": 3})


@pytest.mark.parametrize("field", ["trigger_id", "owner", "skill", "provenance"])
def test_validate_record_refuses_empty_required_fields(field):
    with pytest.raises(ag.GovernorRefusal, match=field):
        ag.validate_record(make_record(**{field: ""}))


@pytest.mark.parametrize(
    "expires_at",
    [
        pytest.param("not-an-iso-date", id="unparsable"),
        pytest.param("", id="empty"),
        pytest.param("2030-01-16T00:00:00", id="naive"),
    ],
)
def test_validate_record_refuses_bad_expires_at(expires_at):
    with pytest.raises(ag.GovernorRefusal, match="expires_at"):
        ag.validate_record(make_record(expires_at=expires_at))


@pytest.mark.parametrize("field", ["budget_cap", "daily_cap"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param("1", id="string"),
        pytest.param(None, id="none"),
        pytest.param(True, id="true-bool"),
        pytest.param(False, id="false-bool"),
    ],
)
def test_validate_record_refuses_bad_caps_including_bools(field, value):
    with pytest.raises(ag.GovernorRefusal, match=field):
        ag.validate_record(make_record(**{field: value}))


def test_validate_record_refuses_acting_action_class():
    with pytest.raises(ag.GovernorRefusal) as excinfo:
        ag.validate_record(make_record(action_class="acting"))

    assert str(excinfo.value) == (
        "action_class: only 'proposing' triggers are allowed; "
        "acting triggers are refused"
    )


def test_validate_record_refuses_unknown_confirm_policy():
    with pytest.raises(ag.GovernorRefusal, match="confirm_policy"):
        ag.validate_record(make_record(confirm_policy="automatic"))


@pytest.mark.parametrize(
    "quiet_hours",
    [
        (7,),
        (7, 8, 9),
        (-1, 8),
        (7, 24),
        (7.5, 8),
        (True, 8),
        [7, 8],
        "7-8",
    ],
)
def test_validate_record_refuses_bad_quiet_hours(quiet_hours):
    with pytest.raises(ag.GovernorRefusal, match="quiet_hours"):
        ag.validate_record(make_record(quiet_hours=quiet_hours))


def test_governor_refuses_duplicate_trigger_ids(tmp_path):
    first = make_record()
    second = replace(first, skill="another-skill")

    with pytest.raises(ag.GovernorRefusal, match="trigger_id"):
        ag.Governor([first, second], tmp_path)


def test_propose_returns_spec_requiring_hash_and_confirm(tmp_path):
    governor = ag.Governor([make_record(confirm_policy="ask")], tmp_path)

    decision = call(governor)

    assert set(decision) == {"trigger_id", "decision", "reason", "proposal"}
    assert decision["trigger_id"] == "trigger-1"
    assert decision["decision"] == "propose"
    assert decision["proposal"] == {
        "skill": "example-skill",
        "emit": "fs_launch_spec",
        "confirm_policy": "ask",
        "requires": "fskills hash + fskills launch --confirm",
    }


def test_duplicate_is_an_idempotent_noop_and_does_not_consume_caps(tmp_path):
    record = make_record(budget_cap=2, daily_cap=2)
    governor = ag.Governor([record], tmp_path)
    payload = {"same": "event"}

    assert call(governor, payload=payload)["decision"] == "propose"
    duplicate = call(governor, payload=payload)
    assert duplicate["decision"] == "duplicate"
    assert duplicate["proposal"] is None

    # The duplicate did not spend the second permitted fire.
    assert call(governor, payload={"different": "event"})["decision"] == "propose"
    assert [row["verdict"] for row in trigger_rows(tmp_path)] == [
        "propose",
        "duplicate",
        "propose",
    ]


@pytest.mark.parametrize(
    "records,global_cap,calls",
    [
        pytest.param(
            [],
            20,
            [("missing", "src", {"x": 1}, NOW, True, "unknown")],
            id="unknown",
        ),
        pytest.param(
            [make_record()],
            20,
            [("trigger-1", "src", {"x": 1}, NOW, True, "propose")],
            id="propose",
        ),
        pytest.param(
            [make_record()],
            20,
            [
                ("trigger-1", "src", {"x": 1}, NOW, True, "propose"),
                ("trigger-1", "src", {"x": 1}, NOW, True, "duplicate"),
            ],
            id="duplicate",
        ),
        pytest.param(
            [make_record(expires_at="2030-01-15T11:59:59+00:00")],
            20,
            [("trigger-1", "src", {"x": 1}, NOW, True, "expired")],
            id="expired",
        ),
        pytest.param(
            [make_record()],
            20,
            [("trigger-1", "src", {"x": 1}, NOW, False, "condition_false")],
            id="condition-false",
        ),
        pytest.param(
            [make_record(quiet_hours=(9, 17))],
            20,
            [("trigger-1", "src", {"x": 1}, NOW, True, "quiet_hours")],
            id="quiet-hours",
        ),
        pytest.param(
            [make_record(budget_cap=1, daily_cap=1)],
            1,
            [
                ("trigger-1", "src", {"x": 1}, NOW, True, "propose"),
                ("trigger-1", "src", {"x": 2}, NOW, True, "budget_cap"),
            ],
            id="budget-cap-before-daily-and-global",
        ),
        pytest.param(
            [make_record(budget_cap=2, daily_cap=1)],
            1,
            [
                ("trigger-1", "src", {"x": 1}, NOW, True, "propose"),
                ("trigger-1", "src", {"x": 2}, NOW, True, "daily_cap"),
            ],
            id="daily-cap-before-global",
        ),
        pytest.param(
            [
                make_record(trigger_id="trigger-1", budget_cap=2, daily_cap=2),
                make_record(trigger_id="trigger-2", budget_cap=2, daily_cap=2),
            ],
            1,
            [
                ("trigger-1", "src", {"x": 1}, NOW, True, "propose"),
                ("trigger-2", "src", {"x": 2}, NOW, True, "global_cap"),
            ],
            id="global-cap",
        ),
    ],
)
def test_every_decision_appends_exactly_one_trigger_ledger_row(
    tmp_path, records, global_cap, calls
):
    governor = ag.Governor(records, tmp_path, global_daily_cap=global_cap)
    skills = {record.trigger_id: record.skill for record in records}

    for index, (trigger_id, source, payload, now, holds, expected) in enumerate(calls):
        before = len(trigger_rows(tmp_path))
        decision = governor.on_event(trigger_id, source, payload, now, holds)
        rows = trigger_rows(tmp_path)

        assert decision["decision"] == expected
        assert set(decision) == {"trigger_id", "decision", "reason", "proposal"}
        assert (decision["proposal"] is not None) == (expected == "propose")
        assert len(rows) == before + 1
        assert len(rows) == index + 1

        row = rows[-1]
        expected_key = ag.payload_key(source, payload)
        assert row["kind"] == "trigger"
        assert row["skill_name"] == skills.get(trigger_id, "unknown")
        assert row["verdict"] == expected
        assert row["trigger_id"] == trigger_id
        assert row["payload_key"] == expected_key
        assert row["decision"] == expected
        assert row["reason"] == decision["reason"]
        assert row["day"] == now.astimezone(timezone.utc).date().isoformat()
        assert row["exit_code"] is None
        assert row["wall_clock_ms"] is None


def test_condition_false_and_quiet_skips_can_be_redelivered(tmp_path):
    condition_governor = ag.Governor(
        [make_record(budget_cap=1, daily_cap=1)], tmp_path / "condition"
    )
    payload = {"delivery": "same-payload"}

    assert call(
        condition_governor, payload=payload, condition_holds=False
    )["decision"] == "condition_false"
    assert call(
        condition_governor, payload=payload, condition_holds=True
    )["decision"] == "propose"

    quiet_governor = ag.Governor(
        [make_record(quiet_hours=(9, 17), budget_cap=1, daily_cap=1)],
        tmp_path / "quiet",
    )
    assert call(quiet_governor, payload=payload)["decision"] == "quiet_hours"
    assert call(
        quiet_governor, payload=payload, now=NOW.replace(hour=17)
    )["decision"] == "propose"


def test_quiet_hours_window_wrapping_midnight(tmp_path):
    governor = ag.Governor(
        [make_record(quiet_hours=(22, 6), expires_at="2030-01-17T00:00:00+00:00")],
        tmp_path,
    )
    payload = {"delivery": "wrapped-window"}

    quiet_times = [
        datetime(2030, 1, 15, 22, 0, tzinfo=timezone.utc),
        datetime(2030, 1, 15, 23, 30, tzinfo=timezone.utc),
        datetime(2030, 1, 16, 2, 0, tzinfo=timezone.utc),
    ]
    for now in quiet_times:
        assert call(governor, payload=payload, now=now)["decision"] == "quiet_hours"

    # The end is exclusive, and none of the quiet-hours skips poisoned dedup.
    assert call(
        governor,
        payload=payload,
        now=datetime(2030, 1, 16, 6, 0, tzinfo=timezone.utc),
    )["decision"] == "propose"


@pytest.mark.parametrize(
    "overrides,expected",
    [
        pytest.param(
            {"budget_cap": 1, "daily_cap": 5}, "budget_cap", id="lifetime-cap"
        ),
        pytest.param(
            {"budget_cap": 5, "daily_cap": 1}, "daily_cap", id="daily-cap"
        ),
    ],
)
def test_per_trigger_caps_and_dedup_survive_restart(tmp_path, overrides, expected):
    record = make_record(**overrides)
    payload = {"delivery": "persisted"}
    first = ag.Governor([record], tmp_path)
    assert call(first, payload=payload)["decision"] == "propose"

    restarted = ag.Governor([record], tmp_path)
    assert call(restarted, payload=payload)["decision"] == "duplicate"
    assert call(restarted, payload={"delivery": "new"})["decision"] == expected
    assert len(trigger_rows(tmp_path)) == 3


def test_global_daily_cap_survives_restart(tmp_path):
    records = [
        make_record(trigger_id="trigger-1"),
        make_record(trigger_id="trigger-2"),
    ]
    first = ag.Governor(records, tmp_path, global_daily_cap=1)
    assert call(first, "trigger-1")["decision"] == "propose"

    restarted = ag.Governor(records, tmp_path, global_daily_cap=1)
    assert call(restarted, "trigger-2")["decision"] == "global_cap"
    assert [row["verdict"] for row in trigger_rows(tmp_path)] == [
        "propose",
        "global_cap",
    ]


def test_naive_now_is_refused_without_writing_a_decision(tmp_path):
    governor = ag.Governor([make_record()], tmp_path)
    naive = datetime(2030, 1, 15, 12, 0)

    with pytest.raises(ag.GovernorRefusal, match="now"):
        call(governor, now=naive)
    with pytest.raises(ag.GovernorRefusal, match="now"):
        governor.sweep_expected(naive)

    assert trigger_rows(tmp_path) == []


def test_sweep_expected_lists_unfired_live_trigger_and_omits_expired(tmp_path):
    live = make_record(trigger_id="live")
    expired = make_record(
        trigger_id="expired",
        expires_at="2030-01-15T11:00:00+00:00",
    )
    governor = ag.Governor([live, expired], tmp_path)

    missing = governor.sweep_expected(NOW)

    assert len(missing) == 1
    assert missing[0]["trigger_id"] == "live"
    assert set(missing[0]) == {"trigger_id", "missing_since_hours"}
    assert isinstance(missing[0]["missing_since_hours"], (int, float))
    assert all(entry["trigger_id"] != "expired" for entry in missing)
    assert trigger_rows(tmp_path) == []


def test_proposal_is_withheld_when_the_ledger_cannot_be_written(tmp_path, monkeypatch):
    from foundationskills.core import activation_governor as ag

    rec = ag.TriggerRecord(trigger_id="t1", skill="s", owner="o", scope="sc", condition="c",
                           expires_at="2099-01-01T00:00:00+00:00", budget_cap=5, daily_cap=5,
                           action_class="proposing", confirm_policy="ask", provenance="p", quiet_hours=None)
    gov = ag.Governor([rec], tmp_path)
    monkeypatch.setattr(ag.ledger, "append", lambda directory, row: False)
    now = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
    out = gov.on_event("t1", "src", {"x": 1}, now, True)
    assert out["decision"] == "ledger_unwritable" and out["proposal"] is None
    monkeypatch.undo()
    assert gov.on_event("t1", "src", {"x": 1}, now, True)["decision"] == "propose"  # nothing was consumed


@pytest.mark.parametrize("junk", [1, "yes", ["x"], {"a": 1}])
def test_truthy_non_bool_condition_never_opens_the_gate(tmp_path, junk):
    governor = ag.Governor([make_record(budget_cap=1, daily_cap=1)], tmp_path / "junk")
    assert call(governor, condition_holds=junk)["decision"] == "condition_false"
