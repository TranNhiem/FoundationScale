import pytest
from ledger import (
    Account,
    InsufficientFunds,
    LedgerError,
    monthly_statement,
    open_account,
    transfer,
)


def test_open_account_zero_balance():
    acct = open_account("a0")
    assert acct.balance == 0.0
    assert acct.id == "a0"


def test_open_account_with_opening_balance():
    acct = open_account("a1", 250.0)
    assert acct.balance == 250.0
    assert acct.history == [("deposit", 250.0, 250.0)]


def test_open_account_negative_opening_balance_raises():
    with pytest.raises(LedgerError):
        open_account("a2", -5.0)


def test_duplicate_account_id_raises():
    open_account("dup1")
    with pytest.raises(LedgerError):
        open_account("dup1")


def test_empty_account_id_raises():
    with pytest.raises(LedgerError):
        Account("")


def test_deposit_adds_to_balance():
    acct = open_account("d1")
    result = acct.deposit(100.0)
    assert result == 100.0
    assert acct.balance == 100.0


def test_deposit_records_history_entry():
    acct = open_account("d2")
    acct.deposit(75.5)
    assert acct.history == [("deposit", 75.5, 75.5)]


def test_negative_deposit_raises():
    acct = open_account("d3")
    with pytest.raises(LedgerError):
        acct.deposit(-10.0)


def test_withdraw_reduces_balance():
    acct = open_account("w1", 200.0)
    result = acct.withdraw(50.0)
    assert result == 150.0
    assert acct.balance == 150.0


def test_withdraw_all_balance_to_zero():
    acct = open_account("w2", 80.0)
    result = acct.withdraw(80.0)
    assert result == 0.0


def test_withdraw_beyond_balance_raises_insufficient():
    acct = open_account("w3", 30.0)
    with pytest.raises(InsufficientFunds):
        acct.withdraw(30.01)


def test_negative_withdraw_raise():
    acct = open_account("w4")
    with pytest.raises(LedgerError):
        acct.withdraw(-1.0)


def test_withdraw_history_has_negative_amount():
    acct = open_account("w5", 50.0)
    acct.withdraw(20.0)
    entries = acct.history_for("withdraw")
    assert entries == [("withdraw", -20.0, 30.0)]


def test_transfer_moves_money():
    src = open_account("t1s", 100.0)
    dst = open_account("t1d", 5.0)
    s_bal, d_bal = transfer(src, dst, 40.0)
    assert s_bal == 60.0
    assert d_bal == 45.0
    assert src.balance == 60.0
    assert dst.balance == 45.0


def test_transfer_insufficient_funds_raises():
    src = open_account("t2s", 25.0)
    dst = open_account("t2d")
    with pytest.raises(InsufficientFunds):
        transfer(src, dst, 30.0)
    assert src.balance == 25.0
    assert dst.balance == 0.0


def test_transfer_to_same_account_raises():
    acct = open_account("t3")
    with pytest.raises(LedgerError):
        transfer(acct, acct, 10.0)


def test_transfer_nonpositive_amount_raises():
    src = open_account("t4s")
    dst = open_account("t4d")
    with pytest.raises(LedgerError):
        transfer(src, dst, 0.0)


def test_apply_interest_adds_to_balance():
    acct = open_account("i1", 1200.0)
    result = acct.apply_interest(0.12)
    # 1200 * 0.12 / 12 = 12.0
    assert result == 1212.0
    assert acct.balance == 1212.0


def test_apply_interest_records_history():
    acct = open_account("i2", 600.0)
    acct.apply_interest(0.06)
    # 600 * 0.06 / 12 = 3.0
    entries = acct.history_for("interest")
    assert entries == [("interest", 3.0, 603.0)]


def test_apply_interest_on_zero_balance():
    acct = open_account("i3")
    result = acct.apply_interest(0.05)
    assert result == 0.0


def test_apply_interest_nonpositive_rate_raises():
    acct = open_account("i4")
    with pytest.raises(LedgerError):
        acct.apply_interest(0.0)


def test_history_returns_a_copy():
    acct = open_account("h1", 10.0)
    snapshot = acct.history
    snapshot.clear()
    assert len(acct.history) == 1


def test_monthly_statement_totals():
    acct = open_account("m1", 100.0)
    acct.deposit(200.0)
    acct.withdraw(50.0)
    stmt = monthly_statement(acct, 0.0)
    assert stmt["opening"] == 300.0
    assert stmt["deposits"] == 300.0
    # opening 100 is a deposit too -> deposits = 100 + 200 = 300
    assert stmt["withdrawals"] == 50.0
    assert stmt["closing"] == 250.0


def test_monthly_statement_includes_interest():
    acct = open_account("m2", 1000.0)
    stmt = monthly_statement(acct, 0.06)
    # interest = 1000 * 0.06 / 12 = 5.0
    assert stmt["interest"] == 5.0
    assert stmt["closing"] == 1005.0
