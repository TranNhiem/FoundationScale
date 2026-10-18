"""A small ledger library: accounts, transactions, transfers, and monthly interest.

This module models simple bank accounts with a mutable ledger of transactions.
All arithmetic is done with ``float`` for simplicity (cents are rounded to
2 decimals to keep balances readable).

Public API
----------
* :class:`LedgerError` -- base exception for all ledger failures.
* :class:`InsufficientFunds` -- raised when a withdrawal/transfer would
  overdraw the account.
* :class:`Account` -- an account with an id, a balance and a history.
* :func:`open_account` -- create and register a brand-new account.
* :func:`transfer` -- move money between two accounts atomically.
* :func:`monthly_statement` -- report activity and interest for one month.
"""

from __future__ import annotations

__all__ = [
    "LedgerError",
    "InsufficientFunds",
    "Account",
    "open_account",
    "transfer",
    "monthly_statement",
]

OVERDRAFT_LIMIT = 0.0
TXN_DEPOSIT = "deposit"
TXN_WITHDRAW = "withdraw"
TXN_INTEREST = "interest"

_REGISTRY: dict[str, Account] = {}


class LedgerError(Exception):
    """Base class for all errors raised by this library."""


class InsufficientFunds(LedgerError):
    """Raised when an operation would push the balance below the overdraft limit."""


def _round(amount: float) -> float:
    """Round a monetary amount to two decimals."""
    return round(amount, 2)


class Account:
    """A bank account with a running balance and a signed transaction history.

    Each history entry is a tuple ``(kind, amount, balance_after)``.
    """

    def __init__(self, account_id: str, opening_balance: float = 0.0) -> None:
        if not account_id:
            raise LedgerError("account_id must be non-empty")
        if account_id in _REGISTRY:
            raise LedgerError("duplicate account id: " + account_id)
        if opening_balance < OVERDRAFT_LIMIT:
            raise LedgerError("opening balance below overdraft limit")
        self.id = account_id
        self._balance = _round(opening_balance)
        self._history: list[tuple[str, float, float]] = []
        if opening_balance > 0:
            self._history.append((TXN_DEPOSIT, self._balance, self._balance))
        _REGISTRY[account_id] = self

    @property
    def balance(self) -> float:
        """The current balance (after rounding to cents)."""
        return self._balance

    @property
    def history(self) -> list[tuple[str, float, float]]:
        """A copy of the transaction history; safe for callers to mutate."""
        return list(self._history)

    def deposit(self, amount: float) -> float:
        """Credit ``amount`` to the account and return the new balance."""
        if amount <= 0:
            raise LedgerError("deposit amount must be positive")
        amount = _round(amount)
        self._balance = _round(self._balance + amount)
        self._history.append((TXN_DEPOSIT, amount, self._balance))
        return self._balance

    def withdraw(self, amount: float) -> float:
        """Debit ``amount`` from the account and return the new balance.

        Raises :class:`InsufficientFunds` if the account would go below
        ``OVERDRAFT_LIMIT`` (which is 0 by default: no overdraft allowed).
        """
        if amount <= 0:
            raise LedgerError("withdrawal amount must be positive")
        amount = _round(amount)
        if self._balance - amount < OVERDRAFT_LIMIT:
            raise InsufficientFunds(f"balance {self._balance} cannot cover {amount}")
        self._balance = _round(self._balance - amount)
        self._history.append((TXN_WITHDRAW, -amount, self._balance))
        return self._balance

    def apply_interest(self, rate: float) -> float:
        """Apply one month of simple interest and return the new balance.

        ``rate`` is an annual rate expressed as a decimal (e.g. 0.05 for 5%).
        Interest is rounded to cents and added to the balance.
        """
        if rate <= 0:
            raise LedgerError("interest rate must be positive")
        interest = _round(self._balance * rate / 12.0)
        self._balance = _round(self._balance + interest)
        self._history.append((TXN_INTEREST, interest, self._balance))
        return self._balance

    def history_for(self, kind: str) -> list[tuple[str, float, float]]:
        """Return history entries whose kind matches ``kind``."""
        return [entry for entry in self._history if entry[0] == kind]


def open_account(account_id: str, opening_balance: float = 0.0) -> Account:
    """Create, register and return a new :class:`Account`."""
    return Account(account_id, opening_balance)


def transfer(source: Account, target: Account, amount: float) -> tuple[float, float]:
    """Atomically move ``amount`` from ``source`` to ``target``.

    Returns the pair of new balances ``(source, target)``.
    Raises :class:`LedgerError` if source is target or amount is not positive.
    Raises :class:`InsufficientFunds` if source cannot cover the transfer.
    """
    if source is target:
        raise LedgerError("cannot transfer to the same account")
    if amount <= 0:
        raise LedgerError("transfer amount must be positive")
    amount = _round(amount)
    if source._balance - amount < OVERDRAFT_LIMIT:
        raise InsufficientFunds(f"{source.id} balance {source._balance} cannot cover {amount}")
    source._balance = _round(source._balance - amount)
    target._balance = _round(target._balance + amount)
    source._history.append((TXN_WITHDRAW, -amount, source._balance))
    target._history.append((TXN_DEPOSIT, amount, target._balance))
    return source._balance, target._balance


def monthly_statement(account: Account, rate: float) -> dict[str, float | int | str]:
    """Compute a one-month statement for ``account``.

    Applies monthly interest and returns a dict with balance before, total
    deposits, total withdrawals, interest credited, and final balance.
    """
    before = account.balance
    deposits = sum(a for k, a, _ in account.history_for(TXN_DEPOSIT))
    withdrawals = -sum(a for k, a, _ in account.history_for(TXN_WITHDRAW))
    account.apply_interest(rate)
    entries = account.history_for(TXN_INTEREST)
    interest = sum(a for _, a, _ in entries)
    return {
        "account": account.id,
        "opening": before,
        "deposits": _round(deposits),
        "withdrawals": _round(withdrawals),
        "interest": _round(interest),
        "closing": account.balance,
    }
