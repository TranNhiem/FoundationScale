from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralReport:
    """The result of measuring one object against one typing Protocol.

    ``checked`` is the denominator: every member on which this report made a
    claim. A protocol member outside ``checked`` was not silently declared
    compatible. ``mismatches`` is empty only when every checked member matched.

    WHAT IS CLAIMED: ``ok`` answers whether the measured structural surface is
    compatible. WHAT IS NOT CLAIMED: semantic compatibility, runtime behavior,
    or anything about members excluded from the protocol denominator.
    """

    protocol: str  # protocol class __name__
    subject: str  # type(obj).__name__
    checked: tuple[str, ...]  # sorted member names actually compared -- the DENOMINATOR
    mismatches: tuple[str, ...]  # sorted human-readable reasons; empty == compatible

    @property
    def ok(self) -> bool:
        """Return whether the checked surface had no measured mismatches."""

        return not self.mismatches


class StructuralRefusal(Exception):
    """Raised when a structural question cannot be answered.

    A refusal is never a finding of compatibility and is intentionally not a
    mismatch. It means the checker could not establish a meaningful denominator
    or could not inspect a member that was nominally inside that denominator.
    """


def structural_report(
    obj: object,
    protocol: type[object],
    *,
    origin: str = "<subject>",
) -> StructuralReport:
    """Measure signature compatibility between ``obj`` and ``protocol``.

    The measurement direction is substitutability: the subject must provide
    every protocol-visible call parameter under the same name and kind, and
    must not introduce an additional required parameter. A subject may add an
    optional parameter, because a defaulted extension accepts every call the
    protocol permits. A ``**kwargs`` catch-all is likewise treated as an
    accepting surface for protocol-named keyword-capable parameters, because
    delegated adapters may forward declared names without restating them.

    ``self`` is ignored on both sides. Parameter and return annotations are
    deliberately not compared: static type checking, especially mypy, is the
    oracle for type equality. This helper measures arity, names, parameter
    kinds, requiredness, and explicit catch-alls only.

    Properties declared on the protocol are checked for presence and still
    appear in ``checked``. A property has no call signature to compare, and
    this helper does not invent one.

    WHAT IS CLAIMED: every name in the returned ``checked`` tuple was compared
    according to the rules above, and a report is returned for all detected
    partial compatibility outcomes. WHAT IS NOT CLAIMED:
    ``runtime_checkable``/``isinstance`` compatibility, annotation or return
    type compatibility, default-value equality, semantic meaning of parameter
    order, property value compatibility, callable behavior, or compatibility
    of private and dunder names. Those excluded names are outside the measured
    denominator.

    Args:
        obj: Object providing the candidate members.
        protocol: A ``typing.Protocol`` class whose surface defines the
            denominator. It need not be decorated with ``runtime_checkable``;
            this helper inspects signatures directly rather than relying on
            method-presence checks.
        origin: Operator-facing location included in refusal and mismatch
            messages.

    Returns:
        A report containing only the measured scope and any concrete mismatch
        reasons.

    Raises:
        StructuralRefusal: If ``protocol`` is not a ``typing.Protocol`` class,
            if it declares zero public callable or property members, if a
            declared callable signature cannot be introspected, or if attribute
            access prevents measurement. A refusal means the question cannot be
            answered; it never means "compatible".
    """

    if not _is_typing_protocol(protocol):
        raise StructuralRefusal(
            f"{origin}: protocol argument is not a typing.Protocol class; got {protocol!r}"
        )

    members = _collect_protocol_members(protocol, origin)
    if not members:
        raise StructuralRefusal(
            f"{origin}: protocol '{protocol.__name__}' declares zero "
            "checkable public members; an empty denominator cannot be "
            "reported as compatible"
        )

    checked = tuple(sorted(members))
    mismatches: list[str] = []

    for name in checked:
        member = members[name]
        if member.property_member:
            if not _subject_has_member(obj, name, origin):
                mismatches.append(
                    f"member '{name}' at {origin}: protocol declares "
                    "@property (presence-only); subject offers no attribute"
                )
            continue

        # Property members are the only members with no signature. A callable
        # member reaching this branch has already refused if its OWN protocol
        # declaration could not be inspected.
        protocol_signature = member.signature
        if protocol_signature is None:
            raise StructuralRefusal(
                f"{origin}: cannot introspect protocol member '{name}' "
                "because its callable signature is unavailable"
            )

        subject_signature, absence = _subject_callable_signature(obj, name, origin)
        if absence is not None:
            # Both arms below are DECIDABLE, so both are mismatches and
            # neither is a refusal. Collapsing "the subject offers an int
            # where the protocol declares a method" into CANNOT-MEASURE
            # would launder a red into an abstention -- the caller asked a
            # question that was answered, and the answer is no.
            mismatches.append(
                f"member '{name}' at {origin}: protocol declares callable "
                f"{name}{protocol_signature}; {absence}"
            )
            continue
        if subject_signature is None:  # pragma: no cover - defensive
            raise StructuralRefusal(
                f"{origin}: member '{name}' produced neither a signature nor "
                "a reason; refusing rather than reporting a compatibility "
                "this function did not establish"
            )

        mismatches.extend(
            _compare_callable_signatures(
                name=name,
                protocol_signature=protocol_signature,
                subject_signature=subject_signature,
                origin=origin,
            )
        )

    return StructuralReport(
        protocol=protocol.__name__,
        subject=type(obj).__name__,
        checked=checked,
        mismatches=tuple(sorted(mismatches)),
    )


@dataclass(frozen=True)
class _ProtocolMember:
    """One member selected for the structural denominator."""

    name: str
    signature: inspect.Signature | None
    property_member: bool


_REGULAR_PARAMETER_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)

# The two kinds that accept a POSITIONAL argument. Either may stand at a
# positional-only slot: POSITIONAL_OR_KEYWORD is merely wider than the contract
# requires, and widening what an implementation accepts is never a mismatch.
_POSITIONALLY_ACCEPTING = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _is_typing_protocol(candidate: object) -> bool:
    """Return whether typing's own metaclass marked a class as a Protocol."""

    # Python 3.10 exposes no public is_protocol() predicate. The private
    # marker is assigned by typing's protocol machinery itself; using it here
    # avoids treating an ordinary ABC, base class, or manually shaped object
    # as a contract.
    return isinstance(candidate, type) and bool(getattr(candidate, "_is_protocol", False))


def _protocol_attribute_names(protocol: type[object], origin: str) -> frozenset[str]:
    """Return the EXACT member set ``isinstance`` checks for this protocol.

    The denominator is not this module's invention. ``isinstance`` against a
    ``@runtime_checkable`` Protocol tests presence of one specific set of
    names, and typing computes that set; borrowing it is what makes this
    function the SAME question asked more strictly, rather than a second
    question with its own scope. Deriving the set independently -- by walking
    ``__mro__`` and skipping ``_``-prefixed names, say -- silently drops
    ``__call__``, which is the central member of every objective protocol in
    this plane, and a helper whose denominator omits the member under test
    reports ``ok`` over a subject that cannot be called at all.

    ``__protocol_attrs__`` is public from 3.12; ``typing._get_protocol_attrs``
    is the same computation on 3.10/3.11. If a future runtime exposes
    neither, this REFUSES: guessing the denominator is how the hole above got
    in, and an unmeasurable scope is not an empty one.
    """

    attrs = getattr(protocol, "__protocol_attrs__", None)
    if attrs is None:
        legacy = getattr(typing, "_get_protocol_attrs", None)
        if legacy is None:
            raise StructuralRefusal(
                f"{origin}: this runtime exposes neither "
                f"'{protocol.__name__}.__protocol_attrs__' nor "
                "typing._get_protocol_attrs, so the member set isinstance "
                "actually checks cannot be read; refusing rather than "
                "deriving a denominator that may omit members"
            )
        attrs = legacy(protocol)
    return frozenset(str(name) for name in attrs)


def _collect_protocol_members(
    protocol: type[object],
    origin: str,
) -> dict[str, _ProtocolMember]:
    """Collect the protocol's callables and properties, subclass first."""

    members: dict[str, _ProtocolMember] = {}
    wanted = _protocol_attribute_names(protocol, origin)

    # User-defined Protocol bases remain part of the inherited contract.
    # typing.Protocol and object are machinery endpoints whose members are
    # explicitly outside this denominator.
    for base in protocol.__mro__:
        if base in (typing.Protocol, object):
            continue

        for name, declared in vars(base).items():
            if name not in wanted or name in members:
                continue

            if isinstance(declared, property):
                members[name] = _ProtocolMember(
                    name=name,
                    signature=None,
                    property_member=True,
                )
                continue

            signature_target: object = declared
            if isinstance(declared, (staticmethod, classmethod)):
                signature_target = declared.__func__

            if not callable(signature_target):
                # A class-body value that is not callable declares an
                # attribute, not a method, so there is no signature to
                # compare -- the same situation as a property. It is recorded
                # as presence-only rather than skipped: `continue` here would
                # drop the name from `checked` while isinstance still checks
                # it, which is the denominator hole this function was just
                # repaired for, wearing a different hat.
                members[name] = _ProtocolMember(
                    name=name,
                    signature=None,
                    property_member=True,
                )
                continue

            try:
                signature = inspect.signature(signature_target)
            except (ValueError, TypeError) as exc:
                raise StructuralRefusal(
                    f"{origin}: cannot introspect protocol member '{name}': "
                    f"inspect.signature raised {type(exc).__name__}: {exc}"
                ) from exc

            members[name] = _ProtocolMember(
                name=name,
                signature=_drop_self(signature),
                property_member=False,
            )

    # An annotation-only member (`tokens: int`, with no assignment) is counted
    # by __protocol_attrs__ and checked by isinstance, but it appears in no
    # base's vars(), so the walk above cannot see it. Reconciling the two sets
    # here is what keeps `checked` EQUAL to the set isinstance consults rather
    # than a subset of it: the whole value of borrowing typing's denominator is
    # lost the moment this function quietly returns fewer names than it was
    # given. An annotation declares an attribute, so it is presence-only.
    for name in sorted(wanted - members.keys()):
        members[name] = _ProtocolMember(
            name=name,
            signature=None,
            property_member=True,
        )

    return members


def _drop_self(signature: inspect.Signature) -> inspect.Signature:
    """Remove an explicit leading ``self`` without touching other names."""

    # A bound implementation method normally has no receiver parameter left in
    # its inspected signature, while an unbound function stored on an object
    # often does. Removing exactly a leading receiver named "self" normalizes
    # those presentation differences without arbitrarily deleting user data.
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    return signature.replace(parameters=parameters)


def _subject_has_member(obj: object, name: str, origin: str) -> bool:
    """Check membership without evaluating a present property itself."""

    try:
        inspect.getattr_static(obj, name)
    except AttributeError:
        # Support objects that expose a genuinely dynamic member through
        # __getattr__. A static absence followed by dynamic AttributeError is
        # a normal missing-member mismatch.
        try:
            getattr(obj, name)
        except AttributeError:
            return False
        except Exception as exc:
            raise StructuralRefusal(
                f"{origin}: cannot inspect subject member '{name}': "
                f"attribute access raised {type(exc).__name__}: {exc}"
            ) from exc
    return True


def _subject_callable_signature(
    obj: object,
    name: str,
    origin: str,
) -> tuple[inspect.Signature | None, str | None]:
    """Return ``(signature, None)``, or ``(None, reason)`` for a mismatch.

    The two-slot return keeps three outcomes distinguishable that a bare
    ``Signature | None`` collapses into two: the member fits and can be
    compared; the member is DECIDABLY wrong (absent, or present but not
    callable at all); and the member is callable but un-introspectable,
    which is the only CANNOT-MEASURE state and the only one that raises.
    """

    try:
        member = getattr(obj, name)
    except AttributeError:
        return None, "subject offers no attribute"
    except Exception as exc:
        raise StructuralRefusal(
            f"{origin}: cannot inspect subject member '{name}': "
            f"attribute access raised {type(exc).__name__}: {exc}"
        ) from exc

    if not callable(member):
        # A protocol that declares `def capabilities(self) -> X: ...` is
        # called as `obj.capabilities()`. A subject supplying a property (or
        # a plain attribute) of the right NAME satisfies isinstance and then
        # raises `TypeError: 'int' object is not callable` at the call site.
        # That is precisely the defect this module exists to catch, so it is
        # named as a mismatch here rather than passed to inspect.signature,
        # whose TypeError would otherwise read as "could not measure".
        return None, (
            f"subject offers a non-callable {type(member).__name__} "
            f"(a protocol-declared method is invoked, not read)"
        )

    try:
        signature = inspect.signature(member)
    except (ValueError, TypeError) as exc:
        raise StructuralRefusal(
            f"{origin}: cannot introspect subject member '{name}': "
            f"inspect.signature raised {type(exc).__name__}: {exc}"
        ) from exc

    return _drop_self(signature), None


def _is_required(parameter: inspect.Parameter) -> bool:
    """Return whether a non-variadic parameter must be supplied."""

    return (
        parameter.kind in _REGULAR_PARAMETER_KINDS and parameter.default is inspect.Parameter.empty
    )


def _missing_parameter_offering(
    parameter: inspect.Parameter,
    signature: inspect.Signature,
    member_name: str,
) -> str:
    """Describe the subject's failure to accept one protocol parameter."""

    offering = (
        f"signature {member_name}{signature} has no parameter named "
        f"'{parameter.name}' with kind {parameter.kind.name}"
    )

    if parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        offering += " and no VAR_KEYWORD catch-all"
    elif parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        offering += " and no VAR_POSITIONAL catch-all"

    return offering


def _compare_callable_signatures(
    *,
    name: str,
    protocol_signature: inspect.Signature,
    subject_signature: inspect.Signature,
    origin: str,
) -> list[str]:
    """Compare arity, names, kinds, and requiredness, excluding annotations."""

    mismatches: list[str] = []
    protocol_parameters = protocol_signature.parameters
    subject_parameters = subject_signature.parameters

    # A POSITIONAL_ONLY parameter is matched by POSITION, never by name. The
    # `/` marker is the protocol author's explicit statement that the name is
    # NOT part of the contract -- no caller may write it, so an implementation
    # is free to choose its own. Looking such a parameter up by name in the
    # subject (as every other kind is, correctly) would refuse a conforming
    # implementation for spelling an unspeakable name differently, which is a
    # false positive from the one module in this plane whose entire job is
    # comparing signatures. Both POSITIONAL_ONLY and POSITIONAL_OR_KEYWORD
    # accept a positional call, so either may stand at a positional-only slot;
    # the second is merely wider than the contract requires.
    protocol_positional = [
        parameter
        for parameter in protocol_parameters.values()
        if parameter.kind in _POSITIONALLY_ACCEPTING
    ]
    subject_positional = [
        parameter
        for parameter in subject_parameters.values()
        if parameter.kind in _POSITIONALLY_ACCEPTING
    ]
    protocol_positional_index = {
        parameter.name: index for index, parameter in enumerate(protocol_positional)
    }
    # Subject parameters consumed positionally, so the reverse sweep below does
    # not then re-report them as required extras the protocol never declared.
    matched_positionally: set[str] = set()

    var_keyword = next(
        (
            parameter
            for parameter in subject_parameters.values()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        None,
    )
    var_positional = next(
        (
            parameter
            for parameter in subject_parameters.values()
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL
        ),
        None,
    )

    for protocol_parameter in protocol_parameters.values():
        if protocol_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            index = protocol_positional_index[protocol_parameter.name]
            if index < len(subject_positional):
                offered = subject_positional[index]
                matched_positionally.add(offered.name)
                if protocol_parameter.default is not inspect.Parameter.empty and _is_required(
                    offered
                ):
                    mismatches.append(
                        f"member '{name}' parameter "
                        f"'{protocol_parameter.name}' at {origin}: protocol "
                        "declares an optional POSITIONAL_ONLY parameter (a "
                        "call may omit it); subject requires the parameter at "
                        f"that position ('{offered.name}')"
                    )
                continue

            # No subject parameter stands at that position. *args still
            # accepts the call; nothing else does.
            if var_positional is None:
                mismatches.append(
                    f"member '{name}' parameter "
                    f"'{protocol_parameter.name}' at {origin}: protocol "
                    "declares POSITIONAL_ONLY; subject offers "
                    + _missing_parameter_offering(
                        protocol_parameter,
                        subject_signature,
                        name,
                    )
                )
            continue

        subject_parameter = subject_parameters.get(protocol_parameter.name)

        if subject_parameter is not None:
            if subject_parameter.kind is not protocol_parameter.kind:
                mismatches.append(
                    f"member '{name}' parameter "
                    f"'{protocol_parameter.name}' at {origin}: protocol "
                    f"declares {protocol_parameter.kind.name}; subject "
                    f"offers {subject_parameter.kind.name}"
                )
                continue

            if (
                protocol_parameter.kind in _REGULAR_PARAMETER_KINDS
                and protocol_parameter.default is not inspect.Parameter.empty
                and _is_required(subject_parameter)
            ):
                mismatches.append(
                    f"member '{name}' parameter "
                    f"'{protocol_parameter.name}' at {origin}: protocol "
                    f"declares an optional {protocol_parameter.kind.name} "
                    "parameter (a call may omit it); subject requires that "
                    "parameter"
                )
            continue

        accepted_by_catch_all = False
        if protocol_parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            # The structural contract explicitly blesses delegated keyword
            # adapters. **kwargs proves that the declared name can be forwarded
            # without forcing a forwarding adapter to restate the surface.
            accepted_by_catch_all = var_keyword is not None

        # POSITIONAL_ONLY does not reach here: it is matched by position above,
        # including its own *args fallback.

        if not accepted_by_catch_all:
            mismatches.append(
                f"member '{name}' parameter "
                f"'{protocol_parameter.name}' at {origin}: protocol declares "
                f"{protocol_parameter.kind.name}; subject offers "
                + _missing_parameter_offering(
                    protocol_parameter,
                    subject_signature,
                    name,
                )
            )

    for subject_parameter in subject_parameters.values():
        if subject_parameter.kind not in _REGULAR_PARAMETER_KINDS:
            continue
        if subject_parameter.default is not inspect.Parameter.empty:
            # An optional extra accepts every protocol call and merely widens
            # the implementation surface, so it is not a structural defect.
            continue
        if subject_parameter.name in protocol_parameters:
            continue
        if subject_parameter.name in matched_positionally:
            # Already accounted for: it stands at a positional-only slot, so
            # the protocol DOES declare a parameter there -- just not one whose
            # name any caller may use. Reporting it here as an undeclared extra
            # would refuse, on the reverse sweep, exactly the implementation the
            # forward sweep just accepted.
            continue

        mismatches.append(
            f"member '{name}' parameter '{subject_parameter.name}' at "
            f"{origin}: protocol declares no parameter named "
            f"'{subject_parameter.name}'; subject offers a required "
            f"{subject_parameter.kind.name} parameter"
        )

    return mismatches
