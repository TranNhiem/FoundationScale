"""The single refusal the FoundationSkills vetting gate raises (A6.1): an input it will not decide.

The gate never guesses. When a skill tree cannot be read, a lock entry conflicts with the bytes it
pins, or a review rejects the code, the gate refuses: it names what is wrong and exits ``96``
rather than passing, red-flagging or measuring the package. This module holds that one exception
type and nothing else, so every module of :mod:`foundationskills.vetting` -- and every caller above
it -- can raise and catch the gate's refusal without importing anything else.
"""


class VettingRefused(ValueError):
    """A vetting input is missing, malformed or unscannable: the gate refuses to decide.

    The message is the reason and always names what the gate could not use (a file, a key, a review
    whose digest no longer matches). It subclasses :class:`ValueError` because "this input is
    invalid for the decision being asked" is precisely what it means; exit code ``96``.
    """
