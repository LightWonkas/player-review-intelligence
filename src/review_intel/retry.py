"""Retries with exponential backoff, plus a pacer that spaces out requests.

Store endpoints are free and public, so we behave politely: pause between pages,
retry only errors that are likely temporary, and back off harder each time.
Both helpers take an injectable `sleep` (and clock / random source) so tests run
instantly and deterministically.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

Sleep = Callable[[float], None]


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    # Up to +25% random extra wait, so many clients that failed together
    # don't all retry at the same instant.
    jitter: float = 0.25


DEFAULT_RETRY = RetryPolicy()


def backoff_delay(
    attempt: int, policy: RetryPolicy, rand: Callable[[], float] = random.random
) -> float:
    """Delay before retry number `attempt` (1-based): 1s, 2s, 4s, ... capped, plus jitter."""
    delay = min(policy.max_delay, policy.base_delay * 2.0 ** (attempt - 1))
    return delay * (1 + policy.jitter * rand())


def with_retries[T](
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    policy: RetryPolicy = DEFAULT_RETRY,
    sleep: Sleep = time.sleep,
    rand: Callable[[], float] = random.random,
    description: str = "request",
) -> T:
    """Call `fn`, retrying temporary failures. Permanent errors (e.g. 404) raise at once."""
    for attempt in range(1, policy.attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == policy.attempts or not is_retryable(exc):
                raise
            delay = backoff_delay(attempt, policy, rand)
            logger.warning(
                "%s failed (%s: %s); retry %d/%d in %.1fs",
                description,
                type(exc).__name__,
                exc,
                attempt,
                policy.attempts - 1,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable: the loop always returns or raises")


class Pacer:
    """Keeps at least `min_interval` seconds between consecutive requests."""

    def __init__(
        self,
        min_interval: float,
        *,
        sleep: Sleep = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._min_interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()
