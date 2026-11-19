import pytest

from review_intel.retry import Pacer, RetryPolicy, backoff_delay, with_retries

NO_JITTER = RetryPolicy(attempts=4, base_delay=1.0, max_delay=30.0, jitter=0.0)


class Flaky:
    """Fails `failures` times with `exc`, then returns "ok"."""

    def __init__(self, failures: int, exc: Exception) -> None:
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"


def _always(_: Exception) -> bool:
    return True


def _never(_: Exception) -> bool:
    return False


def test_backoff_doubles_and_caps() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=0.0)

    delays = [backoff_delay(attempt, policy) for attempt in range(1, 6)]

    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_jitter_adds_at_most_the_configured_fraction() -> None:
    policy = RetryPolicy(base_delay=2.0, jitter=0.25)

    assert backoff_delay(1, policy, rand=lambda: 1.0) == 2.5
    assert backoff_delay(1, policy, rand=lambda: 0.0) == 2.0


def test_retries_temporary_failures_then_succeeds() -> None:
    sleeps: list[float] = []
    fn = Flaky(failures=2, exc=TimeoutError("slow"))

    result = with_retries(fn, is_retryable=_always, policy=NO_JITTER, sleep=sleeps.append)

    assert result == "ok"
    assert fn.calls == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []
    fn = Flaky(failures=10, exc=TimeoutError("down"))

    with pytest.raises(TimeoutError):
        with_retries(fn, is_retryable=_always, policy=NO_JITTER, sleep=sleeps.append)

    assert fn.calls == 4
    assert len(sleeps) == 3


def test_permanent_errors_are_not_retried() -> None:
    sleeps: list[float] = []
    fn = Flaky(failures=1, exc=ValueError("bad id"))

    with pytest.raises(ValueError):
        with_retries(fn, is_retryable=_never, policy=NO_JITTER, sleep=sleeps.append)

    assert fn.calls == 1
    assert sleeps == []


def test_pacer_waits_only_the_remaining_interval() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    pacer = Pacer(1.0, sleep=fake_sleep, clock=lambda: now[0])

    pacer.wait()  # first request: no wait
    now[0] += 0.25
    pacer.wait()  # 0.25s passed, so wait 0.75s
    now[0] += 5.0
    pacer.wait()  # plenty of time passed: no wait

    assert sleeps == [0.75]
