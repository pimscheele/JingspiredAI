"""Rate-limit helper for coordinating OpenAI calls across threads."""

import collections
import threading
import time
from collections.abc import Hashable


class RateLimiter:
    """Enforces request and token limits within a sliding time window."""

    def __init__(self, tpm_limit_per_min: int, rpm_limit_per_min: int, window_seconds: int = 10):
        """Initializes the rate_limiter window and shared counters."""
        self.request_limit = rpm_limit_per_min / (60 / window_seconds)
        self.token_limit = tpm_limit_per_min / (60 / window_seconds)
        self.window_seconds = window_seconds

        self.requests: collections.deque[tuple[object, float]] = collections.deque()
        self.tokens: collections.deque[tuple[object, float, int]] = collections.deque()

        self.lock = threading.Lock()

    def _prune_old_entries(self, now: float) -> None:
        """Remove request and token entries outside the active window."""
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0][1] <= cutoff:
            self.requests.popleft()
        while self.tokens and self.tokens[0][1] <= cutoff:
            self.tokens.popleft()

    def _current_usage(self) -> tuple[int, int]:
        """Return current request and token usage within the active window."""
        return len(self.requests), sum(t for _, _, t in self.tokens)

    def acquire(self, batch_id: Hashable, token_estimate: int) -> None:
        """Blocks until the batch can be processed within the current rate limits."""
        while True:
            wait_time: float | None = None

            with self.lock:
                now = time.monotonic()
                self._prune_old_entries(now)
                req_count, tok_count = self._current_usage()

                if token_estimate >= self.token_limit:
                    raise ValueError("BATCH TO LARGE TO PROCESS")

                can_take = (
                    req_count + 1 <= self.request_limit
                    and tok_count + token_estimate <= self.token_limit
                )

                if can_take:
                    self.requests.append((batch_id, now))
                    self.tokens.append((batch_id, now, int(token_estimate)))
                    return

                candidates = []
                if self.requests:
                    candidates.append((self.requests[0][1] + self.window_seconds) - now)
                if self.tokens:
                    candidates.append((self.tokens[0][1] + self.window_seconds) - now)

                wait_time = max(0.01, min(candidates) if candidates else 0.05)

            time.sleep(wait_time)
