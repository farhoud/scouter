import random
import time

from openai import APIError, RateLimitError

ERROR_MAX_RETRY = "max retries exceeded"


def retry_loop(func, max_retries=5, base_delay=1):
    last_exception: BaseException | None = None

    for attempt in range(max_retries):
        try:
            return func()
        except (RateLimitError, APIError) as e:  # noqa: PERF203
            last_exception = e
            if attempt == max_retries - 1:
                break

            sleep_time = base_delay * (2**attempt) + random.uniform(0, 0.5)  # noqa: S311
            time.sleep(sleep_time)

    # If we reach here, all retries failed
    raise last_exception or RuntimeError(ERROR_MAX_RETRY)
