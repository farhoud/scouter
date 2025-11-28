import asyncio
import contextlib

from watchfiles import awatch

from scripts.run_eval import run_eval


async def watch_and_eval():
    """
    Watch src/ for changes and run mini eval on each change.
    """
    async for _changes in awatch("src/"):
        with contextlib.suppress(Exception):
            run_eval(num_docs=2, num_queries=1)
        # Debounce
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(watch_and_eval())
