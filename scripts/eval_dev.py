"""Development evaluation runner script."""

import os
import subprocess
import sys


def main() -> None:
    """Run evaluation tests in development mode."""
    os.environ["SCOUTER_ENV"] = "development"
    try:
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "ptw", "evals/"],
            check=True,
            shell=False,  # Explicitly disable shell for security
        )
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
