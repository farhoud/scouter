import os
import subprocess
import sys


def main():
    os.environ["SCOUTER_ENV"] = "development"
    try:
        subprocess.run([sys.executable, "-m", "ptw", "evals/"], check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
