# phase00_part1_check_setup.py
#
# A small script that checks your environment is set up correctly.
# You haven't learned Python syntax yet (that's Part 2!) — so for now,
# just RUN this file and read its output. Come back and read the code
# itself after finishing Part 2; it will make a lot more sense then.
#
# How to run this:
#   1. Make sure your virtual environment is activated — you should see
#      (venv) at the start of your terminal prompt.
#   2. Put this file inside your project folder.
#   3. Run it:
#        python phase00_part1_check_setup.py     (Windows)
#        python3 phase00_part1_check_setup.py    (Mac)

import sys
import os


def check_python_version():
    """Confirm Python 3.9 or higher is installed (this course needs 3.9+)."""
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 9
    status = "OK" if ok else "TOO OLD — please reinstall from python.org"
    print(f"[{'PASS' if ok else 'FAIL'}] Python version: "
          f"{version.major}.{version.minor}.{version.micro}  ({status})")
    return ok


def check_pip_available():
    """Confirm pip (the package installer) is available."""
    try:
        import pip
        print(f"[PASS] pip is available")
        return True
    except ImportError:
        print(f"[FAIL] pip is not available — this is unusual, "
              f"try reinstalling Python")
        return False


def check_virtual_environment():
    """
    Confirm we are running INSIDE a virtual environment, not the
    system-wide Python. This checks for the telltale sign Python leaves
    behind when a venv is active.
    """
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    if in_venv:
        print(f"[PASS] Running inside a virtual environment")
    else:
        print(f"[WARN] NOT running inside a virtual environment — "
              f"did you forget to activate it? (see Section 5.4 of the guide)")
    return in_venv


def check_working_directory():
    """Show where this script is actually running from, for sanity checking."""
    cwd = os.getcwd()
    print(f"[INFO] Current folder: {cwd}")


def main():
    print("=" * 60)
    print("  Phase 00 Part 1 — Environment Check")
    print("=" * 60)
    print()

    results = []
    results.append(check_python_version())
    results.append(check_pip_available())
    venv_ok = check_virtual_environment()
    check_working_directory()

    print()
    print("=" * 60)
    if all(results) and venv_ok:
        print("  ALL CHECKS PASSED — you're ready for Phase 00 Part 2!")
    elif all(results):
        print("  Mostly good — fix the virtual environment warning above,")
        print("  then re-run this script.")
    else:
        print("  Some checks failed — see the FAIL lines above, and check")
        print("  the Troubleshooting FAQ in the main Part 1 guide.")
    print("=" * 60)


if __name__ == "__main__":
    main()
