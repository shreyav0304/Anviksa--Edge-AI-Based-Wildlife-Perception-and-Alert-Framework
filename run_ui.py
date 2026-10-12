"""Launch the offline desktop dashboard with actionable startup errors."""
import sys

def main() -> int:
    if sys.version_info < (3, 10):
        print("Anviksa requires Python 3.10 or newer. See docs/raspberry_pi_setup.md.", file=sys.stderr)
        return 1
    try:
        from ui.app import main as launch
        return launch()
    except (ImportError, OSError) as exc:
        print(f"Unable to launch Anviksa: {exc}\nActivate the app environment and install its dependencies. On Pi OS, install python3-tk as well.\nSee docs/raspberry_pi_setup.md.", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
