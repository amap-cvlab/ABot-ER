"""Unsupported legacy entry point. Intentionally disabled before imports."""


def main():
    raise RuntimeError("Legacy WebUI is disabled; use the documented local inference entry point.")


if __name__ == "__main__":
    main()
