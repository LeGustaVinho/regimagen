"""Compatibility entry point for the ImageGen validation tool."""

if __package__:
    from .validate_imagegen import main
else:
    from validate_imagegen import main


if __name__ == "__main__":
    raise SystemExit(main())
