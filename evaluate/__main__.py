from jsonargparse import auto_cli

from .pipeline import main

if __name__ == "__main__":
    auto_cli(main, as_positional=False)
