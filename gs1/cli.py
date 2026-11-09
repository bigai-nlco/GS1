"""Serve a GS1 model through the typed decision API."""

import argparse


def main(argv=None):
    from gs1.api.options import add_serving_arguments, serving_settings

    parser = argparse.ArgumentParser(prog="gs1", description=__doc__)
    parser.add_argument("--version", action="version", version="gs1 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve a local or Hugging Face GS1 model")
    add_serving_arguments(serve)
    args = parser.parse_args(argv)

    import uvicorn

    from gs1.api.app import create_app

    settings = serving_settings(args)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
