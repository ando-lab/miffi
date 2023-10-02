"""
cryo-EM micrograph filtering utilizing Fourier space information
"""

def main():
    import argparse
    import miffi
    
    parser = argparse.ArgumentParser(description=__doc__,prog='miffi')
    
    parser.add_argument(
        '--version', action='version', version=f"miffi version: {miffi.__version__}"
    )
    
    import miffi.categorize
    import miffi.download
    import miffi.inference
    import miffi.write_png
    
    modules = [
        miffi.categorize,
        miffi.download,
        miffi.inference,
        miffi.write_png,
    ]

    subparsers = parser.add_subparsers(title="Choose a command")
    subparsers.required = True

    for module in modules:
        parseri = subparsers.add_parser(module.__name__.split('.')[-1], description=module.__doc__)
        parseri.set_defaults(func=module.main)
        module.add_args(parseri)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()