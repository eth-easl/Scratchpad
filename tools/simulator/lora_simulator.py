import argparse
import humanize


def parse_pair(s: str):
    """Parse a WxH dimension pair (e.g. "4000x4000") into a (W, H) tuple of ints."""
    try:
        w, h = map(int, s.lower().split("x"))
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid pair: '{s}', expected format WIDTHxHEIGHT"
        )
    return w, h


def main():
    p = argparse.ArgumentParser(
        description="Compute total LoRA parameters given linear dims and number of layers."
    )
    p.add_argument(
        "--linears",
        "-l",
        type=parse_pair,
        nargs="+",
        required=True,
        help="One or more linear dims as WxH (e.g. 4000x4000 1000x4000 ...)",
    )
    p.add_argument(
        "--layers",
        "-n",
        type=int,
        default=1,
        help="Number of transformer layers (default: 1)",
    )
    args = p.parse_args()

    total = sum(w * h for w, h in args.linears) * args.layers
    print(f"Total number of parameters: {humanize.intcomma(total)}")


if __name__ == "__main__":
    main()
