import argparse
from dotenv import load_dotenv


def main():
    """Parse command-line arguments for the n-gram predictor project."""
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(
        description="N-gram next-word prediction system."
    )
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all"],
        help="Choose which pipeline step to run."
    )

    args = parser.parse_args()

    if args.step == "dataprep":
        print("Data preparation step selected.")
    elif args.step == "model":
        print("Model step selected.")
    elif args.step == "inference":
        print("Inference step selected.")
    elif args.step == "all":
        print("Running all steps.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()