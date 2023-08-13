import argparse

def main(args):
    print("Input file:", args.input_file)
    print("Output file:", args.output_file)
    print("Verbose mode:", args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with command-line arguments")
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    args = parser.parse_args()
    main(args)