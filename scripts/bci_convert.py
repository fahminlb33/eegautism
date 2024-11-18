import os
import glob
import argparse
import dataclasses
import subprocess
from pprint import pprint


@dataclasses.dataclass
class CliArguments:
    input_path: str
    output_path: str

    @staticmethod
    def parse(args):
        norm = {k.replace("-", "_"): v for k, v in vars(args).items()}
        return CliArguments(**norm)


def main(args: CliArguments):
    for file in glob.glob(os.path.join(args.input_path, "*.dat")):
        print("Processing: ", file)
        output_name = os.path.splitext(os.path.basename(file))[0] + ".edf"
        output_path = os.path.join(args.output_path, output_name)
        subprocess.call(["bin2rec", "-f=EDF", file, output_path])


if __name__ == "__main__":
    # create CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input-path", type=str)
    parser.add_argument("output-path", type=str)

    # parse CLI
    args = CliArguments.parse(parser.parse_args())
    pprint(repr(args))

    # start the app!
    main(args)
