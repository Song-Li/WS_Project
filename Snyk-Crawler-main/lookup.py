#!/usr/bin/env python3
"""
Outputs versions and names of required field
"""
import argparse
import logging
import math
from pathlib import Path
import csv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        type=Path,
        help="path to the csv containing all data",
    )
    parser.add_argument(
        "fields",
        type=str,
        help="Fields that require to output",
        nargs="*"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the output file"
    )

    # verbosity = parser.add_mutually_exclusive_group()
    # verbosity.add_argument(
    #     "-v",
    #     "--verbose",
    #     action="store_const",
    #     const=logging.DEBUG,
    #     default=logging.INFO,
    # )
    # verbosity.add_argument(
    #     "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    # )

    return parser.parse_args()


def main():
    args = parse_args()
    names = ['Label','Vulnerability Name','Package Name','Package Version','Type','Publish Date']
    output_rows = []
    output_path:Path = None
    with open(args.input_csv, mode='r', newline='') as csvfile:
        f = csv.reader(csvfile)
        for row in f:
            # tmp_str = ''
            # for c in row[3]:
            #     if c.isdigit():
            #         tmp_str += c
            #     elif (c == '.') or (c == ';'):
            #         tmp_str += c
            #     else:
            #         continue
            # row[3] = tmp_str
            for field in args.fields:
                if (field in row[1]):
                    output_rows.append(row)
    if args.output is None:
        model_path = Path("output_" + len(output_rows) + ".csv")
    else:
        model_path = args.output
    with open(model_path, mode='a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(names)
        writer.writerows(output_rows)    
    


if __name__ == "__main__":
    main()
