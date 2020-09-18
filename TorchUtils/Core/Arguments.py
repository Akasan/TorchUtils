import argparse
import warnings
warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", help="batch size", type=int, default=256)
    parser.add_argument("-e", "--epochs", help="the number of epochs", type=int, default=10)

    return parser.parse_args()
