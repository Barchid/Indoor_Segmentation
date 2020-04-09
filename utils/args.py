import argparse


def get_args(test_mode=False):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')

    if test_mode:
        argparser.add_argument(
            '-v', '--visualize',
            dest='visualize',
            default='None',
            help='Flag that indicates whether the predictions performed in dataset evaluation must be visualized.'
        )
    args = argparser.parse_args()
    return args
