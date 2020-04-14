import argparse


def get_args(test_args=False):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    if test_args:
        argparser.add_argument(
            '-v', '--visualize',
            dest='visualize',
            default='None',
            help="Flag that indicates whether the images from test must be visualized.s"
        )
    args = argparser.parse_args()
    return args
