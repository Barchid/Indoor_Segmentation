import argparse


def get_args(test_args=False, seg_grad_cam=False):
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
            action='store_true',
            help="Flag that indicates whether the images from test must be visualized.s"
        )

    if seg_grad_cam:
        argparser.add_argument(
            '-l', '--layer',
            dest='layer',
            default='None',
            help='Layer to extract seg grad cam')

        argparser.add_argument(
            '--class_id',
            dest='class_id',
            required=True,
            help='class id to analyse with SEG grad cam'
        )

    args = argparser.parse_args()
    return args
