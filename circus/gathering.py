from circus.shared.utils import *
import circus.shared.files as io
from circus.shared.messages import init_logging


def main(params, nb_cpu, nb_gpu, use_gpu):

    _ = init_logging(params.logfile)
    logger = logging.getLogger('circus.gathering')
    io.collect_data(nb_cpu, params, erase=False)
