from circus.shared.utils import *
from circus.shared import gui
from circus.shared.messages import init_logging, print_and_log
from circus.shared.utils import query_yes_no
import pylab
import re

def main(params, nb_cpu, nb_gpu, use_gpu, extension):

    _ = init_logging(params.logfile)
    logger = logging.getLogger('circus.merging')
    file_out_suff = params.get('data', 'file_out_suff')
    erase_all = params.getboolean('merging', 'erase_all')
    extension_in = extension
    extension_out = '-merged'

    # Erase previous results (if user agrees).
    if comm.rank == 0:
        existing_file_paths = [
            file_path
            for file_path in [
                file_out_suff + ".%s%s.hdf5" % (file_id, extension_out)
                for file_id in ['templates', 'clusters', 'result']
            ]
            if os.path.isfile(file_path)
        ]
        existing_directory_path = [
            directory_path
            for directory_path in [
                file_out_suff + "%s.GUI" % extension_out
            ]
            if os.path.isdir(directory_path)
        ]
        if len(existing_file_paths) > 0 or len(existing_directory_path) > 0:
            if not erase_all:
                erase = query_yes_no(
                    "Merging already done! Do you want to erase previous merging results?", default=None
                )
            else:
                erase = True
            if erase:
                for path in existing_file_paths:
                    os.remove(path)
                    if comm.rank == 0:
                        print_and_log(["Removed file %s" % path], 'debug', logger)
                for path in existing_directory_path:
                    shutil.rmtree(path)
                    if comm.rank == 0:
                        print_and_log(["Removed directory %s" % path], 'debug', logger)

    comm.Barrier()

    if params.getfloat('merging', 'auto_mode') == 0:
        from sys import platform
        if not platform == 'win32':
            if not ('DISPLAY' in os.environ and re.search(":\d", os.environ['DISPLAY'])!=None):
                print_and_log(['Preview mode can not be used, check DISPLAY variable'], 'error', logger)
                sys.exit(0)

        if comm.rank == 0:

            try:
                from PyQt5.QtWidgets import QApplication
            except ImportError:
                from matplotlib.backends import qt_compat
                use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE

                if use_pyside:
                    from PySide.QtGui import QApplication
                else:
                    from PyQt4.QtGui import QApplication

            app = QApplication([])
            try:
                pylab.style.use('ggplot')
            except Exception:
                pass
        else:
            app = None
    else:
        app = None


    if comm.rank == 0:
        print_and_log(['Launching the merging GUI...'], 'debug', logger)

    _ = gui.MergeWindow(params, app, extension_in, extension_out)
    sys.exit(app.exec_())
