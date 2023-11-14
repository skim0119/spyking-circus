import datetime
import os
import re
import psutil
import shutil
import sys
import textwrap
import numpy
import circus

import pkg_resources

try:
    from PyQt5 import uic
    from PyQt5.QtCore import Qt, QUrl, QProcess
    from PyQt5.QtWidgets import (QApplication, QFileDialog, QCheckBox, QDialog,
                                 QPushButton, QLineEdit, QWidget, QMessageBox)
    from PyQt5.QtGui import QTextCursor, QDesktopServices, QFont, QIcon
except ImportError:
    try:
        from PySide import uic
        from PySide.QtCore import Qt, QUrl, QProcess
        from PySide.QtGui import (QApplication, QFileDialog, QCheckBox,
                                  QPushButton, QLineEdit, QDialog,
                                  QWidget, QTextCursor, QMessageBox,
                                  QDesktopServices, QFont, QIcon)
    except ImportError:
        from PyQt4 import uic
        from PyQt4.QtCore import Qt, QUrl, QProcess
        from PyQt4.QtGui import (QApplication, QFileDialog, QCheckBox,
                                 QPushButton, QLineEdit,
                                 QWidget, QTextCursor, QMessageBox,
                                 QDesktopServices, QFont, QDialog, QIcon)

from circus.shared.messages import print_error, print_info, print_and_log, get_colored_header, init_logging
from circus.files import __supported_data_files__, list_all_file_format


if sys.platform == 'win32':
    import ctypes

    # This is the data type of the pointer that QProcess.pid() returns on Windows
    # FIXME: With QT > 5.3, use QProcess.processId() instead
    class WinProcInfo(ctypes.Structure):
        _fields_ = [
            ('hProcess', ctypes.wintypes.HANDLE),
            ('hThread', ctypes.wintypes.HANDLE),
            ('dwProcessID', ctypes.wintypes.DWORD),
            ('dwThreadID', ctypes.wintypes.DWORD),
            ]
    LPWinProcInfo = ctypes.POINTER(WinProcInfo)


def strip_ansi_codes(s):
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)


def to_str(b):
    """
    Helper function to convert a byte string (or a QByteArray) to a string --
    for Python 3, this specifies an encoding to not end up with "b'...'".
    """
    if sys.version_info[0] == 3:
        return str(b, encoding='ascii', errors='ignore')
    else:
        return str(b)


def overwrite_text(cursor, text):
    text_length = len(text)
    cursor.clearSelection()
    # Select the text after the current position (if any)
    current_position = cursor.position()
    cursor.movePosition(QTextCursor.Right,
                        mode=QTextCursor.MoveAnchor,
                        n=text_length)
    cursor.movePosition(QTextCursor.Left,
                        mode=QTextCursor.KeepAnchor,
                        n=cursor.position()-current_position)
    # Insert the text (will overwrite the selected text)
    cursor.insertText(text)


class LaunchGUI(QDialog):
    def __init__(self, app):
        super(LaunchGUI, self).__init__()
        self.app = app
        self.init_gui_layout()

    def init_gui_layout(self):
        gui_fname = pkg_resources.resource_filename('circus',
                                                    os.path.join('qt_GUI',
                                                                 'qt_launcher.ui'))
        self.ui = uic.loadUi(gui_fname)
        self.task_comboboxes = [cb for cb in self.ui.grp_tasks.children()
                                if isinstance(cb, QCheckBox)]

        logo = pkg_resources.resource_filename('circus', os.path.join('icons','icon.png'))
        self.ui.setWindowIcon(QIcon(logo))

        try:
            import cudamat as cmt
            cmt.init()
            self.HAVE_CUDA = True
        except Exception:
            self.HAVE_CUDA = False

        self.params = None
        self.ui.btn_run.clicked.connect(self.run)
        self.ui.btn_plots.clicked.connect(self.open_plot_folder)
        self.ui.btn_phy.clicked.connect(self.help_phy)
        self.ui.btn_matlab.clicked.connect(self.help_matlab)
        self.ui.btn_help_cpus.clicked.connect(self.help_cpus)
        self.ui.btn_help_gpus.clicked.connect(self.help_gpus)
        self.ui.btn_help_file_format.clicked.connect(self.help_file_format)
        self.ui.tabWidget.currentChanged.connect(self.changing_tab)
        self.ui.btn_stop.clicked.connect(self.stop)
        self.ui.btn_file.clicked.connect(self.update_data_file)
        self.ui.btn_about.clicked.connect(self.show_about)
        self.ui.rb_gui_matlab.clicked.connect(self.update_gui_command)
        self.ui.rb_gui_python.clicked.connect(self.update_gui_command)
        self.ui.btn_output.clicked.connect(self.update_output_file)
        self.ui.btn_hostfile.clicked.connect(self.update_host_file)
        self.ui.btn_log.clicked.connect(self.open_log_file)
        self.ui.cb_batch.toggled.connect(self.update_batch_mode)
        self.ui.cb_preview.toggled.connect(self.update_preview_mode)
        self.ui.cb_results.toggled.connect(self.update_results_mode)
        self.ui.cb_benchmarking.toggled.connect(self.update_benchmarking)
        self.ui.cb_merging.toggled.connect(self.update_extension)
        self.ui.cb_converting.toggled.connect(self.update_extension)
        self.ui.cb_deconverting.toggled.connect(self.update_extension)
        self.update_benchmarking()
        self.update_extension()
        for cb in self.task_comboboxes:
            cb.toggled.connect(self.store_tasks)
            cb.toggled.connect(self.update_command)
        self.ui.edit_file.textChanged.connect(self.update_command)
        self.ui.edit_output.textChanged.connect(self.update_command)
        self.ui.edit_hostfile.textChanged.connect(self.update_command)
        self.ui.edit_extension.textChanged.connect(self.update_command)
        self.ui.gui_extension.textChanged.connect(self.update_gui_command)
        self.ui.param_editor.textChanged.connect(self.save_params)
        self.ui.spin_cpus.valueChanged.connect(self.update_command)
        if not self.HAVE_CUDA:
            self.ui.spin_gpus.setEnabled(False)
        self.ui.spin_gpus.valueChanged.connect(self.update_command)
        self.store_tasks()
        self.process = None
        self.ui.closeEvent = self.closeEvent
        self.last_log_file = None
        try:
            import sklearn
        except ImportError:
            self.ui.cb_validating.setEnabled(False)
        self.ui.show()

    def store_tasks(self):
        self.stored_tasks = [cb.isChecked() for cb in self.task_comboboxes]
        if not numpy.any(self.stored_tasks):
            self.ui.btn_run.setEnabled(False)
        elif str(self.ui.edit_file.text()) != '':
            self.ui.btn_run.setEnabled(True)
            self.ui.btn_plots.setEnabled(True)

    def restore_tasks(self):
        for cb, prev_state in zip(self.task_comboboxes,
                                  self.stored_tasks):
            cb.setEnabled(True)
            cb.setChecked(prev_state)

    def update_batch_mode(self):
        batch_mode = self.ui.cb_batch.isChecked()
        self.ui.spin_cpus.setEnabled(not batch_mode)
        self.ui.spin_gpus.setEnabled(not batch_mode)
        self.ui.edit_hostfile.setEnabled(not batch_mode)
        self.ui.btn_hostfile.setEnabled(not batch_mode)
        self.update_tasks()
        self.update_extension()
        self.update_benchmarking()
        if batch_mode:
            self.ui.spin_cpus.setEnabled(not batch_mode)
            self.ui.spin_gpus.setEnabled(not batch_mode)
            self.ui.edit_hostfile.setEnabled(not batch_mode)
            self.ui.btn_hostfile.setEnabled(not batch_mode)
            self.update_tasks()
            self.update_extension()
            self.update_benchmarking()
            self.update_command()
            self.ui.cb_preview.setChecked(False)
            self.ui.cb_results.setChecked(False)
            self.ui.lbl_file.setText('Command file')
        else:
            self.last_mode = None
            self.ui.lbl_file.setText('Data file')
        self.update_command()

    def changing_tab(self):
        if self.ui.tabWidget.currentIndex() == 0:
            self.update_command()
        elif self.ui.tabWidget.currentIndex() == 2:
            self.update_gui_command()

    def update_preview_mode(self):
        preview_mode = self.ui.cb_preview.isChecked()

        self.update_tasks()

        if preview_mode:
            self.ui.cb_batch.setChecked(False)
            self.ui.cb_results.setChecked(False)

        self.update_command()

    def update_results_mode(self):
        results_mode = self.ui.cb_results.isChecked()
        self.ui.spin_cpus.setEnabled(not results_mode)
        self.ui.spin_gpus.setEnabled(not results_mode)
        self.ui.edit_hostfile.setEnabled(not results_mode)
        self.ui.btn_hostfile.setEnabled(not results_mode)
        self.update_tasks()
        self.update_extension()
        self.update_benchmarking()
        self.update_command()
        if results_mode:
            self.ui.cb_batch.setChecked(False)
            self.ui.cb_preview.setChecked(False)

    def update_result_tab(self):
        if str(self.ui.edit_file.text()) != '':
            f_next, _ = os.path.splitext(str(self.ui.edit_file.text()))
            ft = os.path.basename(os.path.normpath(f_next))
            f_results = os.path.join(f_next, ft + '.result.hdf5')
            if os.path.exists(f_results):
                self.ui.selection_gui.setEnabled(True)
                self.ui.extension_gui.setEnabled(True)
        else:
            self.ui.selection_gui.setEnabled(False)
            self.ui.extension_gui.setEnabled(False)

    def update_extension(self):
        batch_mode = self.ui.cb_batch.isChecked()
        if (not batch_mode and (self.ui.cb_merging.isChecked() or self.ui.cb_converting.isChecked() or self.ui.cb_deconverting.isChecked())):
            self.ui.edit_extension.setEnabled(True)
        else:
            self.ui.edit_extension.setEnabled(False)

    def update_benchmarking(self):
        batch_mode = self.ui.cb_batch.isChecked()
        enable = not batch_mode and self.ui.cb_benchmarking.isChecked()
        self.ui.edit_output.setEnabled(enable)
        self.ui.btn_output.setEnabled(enable)
        self.ui.cmb_type.setEnabled(enable)
        self.update_command()

    def update_tasks(self):
        batch_mode = self.ui.cb_batch.isChecked()
        preview_mode = self.ui.cb_preview.isChecked()
        results_mode = self.ui.cb_results.isChecked()
        if batch_mode or results_mode:
            self.restore_tasks()
            for cb in self.task_comboboxes:
                cb.setEnabled(False)
        elif preview_mode:
            prev_stored_tasks = self.stored_tasks
            for cb in self.task_comboboxes:
                cb.setEnabled(False)
                cb.setChecked(False)
            self.ui.cb_filtering.setChecked(True)
            self.ui.cb_whitening.setChecked(True)
            self.stored_tasks = prev_stored_tasks
        else:  # We come back from batch or preview mode
            self.restore_tasks()
        self.update_command()

    def update_data_file(self):
        if self.ui.cb_batch.isChecked():
            title = 'Select file with list of commands'
        else:
            title = 'Select data file'
        fname = QFileDialog.getOpenFileName(self, title,
                                            self.ui.edit_file.text())
        # With PyQt API 2, the return value will be a tuple (filename and filter)
        if isinstance(fname, tuple):
            fname, _ = fname
        if fname:
            self.ui.edit_file.setText(fname)

        if str(self.ui.edit_file.text()) != '':
            self.ui.btn_run.setEnabled(True)
            f_next, _ = os.path.splitext(str(self.ui.edit_file.text()))
            self.params = f_next + '.params'
            self.last_log_file = f_next + '.log'
            if os.path.exists(self.params):
                self.ui.btn_plots.setEnabled(True)
                self.update_params()
        else:
            self.ui.btn_run.setEnabled(False)

        if self.ui.tabWidget.currentIndex() == 0:
            self.update_command()
        elif self.ui.tabWidget.currentIndex() == 2:
            self.update_gui_command()

        self.update_result_tab()

        if self.params is not None:
            if not os.path.exists(self.params):
                self.create_params_file(self.params)

    def update_params(self):
        f = open(self.params, 'r')
        lines = f.readlines()
        f.close()
        text = ''.join(lines)
        self.ui.param_editor.setPlainText(text)

    def save_params(self):

        all_text = self.ui.param_editor.toPlainText()
        myfile = open(self.params, 'w')
        myfile.write(all_text)
        myfile.close()

    def update_host_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Select MPI host file',
                                            self.ui.edit_hostfile.text())
        # With PyQt API 2, the return value will be a tuple (filename and filter)
        if isinstance(fname, tuple):
            fname, _ = fname
        if fname:
            self.ui.edit_hostfile.setText(fname)

    def update_output_file(self):
        fname = QFileDialog.getSaveFileName(self, 'Output file name',
                                            self.ui.edit_output.text())
        # With PyQt API 2, the return value will be a tuple (filename and filter)
        if isinstance(fname, tuple):
            fname, _ = fname
        if fname:
            self.ui.edit_output.setText(fname)

    def open_log_file(self):
        assert self.last_log_file is not None
        QDesktopServices.openUrl(QUrl(self.last_log_file))

    def gui_command_line_args(self):

        if self.ui.rb_gui_matlab.isChecked():
            args = ['circus-gui-matlab']
        elif self.ui.rb_gui_python.isChecked():
            args = ['circus-gui-python']

        fname = str(self.ui.edit_file.text()).strip()
        if fname:
            args.append(fname)

        extension = str(self.ui.gui_extension.text()).strip()

        if extension:
            args.extend(['--extension', extension])

        return args

    def command_line_args(self):
        batch_mode = self.ui.cb_batch.isChecked()
        preview_mode = self.ui.cb_preview.isChecked()
        results_mode = self.ui.cb_results.isChecked()

        args = ['spyking-circus']
        fname = str(self.ui.edit_file.text()).strip()
        if fname:
            args.append(fname)

        if batch_mode:
            args.append('--batch')
        elif preview_mode:
            args.append('--preview')
            if self.ui.spin_cpus.value() > 1:
                args.extend(['--cpu', str(self.ui.spin_cpus.value())])
            if self.ui.spin_gpus.value() > 0:
                args.extend(['--gpu', str(self.ui.spin_gpus.value())])
        elif results_mode:
            args.append('--result')
        else:
            tasks = []
            for cb in self.task_comboboxes:
                if cb.isChecked():
                    label = str(cb.text()).lower()
                    tasks.append(label)
            if len(tasks) > 0:
                args.extend(['--method', ','.join(tasks)])
            if self.ui.spin_cpus.value() > 1:
                args.extend(['--cpu', str(self.ui.spin_cpus.value())])
            if self.ui.spin_gpus.value() > 0:
                args.extend(['--gpu', str(self.ui.spin_gpus.value())])
            hostfile = str(self.ui.edit_hostfile.text()).strip()
            if hostfile:
                args.extend(['--hostfile', hostfile])
            if 'merging' in tasks or 'converting' in tasks:
                extension = str(self.ui.edit_extension.text()).strip()
                if extension:
                    args.extend(['--extension', extension])
            if 'benchmarking' in tasks:
                args.extend(['--output', str(self.ui.edit_output.text())])
                args.extend(['--type', str(self.ui.cmb_type.currentText())])
        return args

    def update_gui_command(self):
        args = ' '.join(self.gui_command_line_args())
        self.ui.edit_command.setPlainText(args)

    def update_command(self, text=None):
        _ = text  # (PyCharm code inspection)
        args = ' '.join(self.command_line_args())
        self.ui.edit_command.setPlainText(args)

    def run(self):
        if self.ui.cb_batch.isChecked():
            self.last_log_file = None
        else:
            if self.params is None:
                self.create_params_file(self.params)
                return
            elif not os.path.exists(self.params):
                self.create_params_file(self.params)
                return

        if self.ui.tabWidget.currentIndex() == 0:
            args = self.command_line_args()
        elif self.ui.tabWidget.currentIndex() == 2:
            args = self.gui_command_line_args()

        self.update_result_tab()

        # # Start process
        self.ui.edit_stdout.clear()
        format_ = self.ui.edit_stdout.currentCharFormat()
        format_.setFontWeight(QFont.Normal)
        format_.setForeground(Qt.blue)
        self.ui.edit_stdout.setCurrentCharFormat(format_)
        time_str = datetime.datetime.now().ctime()
        start_msg = '''\
                       Starting spyking circus at {time_str}.

                       Command line call:
                       {call}
                    '''.format(time_str=time_str, call=' '.join(args))
        self.ui.edit_stdout.appendPlainText(textwrap.dedent(start_msg))
        format_.setForeground(Qt.black)
        self.ui.edit_stdout.setCurrentCharFormat(format_)
        self.ui.edit_stdout.appendPlainText('\n')

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.append_output)
        self.process.readyReadStandardError.connect(self.append_error)
        self.process.started.connect(self.process_started)
        self.process.finished.connect(self.process_finished)
        self.process.error.connect(self.process_errored)
        self._interrupted = False
        self.process.start(args[0], args[1:])

    def restore_gui(self):
        for widget, previous_state in self._previous_state:
            widget.setEnabled(previous_state)
        self.app.restoreOverrideCursor()

    def process_started(self):
        # Disable everything except for the stop button and the output area
        all_children = [obj for obj in self.ui.findChildren(QWidget)
                        if isinstance(obj, (QCheckBox, QPushButton, QLineEdit))]
        self._previous_state = [(obj, obj.isEnabled()) for obj in all_children]
        for obj in all_children:
            obj.setEnabled(False)
        self.ui.btn_stop.setEnabled(True)
        # If we let the user interact, this messes with the cursor we use to
        # support the progress bar display
        self.ui.edit_stdout.setTextInteractionFlags(Qt.NoTextInteraction)
        self.app.setOverrideCursor(Qt.WaitCursor)

    def process_finished(self, exit_code):
        format_ = self.ui.edit_stdout.currentCharFormat()
        format_.setFontWeight(QFont.Bold)
        if exit_code == 0:
            if self._interrupted:
                color = Qt.red
                msg = 'Process interrupted by user'
            else:
                color = Qt.green
                msg = 'Process exited normally'
        else:
            color = Qt.red
            msg = ('Process exited with exit code %d' % exit_code)
        format_.setForeground(color)
        self.ui.edit_stdout.setCurrentCharFormat(format_)
        self.ui.edit_stdout.appendPlainText(msg)
        self.restore_gui()
        self.ui.edit_stdout.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.process = None
        self.ui.btn_log.setEnabled(self.last_log_file is not None and os.path.isfile(self.last_log_file))

    def process_errored(self):
        try:
            exit_code = self.process.exitCode()
        except Exception:
            exit_code = 0
        format_ = self.ui.edit_stdout.currentCharFormat()
        format_.setFontWeight(QFont.Bold)
        format_.setForeground(Qt.red)
        self.ui.edit_stdout.setCurrentCharFormat(format_)
        self.ui.edit_stdout.appendPlainText('Process exited with exit code %s' % exit_code)
        self.restore_gui()
        self.ui.edit_stdout.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.process = None

    def add_output_lines(self, lines):
        """Add the output line by line to the text area, jumping back to the
        beginning of the line when we encounter a carriage return (to
        correctly display progress bars)
        """
        cursor = self.ui.edit_stdout.textCursor()
        cursor.clearSelection()
        splitted_lines = lines.split('\n')
        for idx, line in enumerate(splitted_lines):
            if '\r' in line:
                chunks = line.split('\r')
                overwrite_text(cursor, chunks[0])  # No going back for the first chunk
                for chunk in chunks[1:]:  # Go back to start of line for each \r
                    cursor.movePosition(QTextCursor.StartOfLine)
                    overwrite_text(cursor, chunk)
            else:
                overwrite_text(cursor, line)

            # Take care to not introduce new newlines
            if '\n' in lines and (idx == 0 or idx != len(splitted_lines) - 1):
                cursor.movePosition(QTextCursor.EndOfLine)
                cursor.insertText('\n')
        self.ui.edit_stdout.setTextCursor(cursor)

    def append_output(self):
        if self.process is None:  # Can happen when manually interrupting
            return
        lines = strip_ansi_codes(to_str(self.process.readAllStandardOutput()))
        self.add_output_lines(lines)
        # We manually deal with keyboard input in the output
        if 'Export already made! Do you want to erase everything? (y)es / (n)o' in lines:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle('Erase everything?')
            msg.setText('Export already made! Do you want to erase everything?')
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            answer = msg.exec_()
            if answer == QMessageBox.Yes:
                answer_string = 'y'
            else:
                answer_string = 'n'
        elif 'Do you want SpyKING CIRCUS to export PCs? (a)ll / (s)ome / (n)o' in lines:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle('Export PCs?')
            msg.setText('Do you want SpyKING CIRCUS to export PCs?')
            no_button = msg.addButton('No', QMessageBox.NoRole)
            some_button = msg.addButton('Some', QMessageBox.YesRole)
            all_button = msg.addButton('All', QMessageBox.YesRole)
            msg.exec_()
            if msg.clickedButton() == no_button:
                answer_string = 'n'
            elif msg.clickedButton() == some_button:
                answer_string = 's'
            elif msg.clickedButton() == all_button:
                answer_string = 'a'
            else:
                answer_string = 'n'
        elif 'Do you want to delete these files? [Y/n]' in lines:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle('Delete everything?')
            msg.setText('Files already deconverted! Do you want to delete everything?')
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            answer = msg.exec_()
            if answer == QMessageBox.Yes:
                answer_string = 'y'
            else:
                answer_string = 'n'
        elif 'You should re-export the data because of a fix in 0.6' in lines:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle('You should re-export the data because of a fix in 0.6')
            msg.setText('Continue anyway (results may not be fully correct)?')
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            answer = msg.exec_()
            if answer == QMessageBox.Yes:
                answer_string = 'y'
            else:
                answer_string = 'n'
        else:
            answer_string = ''

        if answer_string:
            to_write = answer_string + '\n'
            to_write = to_write.encode('utf-8')
            self.process.write(to_write)
            self.add_output_lines(to_write)

    def append_error(self):
        if self.process is None:  # Can happen when manually interrupting
            return
        lines = strip_ansi_codes(to_str(self.process.readAllStandardError()))
        self.add_output_lines(lines)

    def stop(self, force=False):
        if self.process is not None:

            if not force:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle('Confirm process termination')
                msg.setText(
                    'This will terminate the running process. Are you sure '
                    'you want to do this?')
                msg.setInformativeText(
                    'Interrupting the process may leave partly '
                    'created files that cannot be used for '
                    'further analysis.')
                msg.addButton('Stop process', QMessageBox.YesRole)
                cancel_button = msg.addButton('Cancel', QMessageBox.NoRole)
                msg.setDefaultButton(cancel_button)
                msg.exec_()
                if msg.clickedButton() == cancel_button:
                    # Continue to run
                    return

            self._interrupted = True
            # Terminate child processes as well
            pid = int(self.process.pid())
            if sys.platform == 'win32' and pid != 0:
                # The returned value is not a PID but a pointer
                lp = ctypes.cast(pid, LPWinProcInfo)
                pid = lp.contents.dwProcessID

            if pid != 0:
                process = psutil.Process(pid)
                children = process.children(recursive=True)
                for proc in children:
                    proc.terminate()
                gone, alive = psutil.wait_procs(children, timeout=3)
                for proc in alive:
                    proc.kill()

                self.process.terminate()
                self.process = None

    def create_params_file(self, fname):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Parameter file %r not found, do you want SpyKING CIRCUS to "
                    "create it for you?" % fname)
        msg.setWindowTitle("Generate parameter file?")
        msg.setInformativeText("This will create a parameter file from a "
                               "template file and open it in your system's "
                               "standard text editor. Fill properly before "
                               "launching the code. See the documentation "
                               "for details")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        answer = msg.exec_()
        if answer == QMessageBox.Yes:
            user_path = os.path.join(os.path.expanduser('~'), 'spyking-circus')
            if os.path.exists(user_path + 'config.params'):
                config_file = os.path.abspath(user_path + 'config.params')
            else:
                config_file = os.path.abspath(
                    pkg_resources.resource_filename('circus', 'config.params'))
            shutil.copyfile(config_file, fname)
            self.params = fname
            self.last_log_file = fname.replace('.params', '.log')
            self.update_params()

    def show_about(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("SpyKING CIRCUS v%s" %circus.__version__)
        msg.setWindowTitle("About")
        msg.setInformativeText(
            "Documentation can be found at\n"
            "http://spyking-circus.rtfd.org\n"
            "\n"
            "Open a browser to see the online help?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        answer = msg.exec_()
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl("http://spyking-circus.rtfd.org"))

    def help_cpus(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Setting the number of CPUs")
        msg.setWindowTitle("Number of CPUs")
        msg.setInformativeText(
            "SpyKING CIRCUS can use several CPUs "
            "either locally or on multiple machines "
            "using MPI (see documentation) "
            "\n"
            "\n"
            "You have %d local CPUs available" % psutil.cpu_count()
        )
        msg.setStandardButtons(QMessageBox.Close)
        msg.setDefaultButton(QMessageBox.Close)
        answer = msg.exec_()

    def help_gpus(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Setting the number of GPUs")
        msg.setWindowTitle("Number of GPUs")
        if not self.HAVE_CUDA:
            info = "No GPUs are detected on your system"
        else:
            gpu_id = 0
            is_available = True
            while is_available:
                try:
                    cmt.cuda_set_device(gpu_id)
                    is_available = True
                except Exception:
                    is_available = False
            info = "%d GPU is detected on your system" % (gpu_id + 1)

        msg.setInformativeText(
            "SpyKING CIRCUS can use several GPUs\n"
            "either locally or on multiple machine\n"
            "using MPI (see documentation)"
            "\n"
            "\n"
            "%s" % info
        )
        msg.setStandardButtons(QMessageBox.Close)
        msg.setDefaultButton(QMessageBox.Close)
        answer = msg.exec_()

    def help_file_format(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Supported file formats")
        msg.setWindowTitle("File formats")

        msg.setInformativeText("\n".join(list_all_file_format()))
        msg.setStandardButtons(QMessageBox.Close)
        msg.setDefaultButton(QMessageBox.Close)
        answer = msg.exec_()

    def open_plot_folder(self):
        f_next, _ = os.path.splitext(str(self.ui.edit_file.text()))
        plot_folder = os.path.join(f_next, 'plots')
        QDesktopServices.openUrl(QUrl(plot_folder))

    def help_phy(self):
        QDesktopServices.openUrl(QUrl("https://github.com/kwikteam/phy-contrib"))

    def help_matlab(self):
        QDesktopServices.openUrl(QUrl("http://ch.mathworks.com/products/matlab/"))

    def closeEvent(self, event):
        if self.process is not None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Confirm process interruption")
            msg.setText(
                "Closing the window will terminate the running process. "
                "Do you really want to exit?"
            )
            msg.setInformativeText(
                "Interrupting the process may leave partly "
                "created files that cannot be used for "
                "further analysis."
            )
            close_button = msg.addButton("Stop and close", QMessageBox.YesRole)
            cancel_button = msg.addButton("Cancel", QMessageBox.NoRole)
            msg.setDefaultButton(cancel_button)
            msg.exec_()
            if msg.clickedButton() == close_button:
                self.stop(force=True)
                super(LaunchGUI, self).closeEvent(event)
            else:
                event.ignore()


def main():
    app = QApplication([])
    gui = LaunchGUI(app)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
