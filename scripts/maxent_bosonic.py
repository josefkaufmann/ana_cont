#!/usr/bin/env python
import sys, os, traceback
from PyQt5 import QtWidgets

sys.excepthook = traceback.print_exception

file_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, package_dir)
from gui.maxent_bosonic_frontend import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())