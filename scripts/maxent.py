import sys, os
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import scipy.interpolate as interp
from PyQt5 import QtWidgets

file_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = '/'.join(file_dir.split('/')[:-1])
sys.path.insert(0, package_dir)

# import ana_cont.continuation as cont
# from gui.maxent_ui import Ui_MainWindow
from gui.maxent_backend import MainWindow




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())