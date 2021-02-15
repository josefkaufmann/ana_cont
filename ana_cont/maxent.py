import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

# import continuation as cont
from maxent_ui import Ui_MainWindow

def test_function(some_text):
    print('you clicked the button {}'.format(some_text))

class RealFrequencyGrid(object):
    def __init__(self, wmax=None, nw=None, type=None):
        self.wmax = wmax
        self.nw = nw
        self.type = type

    def update_wmax(self, wmax):
        print('update wmax to {}'.format(wmax))
        self.wmax = wmax

    def update_nw(self, nw):
        self.nw = nw

    def update_type(self, type):
        self.type = type

    def __str__(self):
        return 'real-frequency grid (wmax: {}, nw: {}, type: {})'.format(self.wmax, self.nw, self.type)

    def create_grid(self):
        if self.type == 'linear':
            self.grid = np.linspace(-self.wmax, self.wmax, num=self.nw, endpoint=True)
        elif self.type == 'centered':
            self.grid = self.wmax * np.tan(np.linspace(-np.pi / 2.1, np.pi / 2.1, num=self.nw)) / np.tan(np.pi / 2.1)
        print(self)

class InputData(object):
    def __init__(self, fname=None, iter_type=None, iter_num=None, data_type=None,
                 atom=None, orbital=None, spin=None, num_mats=None):
        self.fname = fname
        self.iter_type = iter_type
        self.iter_num = iter_num
        self.data_type = data_type
        try:
            self.atom = int(atom)
        except ValueError:
            print('could not set atom')

        try:
            self.orbital = orbital
        except ValueError:
            print('could not set orbital')

        self.spin = spin

        try:
            self.num_mats = num_mats
        except ValueError:
            print('could not set num mats')

        if self.iter_type is None:
            self.iter_type = 'dmft'
        if self.iter_num is None or self.iter_num == '':
            self.iter_num = 'last'
        self.get_iteration()

    def update_fname(self, fname):
        self.fname = fname

    def update_iter_type(self, iter_type):
        self.iter_type = iter_type.lower()
        self.get_iteration()

    def update_iter_num(self, iter_num):
        if iter_num == '' or iter_num == 'last' or iter_num == -1:
            self.iter_num = 'last'
        elif type(iter_num) == str:
            self.iter_num = '{:03}'.format(int(iter_num))
        else:
            raise ValueError('cannot read iteration number')
        self.get_iteration()

    def get_iteration(self):
        self.iteration = '{}-{}'.format(self.iter_type.lower(), self.iter_num)
        print('Iteration: {}'.format(self.iteration))

    def update_data_type(self, data_type):
        self.data_type = data_type

    def update_atom(self, atom):
        self.atom = int(atom)

    def update_orbital(self, orbital):
        self.orbital = orbital

    def update_spin(self, spin):
        self.spin = spin

    def update_num_mats(self, num_mats):
        self.num_mats = num_mats

    def load_data(self):
        print(self.iteration)
        path_to_group = '{}/ineq-{:03}/siw-full/'.format(self.iteration, self.atom)
        path_to_value = path_to_group + 'value'
        path_to_error = path_to_group + 'error'
        print(path_to_group)
        f = h5py.File(self.fname, 'r')
        beta = f['.config'].attrs['general.beta']
        if self.spin == 'up':
            data = f[path_to_value][self.orbital, 0, self.orbital, 0]
            try:
                err = f[path_to_error][self.orbital, 0, self.orbital, 0]
            except:
                print('could not read error')
        elif self.spin == 'down':
            data = f[path_to_value][self.orbital, 1, self.orbital, 1]
            err = f[path_to_error][self.orbital, 1, self.orbital, 1]
        elif self.spin == 'average':
            data = 0.5 * (f[path_to_value][self.orbital, 0, self.orbital, 0]
                          + f[path_to_value][self.orbital, 1, self.orbital, 1])
            err = 0.5 * (f[path_to_error][self.orbital, 0, self.orbital, 0]
                          + f[path_to_error][self.orbital, 1, self.orbital, 1])
        f.close()
        niw = data.shape[0] // 2
        self.value = data[niw:niw + self.num_mats]
        self.error = err[niw:niw + self.num_mats]
        self.mats = np.pi / beta * (2. * np.arange(self.num_mats) + 1.)
        plt.plot(self.mats, self.value.real)
        plt.plot(self.mats, self.value.imag)
        plt.show()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.realgrid = RealFrequencyGrid(wmax=float(self.max_real_freq.text()),
                                          nw=int(self.num_real_freq.text()),
                                          type=str(self.grid_type_combo.currentText()))

        self.connect_realgrid_button()
        self.connect_input_button()
        self.connect_doit_button()
        self.connect_wmax()
        self.connect_nw()
        self.connect_grid_type()

        self.input_data = InputData(fname=str(self.inp_file_name.text()),
                                    iter_type=str(self.iteration_type_combo.currentText()),
                                    iter_num=str(self.iteration_number.text()),
                                    data_type=str(self.inp_data_type.currentText()),
                                    atom=str(self.atom_number.text()),
                                    orbital=str(self.orbital_number.text()),
                                    spin=str(self.spin_type_combo.currentText()),
                                    num_mats=str(self.num_mats_freq.text()))
        self.connect_fname_input()
        self.connect_data_type()
        self.connect_iteration_type()
        self.connect_iteration_number()
        self.connect_atom()
        self.connect_orbital()
        self.connect_spin()
        self.connect_num_mats()
        self.connect_load_button()

    def connect_realgrid_button(self):
        self.gen_real_grid_button.clicked.connect(lambda: self.realgrid.create_grid())

    def connect_input_button(self):
        self.load_data_button.clicked.connect(lambda: test_function('asdf'))

    def connect_doit_button(self):
        self.doit_button.clicked.connect(lambda: test_function('Do it!!!'))

    def connect_wmax(self):
        self.max_real_freq.returnPressed.connect(
            lambda: self.realgrid.update_wmax(float(self.max_real_freq.text())))
        self.max_real_freq.editingFinished.connect(
            lambda: self.realgrid.update_wmax(float(self.max_real_freq.text())))

    def connect_nw(self):
        self.num_real_freq.returnPressed.connect(
            lambda: self.realgrid.update_nw(int(self.num_real_freq.text())))
        self.num_real_freq.editingFinished.connect(
            lambda: self.realgrid.update_nw(int(self.num_real_freq.text())))

    def connect_grid_type(self):
        self.grid_type_combo.activated.connect(
            lambda: self.realgrid.update_type(str(self.grid_type_combo.currentText()))
        )

    def connect_fname_input(self):
        self.inp_file_name.editingFinished.connect(
            lambda: self.input_data.update_fname(str(self.inp_file_name.text())))

    def connect_data_type(self):
        self.inp_data_type.activated.connect(
            lambda: self.input_data.update_data_type(str(self.inp_data_type.currentText())))

    def connect_iteration_type(self):
        self.iteration_type_combo.activated.connect(
            lambda: self.input_data.update_iter_type(str(self.iteration_type_combo.currentText())))

    def connect_iteration_number(self):
        self.iteration_number.editingFinished.connect(
            lambda: self.input_data.update_iter_num(str(self.iteration_number.text())))

    def connect_atom(self):
        self.atom_number.editingFinished.connect(
            lambda: self.input_data.update_atom(int(self.atom_number.text())))

    def connect_orbital(self):
        self.orbital_number.editingFinished.connect(
            lambda: self.input_data.update_orbital(int(self.orbital_number.text())))

    def connect_spin(self):
        self.spin_type_combo.activated.connect(
            lambda: self.input_data.update_spin(str(self.spin_type_combo.currentText())))

    def connect_num_mats(self):
        self.num_mats_freq.editingFinished.connect(
            lambda: self.input_data.update_num_mats(int(self.num_mats_freq.text()))
        )

    def connect_load_button(self):
        self.load_data_button.clicked.connect(
            lambda: self.input_data.load_data()
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())