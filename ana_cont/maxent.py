import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from PyQt5 import QtCore, QtGui, QtWidgets

sys.path.insert(0, '/home/josef/ana_cont_gui')
import ana_cont.continuation as cont
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
        if self.type == 'equispaced grid':
            self.grid = np.linspace(-self.wmax, self.wmax, num=self.nw, endpoint=True)
        elif self.type == 'centered grid':
            self.grid = self.wmax * np.tan(np.linspace(-np.pi / 2.1, np.pi / 2.1, num=self.nw)) / np.tan(np.pi / 2.1)
        print(self)
        print(self.grid)

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
            self.orbital = int(orbital)
        except ValueError:
            print('could not set orbital')

        self.spin = spin

        try:
            self.num_mats = int(num_mats)
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
        if self.data_type == "Self-energy":
            pass
        elif self.data_type == "Green's function":
            pass
        else:
            raise ValueError('Unknown data type to read. Must be \'Self-energy\' or \'Green\'s function\'')

    def update_atom(self, atom):
        self.atom = int(atom)

    def update_orbital(self, orbital):
        self.orbital = int(orbital)

    def update_spin(self, spin):
        self.spin = spin

    def update_num_mats(self, num_mats):
        self.num_mats = int(num_mats)

    def load_data(self):
        self.generate_mats_freq()

        if self.data_type == "Self-energy":
            print('Try to load self-energy')
            f = h5py.File(self.fname, 'r')
            path_to_atom = '{}/ineq-{:03}/'.format(self.iteration, self.atom)
            all_keys = list(f[path_to_atom].keys())
            print(all_keys)
            f.close()
            if 'siw-full-jkerr' in all_keys:
                print('load siw 1')
                self.load_siw_1()
            elif 'siw-full' in all_keys:
                print('load siw 2')
                self.load_siw_2()
        elif self.data_type == "Green's function":
            self.load_giw()
        else:
            raise ValueError(
                'Unknown data type (Must be either \'Self-energy\' or \'Green\'s function\')')

    def plot(self):
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.plot(self.mats, self.value.real, label='real part')
        ax.plot(self.mats, self.value.imag, label='imaginary part')
        ax.set_title('Input data')
        ax.set_xlabel('Matsubara frequency')
        ax.set_ylabel('{}'.format(self.data_type))
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()

        if self.data_type == "Self-energy":
            plt.text(xmin + 0.1 * (xmax - xmin), ymin + 0.9 * (ymax - ymin),
                     'Hartree energy = {:5.4f}'.format(self.hartree))
        plt.legend()
        plt.show()

    def generate_mats_freq(self):
        f = h5py.File(self.fname, 'r')
        beta = f['.config'].attrs['general.beta']
        f.close()
        self.mats = np.pi / beta * (2. * np.arange(self.num_mats) + 1.)

    def load_siw_1(self):
        path_to_atom = '{}/ineq-{:03}/'.format(self.iteration, self.atom)
        path_to_group = '{}/siw-full-jkerr/'.format(path_to_atom)
        path_to_value = path_to_group + 'value'
        path_to_error = path_to_group + 'error'
        path_to_smom = path_to_atom + 'smom-full/value'
        f = h5py.File(self.fname, 'r')
        if self.spin == 'up':
            data = f[path_to_value][:, self.orbital - 1, 0, self.orbital - 1, 0]
            err = f[path_to_error][:, self.orbital - 1, 0, self.orbital - 1, 0]
            smom = f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 0]
        elif self.spin == 'down':
            data = f[path_to_value][:, self.orbital - 1, 1, self.orbital - 1, 1]
            err = f[path_to_error][:, self.orbital - 1, 1, self.orbital - 1, 1]
            smom = f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 0]
        elif self.spin == 'average':
            data = 0.5 * (f[path_to_value][:, self.orbital - 1, 0, self.orbital - 1, 0]
                          + f[path_to_value][:, self.orbital - 1, 1, self.orbital - 1, 1])
            err = 1. / np.sqrt(2.) * (f[path_to_error][:, self.orbital - 1, 0, self.orbital - 1, 0]
                                     + f[path_to_error][:, self.orbital - 1, 1, self.orbital - 1, 1])
            smom = 0.5 * (f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 0]
                          + f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 0])
        f.close()
        niw = data.shape[0] // 2
        self.value = data[niw:niw + self.num_mats] - smom
        self.hartree = smom
        self.error = err[niw:niw + self.num_mats]

    def load_siw_2(self):
        path_to_atom = '{}/ineq-{:03}/'.format(self.iteration, self.atom)
        path_to_group = '{}/siw-full/'.format(path_to_atom)
        path_to_value = path_to_group + 'value'
        path_to_error = path_to_group + 'error'
        path_to_smom = path_to_atom + 'smom-full/value'
        err_const = 0.001
        f = h5py.File(self.fname, 'r')
        if self.spin == 'up':
            data = f[path_to_value][self.orbital - 1, 0, self.orbital - 1, 0]
            smom = f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 0]
            try:
                err = f[path_to_error][self.orbital - 1, 0, self.orbital - 1, 0]
            except KeyError:
                print('Warning: No self-energy error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        elif self.spin == 'down':
            data = f[path_to_value][self.orbital - 1, 1, self.orbital - 1, 1]
            smom = f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 0]
            try:
                err = f[path_to_error][self.orbital - 1, 1, self.orbital - 1, 1]
            except KeyError:
                print('Warning: No self-energy error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        elif self.spin == 'average':
            data = 0.5 * (f[path_to_value][self.orbital - 1, 0, self.orbital - 1, 0]
                          + f[path_to_value][self.orbital - 1, 1, self.orbital - 1, 1])
            smom = 0.5 * (f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 0]
                          + f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 0])
            try:
                err = 1. / np.sqrt(2.) * (f[path_to_error][self.orbital - 1, 0, self.orbital - 1, 0]
                                          + f[path_to_error][self.orbital - 1, 1, self.orbital - 1, 1])
            except KeyError:
                print('Warning: No self-energy error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        f.close()
        niw = data.shape[0] // 2
        self.value = data[niw:niw + self.num_mats] - smom
        self.hartree = smom
        self.error = err[niw:niw + self.num_mats]

    def load_giw(self):
        path_to_atom = '{}/ineq-{:03}/'.format(self.iteration, self.atom)
        path_to_group = '{}/giw-full/'.format(path_to_atom)
        path_to_value = path_to_group + 'value'
        path_to_error = path_to_group + 'error'

        err_const = 0.001
        f = h5py.File(self.fname, 'r')
        if self.spin == 'up':
            data = f[path_to_value][self.orbital - 1, 0, self.orbital - 1, 0]
            try:
                err = f[path_to_error][self.orbital - 1, 0, self.orbital - 1, 0]
            except KeyError:
                print('Warning: No self-energy error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        elif self.spin == 'down':
            data = f[path_to_value][self.orbital - 1, 1, self.orbital - 1, 1]
            try:
                err = f[path_to_error][self.orbital - 1, 1, self.orbital - 1, 1]
            except KeyError:
                print('Warning: No self-energy error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        elif self.spin == 'average':
            data = 0.5 * (f[path_to_value][self.orbital - 1, 0, self.orbital - 1, 0]
                          + f[path_to_value][self.orbital - 1, 1, self.orbital - 1, 1])
            try:
                err = 1. / np.sqrt(2.) * (f[path_to_error][self.orbital - 1, 0, self.orbital - 1, 0]
                                          + f[path_to_error][self.orbital - 1, 1, self.orbital - 1, 1])
            except KeyError:
                print('Warning: No self-energy error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        f.close()
        niw = data.shape[0] // 2
        self.value = data[niw:niw + self.num_mats]
        self.hartree = None
        self.error = err[niw:niw + self.num_mats]

class OutputData(object):
    def __init__(self):
        pass

    def update_fname(self, fname):
        self.fname = fname
        print('Update output file name to: {}'.format(self.fname))

    def update(self, w, spec, input_data):
        self.w = w
        self.spec = spec
        self.input_data = input_data

    def save(self, interpolate=False, n_reg=None):
        if self.input_data.data_type == "Self-energy":
            self.self_energy = cont.GreensFunction(spectrum=self.spec, wgrid=self.w, kind='fermionic').kkt()
            if interpolate:
                sw_real = self.interpolate(self.w, self.self_energy.real, n_reg)
                sw_imag = self.interpolate(self.w, self.self_energy.imag, n_reg)
                np.savetxt(self.fname, np.vstack((self.w_reg,
                                                  sw_real + self.input_data.hartree,
                                                  sw_imag)).T)
            else:
                np.savetxt(self.fname, np.vstack((self.w,
                                                  self.self_energy.real + self.input_data.hartree,
                                                  self.self_energy.imag)).T)

        elif self.input_data.data_type == "Green's function":
            if interpolate:
                spec_interp = self.interpolate(self.w, self.spec, n_reg)
                np.savetxt(self.fname, np.vstack((self.w_reg, spec_interp)).T)
            else:
                np.savetxt(self.fname, np.vstack((self.w, self.spec)).T)

    def interpolate(self, original_grid, original_function, n_reg):
        self.w_reg = np.linspace(np.amin(self.w), np.amax(self.w), num=n_reg, endpoint=True)
        interp_function = interp.InterpolatedUnivariateSpline(original_grid, original_function)(self.w_reg)
        return interp_function


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.file_dialog = QtWidgets.QFileDialog()

        self.realgrid = RealFrequencyGrid(wmax=float(self.max_real_freq.text()),
                                          nw=int(self.num_real_freq.text()),
                                          type=str(self.grid_type_combo.currentText()))

        self.connect_realgrid_button()

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
        self.connect_select_button()
        self.connect_data_type()
        self.connect_iteration_type()
        self.connect_iteration_number()
        self.connect_atom()
        self.connect_orbital()
        self.connect_spin()
        self.connect_num_mats()
        self.connect_load_button()
        self.connect_show_button()

        self.text_output.setReadOnly(True)
        self.connect_doit_button()

        self.output_data = OutputData()
        self.connect_select_output_button()
        self.connect_fname_output()
        self.connect_save_button()

    def connect_realgrid_button(self):
        self.gen_real_grid_button.clicked.connect(lambda: self.realgrid.create_grid())

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
        self.inp_file_name.textChanged.connect(
            lambda: self.input_data.update_fname(str(self.inp_file_name.text())))

    def get_fname(self):
        self.inp_file_name.setText(
            QtWidgets.QFileDialog.getOpenFileName(self,
                    'Open file', os.path.expanduser("~"), "HDF5 files (*.hdf5)")[0])

    def connect_select_button(self):
        self.select_file_button.clicked.connect(self.get_fname)

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

    def connect_show_button(self):
        self.show_data_button.clicked.connect(
            lambda: self.input_data.plot()
        )

    def connect_load_button(self):
        self.load_data_button.clicked.connect(
            lambda: self.input_data.load_data()
        )

    def main_function(self):
        self.ana_cont_probl = cont.AnalyticContinuationProblem(im_axis=self.input_data.mats,
                                                               im_data=self.input_data.value,
                                                               re_axis=self.realgrid.grid,
                                                               kernel_mode='freq_fermionic')
        model = np.ones_like(self.realgrid.grid)
        model /= np.trapz(model, self.realgrid.grid)

        preblur = self.preblur_button.isChecked()
        bw = float(self.blur_width.text()) if preblur else 0.

        sol = self.ana_cont_probl.solve(method='maxent_svd',
                                        optimizer='newton',
                                        alpha_determination='chi2kink',
                                        model=model,
                                        stdev=self.input_data.error,
                                        interactive=False, alpha_start=1e10, alpha_end=1e-3,
                                        preblur=preblur, blur_width=bw)

        inp_str = 'atom {}, orb {}, spin {}, blur {}: '.format(self.input_data.atom,
                                                               self.input_data.orbital,
                                                               self.input_data.spin,
                                                               bw)
        res_str = 'alpha_opt={:3.2f}, chi2(alpha_opt)={:3.2f}, min(chi2)={:3.2}'.format(
            sol[0].alpha, sol[0].chi2, sol[1][-1].chi2
        )
        self.text_output.append(inp_str + res_str)
        alphas = [s.alpha for s in sol[1]]
        chis = [s.chi2 for s in sol[1]]

        self.output_data.update(self.realgrid.grid, sol[0].A_opt, self.input_data)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(11.75, 8.25))  # A4 paper size
        ax[0, 0].loglog(alphas, chis, marker='s', color='black')
        ax[0, 0].loglog(sol[0].alpha, sol[0].chi2, marker='*', color='red', markersize=15)
        ax[0, 0].set_xlabel(r'$\alpha$')
        ax[0, 0].set_ylabel(r'$\chi^2(\alpha)$')

        ax[1, 0].plot(self.realgrid.grid, sol[0].A_opt)
        ax[1, 0].set_xlabel(r'$\omega$')
        ax[1, 0].set_ylabel('spectrum')

        ax[0, 1].plot(self.input_data.mats, self.input_data.value.real,
                      color='blue', ls=':', marker='x', markersize=5,
                      label='Re[data]')
        ax[0, 1].plot(self.input_data.mats, self.input_data.value.imag,
                      color='green', ls=':', marker='+', markersize=5,
                      label='Im[data]')
        ax[0, 1].plot(self.input_data.mats, sol[0].backtransform.real,
                      ls='--', color='gray', label='Re[fit]')
        ax[0, 1].plot(self.input_data.mats, sol[0].backtransform.imag,
                      color='gray', label='Im[fit]')
        ax[0, 1].set_xlabel(r'$\nu_n$')
        ax[0, 1].set_ylabel(self.input_data.data_type)
        ax[0, 1].legend()

        ax[1, 1].plot(self.input_data.mats, (self.input_data.value - sol[0].backtransform).real,
                      ls='--', label='real part')
        ax[1, 1].plot(self.input_data.mats, (self.input_data.value - sol[0].backtransform).imag,
                      label='imaginary part')
        ax[1, 1].set_xlabel(r'$\nu_n$')
        ax[1, 1].set_ylabel('data $-$ fit')
        ax[1, 1].legend()
        plt.tight_layout()
        plt.show()

    def connect_doit_button(self):
        self.doit_button.clicked.connect(lambda: self.main_function())

    def connect_fname_output(self):
        self.out_file_name.editingFinished.connect(
            lambda: self.output_data.update_fname(str(self.out_file_name.text())))
        self.inp_file_name.textChanged.connect(
            lambda: self.output_data.update_fname(str(self.out_file_name.text())))

    def get_fname_output(self):
        fname_out = QtWidgets.QFileDialog.getSaveFileName(self,
                    'Save as', '/'.join(self.input_data.fname.split('/')[:-1]), "DAT files (*.dat)")[0]
        self.out_file_name.setText(fname_out)
        self.output_data.update_fname(fname_out)

    def connect_select_output_button(self):
        self.output_directory_button.clicked.connect(self.get_fname_output)

    def connect_save_button(self):
        self.save_button.clicked.connect(
            lambda: self.output_data.save(interpolate=self.interpolate_button.isChecked(),
                                          n_reg=int(self.n_interpolation.text())))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())