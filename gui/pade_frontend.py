import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

import ana_cont.continuation as cont
from gui.pade_ui import Ui_MainWindow
from gui.gui_backend import RealFrequencyGrid, InputData, TextInputData, OutputData


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """The Main Window of the graphical user interface.

    The class MainWindow inherits from Ui_MainWindow, which is
    defined in pade_ui.py. The latter file is autogenerated
    by pyuic from pade_ui.ui [`pyuic5 pade_ui.ui -o pade_ui.py`]
    The ui file can be edited by the QtDesigner.
    """

    def __init__(self, *args, obj=None, **kwargs):
        """Connect the widgets, instantiate the main classes."""
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # real-frequency grid
        self.realgrid = RealFrequencyGrid(wmax=float(self.max_real_freq.text()),
                                          nw=int(self.num_real_freq.text()),
                                          type=str(self.grid_type_combo.currentText()))
        self.connect_realgrid_button()
        self.connect_wmax()
        self.connect_nw()
        self.connect_grid_type()

        # input data
        self.input_data = InputData(fname=str(self.inp_file_name.text()),
                                    iter_type=str(self.iteration_type_combo.currentText()),
                                    iter_num=str(self.iteration_number.text()),
                                    data_type=str(self.inp_data_type.currentText()),
                                    atom=str(self.atom_number.text()),
                                    orbital=str(self.orbital_number.text()),
                                    spin=str(self.spin_type_combo.currentText()),
                                    num_mats=str(self.num_mats_freq.text()))
        self.connect_select_button()
        self.connect_load_button()
        self.connect_show_button()
        self.connect_load_button_text()
        self.connect_show_button_2()
        self.connect_select_button_2()

        self.connect_doit_button()

        # output data
        self.output_data = OutputData()
        self.connect_select_output_button()
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
                    'Open file', os.getcwd(), "HDF5 files (*.hdf5)")[0])

    def connect_select_button(self):
        self.select_file_button.clicked.connect(self.get_fname)

    def get_fname_text(self):
        self.inp_file_name_2.setText(
            QtWidgets.QFileDialog.getOpenFileName(self,
                    'Open file', os.getcwd(), "text files (*.dat *.txt)")[0])

    def connect_select_button_2(self):
        self.select_file_button_2.clicked.connect(self.get_fname_text)

    def connect_show_button(self):
        self.show_data_button.clicked.connect(
            lambda: self.input_data.plot()
        )

    def connect_show_button_2(self):
        self.show_data_button_2.clicked.connect(
            lambda: self.input_data.plot()
        )

    def load_w2dynamics_data(self):
        self.input_data = InputData(fname=str(self.inp_file_name.text()),
                                    iter_type=str(self.iteration_type_combo.currentText()),
                                    iter_num=str(self.iteration_number.text()),
                                    data_type=str(self.inp_data_type.currentText()),
                                    atom=str(self.atom_number.text()),
                                    orbital=str(self.orbital_number.text()),
                                    spin=str(self.spin_type_combo.currentText()),
                                    num_mats=str(self.num_mats_freq.text()),
                                    ignore_real_part=self.ignore_checkbox.isChecked())
        self.input_data.load_data()

    def connect_load_button(self):
        self.load_data_button.clicked.connect(self.load_w2dynamics_data)

    def load_text_data(self):
        self.input_data = TextInputData(fname=str(self.inp_file_name_2.text()),
                                        data_type=str(self.inp_data_type_text.currentText()),
                                        n_skip=str(self.n_skip.text()),
                                        num_mats=str(self.num_mats_freq_text.text()))
        self.input_data.read_data()

    def connect_load_button_text(self):
        self.load_data_button_2.clicked.connect(self.load_text_data)

    def parse_mats_ind(self):
        mats_ind_str = self.mats_ind_inp.text()
        mats_list_str = [part.strip() for part in mats_ind_str.split(',')]
        if '' in mats_list_str:
            mats_list_str.remove('')

        mats_ind = np.array([int(ind) for ind in mats_list_str])
        print(mats_ind)
        return mats_ind


    def main_function(self):
        """Main function for the analytic continuation procedure.

        This function is called when the "Do it" button is clicked.
        It performs an analytical continuation for the present settings
        and shows a plot.
        """

        mats_ind = self.parse_mats_ind()
        self.ana_cont_probl = cont.AnalyticContinuationProblem(im_axis=self.input_data.mats[mats_ind],
                                                               im_data=self.input_data.value[mats_ind],
                                                               re_axis=self.realgrid.grid,
                                                               kernel_mode='freq_fermionic')

        sol = self.ana_cont_probl.solve(method='pade')
        check_axis = np.linspace(0., 1.25 * self.input_data.mats[mats_ind[-1]], num=500)
        check = self.ana_cont_probl.solver.check(im_axis_fine=check_axis)

        self.output_data.update(self.realgrid.grid, sol.A_opt, self.input_data)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(11.75, 8.25))  # A4 paper size


        ax[0, 0].plot(self.realgrid.grid, sol.A_opt)
        ax[0, 0].set_xlabel(r'$\omega$')
        ax[0, 0].set_ylabel('spectrum')

        ax[0, 1].plot(self.input_data.mats[mats_ind], self.input_data.value.real[mats_ind],
                      color='red', ls='None', marker='.', markersize=12, alpha=0.33,
                      label='Re[selected data]')
        ax[0, 1].plot(self.input_data.mats[mats_ind], self.input_data.value.imag[mats_ind],
                      color='red', ls='None', marker='.', markersize=12, alpha=0.33,
                      label='Im[selected data]')
        ax[0, 1].plot(self.input_data.mats, self.input_data.value.real,
                      color='blue', ls=':', marker='x', markersize=5,
                      label='Re[full data]')
        ax[0, 1].plot(self.input_data.mats, self.input_data.value.imag,
                      color='green', ls=':', marker='+', markersize=5,
                      label='Im[full data]')

        ax[1, 0].plot(self.input_data.mats[mats_ind], self.input_data.value.real[mats_ind],
                      color='red', ls='None', marker='.', markersize=12, alpha=0.33,
                      label='Re[selected data]')
        ax[1, 0].plot(self.input_data.mats[mats_ind], self.input_data.value.imag[mats_ind],
                      color='red', ls='None', marker='.', markersize=12, alpha=0.33,
                      label='Im[selected data]')
        # ax[1, 0].plot(self.input_data.mats, self.input_data.value.real,
        #               color='blue', ls=':', marker='x', markersize=5,
        #               label='Re[full data]')
        # ax[1, 0].plot(self.input_data.mats, self.input_data.value.imag,
        #               color='green', ls=':', marker='+', markersize=5,
        #               label='Im[full data]')
        ax[1, 0].plot(check_axis, check.real,
                      ls='--', color='gray', label='Re[Pade interpolation]')
        ax[1, 0].plot(check_axis, check.imag,
                      color='gray', label='Im[Pade interpolation]')
        ax[1, 0].set_xlabel(r'$\nu_n$')
        ax[1, 0].set_ylabel(self.input_data.data_type)
        ax[1, 0].legend()
        ax[1, 0].set_xlim(0., 1.05 * check_axis[-1])

        # ax[1, 1].plot(self.input_data.mats, (self.input_data.value - sol[0].backtransform).real,
        #               ls='--', label='real part')
        # ax[1, 1].plot(self.input_data.mats, (self.input_data.value - sol[0].backtransform).imag,
        #               label='imaginary part')
        # ax[1, 1].set_xlabel(r'$\nu_n$')
        # ax[1, 1].set_ylabel('data $-$ fit')
        # ax[1, 1].legend()
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

    def save_output(self):
        fname_out = str(self.out_file_name.text())
        if fname_out == '':
            print('Error in saving: First you have to specify the output file name.')
            return 1

        self.output_data.update_fname(fname_out)
        self.output_data.save(interpolate=False)

    def connect_save_button(self):
        self.save_button.clicked.connect(lambda: self.save_output())