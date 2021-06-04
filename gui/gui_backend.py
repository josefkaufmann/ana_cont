import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import ana_cont.continuation as cont


class RealFrequencyGrid(object):
    """Class for real-frequency grids.

    Our real-frequency grids are always symmetric around 0.
    Thus they cover an interval [-wmax, wmax], where the endpoint is included.
    If the number of grid-points nw is an odd number, zero is included.
    We can create two types of grids: equispaced or non-equispaced (centered).
    The  latter is more dense in the low-frequency region, and more
    sparse in the high-frequency region.
    """

    def __init__(self, wmax=None, nw=None, type=None):
        """Initialize the real-frequency grid.

        wmax -- border of the real-frequency interval [-wmax, wmax]
        nw -- total number of real frequency grid points
        type -- grid type: 'equispaced grid' or 'centered grid'
        """
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
        """Create the real-frequency grid.

        There are four possible types of real-frequency grids.
        In each case, the endpoint is included in the grid.

        * An 'equispaced symmetric' grid is simply a linspace, containing also the endpoint.
        * The 'centered symmetric' grid is created by a tangent function,
          such that the grid points are denser around zero.
        * 'equispaced positive' is a simple linspace, starting from zero
          and containing only positive values.
        * 'centered positive' also starts from zero and contains only positive
          values, but close to zero they are lying denser.
        """
        if self.type == 'equispaced symmetric':
            self.grid = np.linspace(-self.wmax, self.wmax, num=self.nw, endpoint=True)
        elif self.type == 'centered symmetric':
            self.grid = self.wmax * np.tan(
                np.linspace(-np.pi / 2.1, np.pi / 2.1, num=self.nw, endpoint=True)) / np.tan(np.pi / 2.1)
        elif self.type == 'equispaced positive':
            self.grid = np.linspace(0., self.wmax, num=self.nw, endpoint=True)
        elif self.type == 'centered positive':
            self.grid = self.wmax * np.tan(
                np.linspace(0., np.pi / 2.1, num=self.nw, endpoint=True)) / np.tan(np.pi / 2.1)
        else:
            raise ValueError('Unknown real-frequency grid type.')
        print(self.grid)


def input_data_plot(mats, value, error, datatype, mom1=None, hartree=None):
    """Generate a very simple plot of the Matsubara data."""
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.errorbar(mats, value.real,
                yerr=error, label='real part')
    ax.errorbar(mats, value.imag,
                yerr=error, label='imaginary part')
    if mom1 is not None:
        asymptote = mom1/mats
        astartind = np.searchsorted(asymptote,
                                    np.amin(value.imag))
        ax.plot(mats[astartind:], asymptote[astartind:], ':',
                label='asymptotic imaginary part', zorder=np.inf)
    ax.set_title('Input data')
    ax.set_xlabel('Matsubara frequency')
    ax.secondary_xaxis('top',
                       functions=(
                           interp.interp1d(mats,
                                           np.arange(mats.size),
                                           fill_value="extrapolate"),
                           interp.interp1d(np.arange(mats.size),
                                           mats,
                                           fill_value="extrapolate")
                       )).set_xlabel('Index')
    ax.set_ylabel('{}'.format(datatype))
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    if datatype == "Self-energy":
        plt.text(xmin + 0.1 * (xmax - xmin), ymin + 0.9 * (ymax - ymin),
                 'Hartree energy = {:5.4f}'.format(hartree))
    plt.legend()
    plt.grid()
    plt.show()


class InputData(object):
    """Input data for the analytic continuation of a w2dynamics result."""

    def __init__(self, fname=None, iter_type=None, iter_num=None, data_type=None,
                 atom=None, orbital=None, spin=None, num_mats=None,
                 ignore_real_part=False):
        """Initialize the input data object.

        fname -- Name (str) of a valid w2dynamics DMFT output file.
            A w2dynamics version from 2020 or newer should be used.
        iter_type -- iteration type: 'dmft' or 'stat'
        iter_num -- iteration number: integer or 'last' (-1 also points to last)
        data_type -- "Green's function" or "Self-energy"
        atom -- which inequivalent atom (one-based integer), e.g. 1 leads to 'ineq-001'
        orbital -- which orbital component to load (one-based integer)
        spin -- which spin projection to load: 'up', 'down', 'average'
        num_mats -- number of Matsubara frequencies for continuation (integer)
        ignore_real_part -- if True, the real part is set to zero.
        """
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
            print('could not set num mats, using all available')
            self.num_mats = None

        if self.iter_type is None:
            self.iter_type = 'dmft'
        self.update_iter_num(self.iter_num)
        self.ignore_real_part = ignore_real_part

    def update_fname(self, fname):
        self.fname = fname

    def update_iter_type(self, iter_type):
        self.iter_type = iter_type.lower()
        self.get_iteration()

    def update_iter_num(self, iter_num):
        if iter_num is None or iter_num in ('', 'last', -1):
            self.iter_num = 'last'
        elif type(iter_num) == str or isinstance(iter_num, int):
            self.iter_num = '{:03}'.format(int(iter_num))
        else:
            raise ValueError('cannot read iteration number')
        self.get_iteration()

    def get_iteration(self):
        """Compose the whole iteration string, e.g. 'dmft-003'."""
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
        """Load Matsubara-frequency data.

        Load w2dynamics data on Matsubara frequency axis.
        This is either a self-energy, or a Green's function.
        For the self-energy, there are two different possible formats.
        """
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

        if self.ignore_real_part:
            self.value = 1j * self.value.imag

    def plot(self):
        input_data_plot(self.mats, self.value, self.error,
                        self.data_type, self.smom, self.hartree)

    def generate_mats_freq(self):
        """Generate the Matsubara frequency grid."""
        f = h5py.File(self.fname, 'r')
        beta = f['.config'].attrs['general.beta']
        if self.num_mats is None:
            self.num_mats = f['.config'].attrs['qmc.niw']
        f.close()
        self.mats = np.pi / beta * (2. * np.arange(self.num_mats) + 1.)

    def load_siw_1(self):
        """Load self-energy with jackknife error, type 1.

        In the first implementation, the self-energy with jackknife error
        was written in the group 'siw-full-jkerr', where the frequency
        axis is the first, followed by orbital and spin: (freq, orb, spin, orb, spin)
        The Hartree term is subtracted automatically.
        """
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
            self.smom = f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 1]
        elif self.spin == 'down':
            data = f[path_to_value][:, self.orbital - 1, 1, self.orbital - 1, 1]
            err = f[path_to_error][:, self.orbital - 1, 1, self.orbital - 1, 1]
            smom = f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 0]
            self.smom = f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 1]
        elif self.spin == 'average':
            data = 0.5 * (f[path_to_value][:, self.orbital - 1, 0, self.orbital - 1, 0]
                          + f[path_to_value][:, self.orbital - 1, 1, self.orbital - 1, 1])
            err = 1. / np.sqrt(2.) * (f[path_to_error][:, self.orbital - 1, 0, self.orbital - 1, 0]
                                     + f[path_to_error][:, self.orbital - 1, 1, self.orbital - 1, 1])
            smom = 0.5 * (f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 0]
                          + f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 0])
            self.smom = 0.5 * (f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 1]
                               + f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 1])
        f.close()
        niw = data.shape[0] // 2
        self.value = data[niw:niw + self.num_mats] - smom
        self.hartree = smom
        self.error = err[niw:niw + self.num_mats]

    def load_siw_2(self):
        """Load self-energy with jackknife error, type 2.

        In newer versions, the self-energy with jackknife error is stored
        in siw-full, in standard format (orb, spin, orb, spin, freq).
        If the QMC error is not present, it is set to a constant value err_const.
        The Hartree term is subtracted automatically.
        """
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
            self.smom = f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 1]
            try:
                err = f[path_to_error][self.orbital - 1, 0, self.orbital - 1, 0]
            except KeyError:
                print('Warning: No self-energy error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        elif self.spin == 'down':
            data = f[path_to_value][self.orbital - 1, 1, self.orbital - 1, 1]
            smom = f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 0]
            self.smom = f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 1]
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
            self.smom = 0.5 * (f[path_to_smom][self.orbital - 1, 0, self.orbital - 1, 0, 1]
                               + f[path_to_smom][self.orbital - 1, 1, self.orbital - 1, 1, 1])
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
        """Load Matsubara Green's function data.

        Here we have only one possible format, i.e. (orb, spin, orb, spin, freq).
        If no QMC error is present, we set it to a constant value err_const.
        """
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
                print('Warning: No Green\'s function error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        elif self.spin == 'down':
            data = f[path_to_value][self.orbital - 1, 1, self.orbital - 1, 1]
            try:
                err = f[path_to_error][self.orbital - 1, 1, self.orbital - 1, 1]
            except KeyError:
                print('Warning: No Green\'s function error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        elif self.spin == 'average':
            data = 0.5 * (f[path_to_value][self.orbital - 1, 0, self.orbital - 1, 0]
                          + f[path_to_value][self.orbital - 1, 1, self.orbital - 1, 1])
            try:
                err = 1. / np.sqrt(2.) * (f[path_to_error][self.orbital - 1, 0, self.orbital - 1, 0]
                                          + f[path_to_error][self.orbital - 1, 1, self.orbital - 1, 1])
            except KeyError:
                print('Warning: No Green\'s function error found; setting to constant {}'.format(err_const))
                err = np.ones_like(data) * err_const
        f.close()
        niw = data.shape[0] // 2
        self.value = data[niw:niw + self.num_mats]
        self.hartree = None
        self.smom = -1.
        self.error = err[niw:niw + self.num_mats]

class TextInputData(object):
    """Input data for analytic continuation, from a generic text file."""

    def __init__(self, fname=None, data_type=None,
                 num_mats=None, n_skip=None):
        """Initialize the text file input.

        fname -- file name of the text file
        data_type -- either "Green's function" or "Self-energy"
        num_mats -- number of Matsubara frequencies
        n_skip -- number of lines to skip at the beginning of the text file.

        Note the following points:
        * The Hartree term has to be subtracted beforehand,
            i.e. both the real and imaginary part of the data in the input file
            have to approach zero in the high-frequency limit,
        * The input file must contain only data on the positive half of the Matsubara axis,
        * data_type will not have any impact on the analytic continuation itself,
            but if it is a self-energy, the real part is calculated by Kramers-Kronig
            after the continuation, and the full self-energy is stored.
        * The Hartree term has to be handeled completely manually in pre- and postprocessing.
        """
        self.fname = fname
        self.data_type = data_type
        try:
            self.num_mats = int(num_mats)
        except ValueError:
            print('inferring number of Matsubaras from data')
            self.num_mats = None
        try:
            self.n_skip = int(n_skip)
        except ValueError:
            print('will not skip any lines')
            self.n_skip = 0

        self.atom = ''
        self.orbital = ''
        self.spin = ''

    def update_fname(self, fname):
        self.fname = fname

    def update_data_type(self, data_type):
        self.data_type = data_type

    def update_n_skip(self, n_skip):
        self.n_skip = n_skip

    def read_data(self):
        """Read the text file by np.loadtxt."""
        if self.data_type == 'bosonic':
            mats, val_re, err = np.loadtxt(self.fname, skiprows=self.n_skip, unpack=True)
            val_im = np.zeros_like(val_re)
        else:
            mats, val_re, val_im, err = np.loadtxt(self.fname, skiprows=self.n_skip, unpack=True)
        n_mats_data = mats.shape[0]
        if self.num_mats is None:
            self.num_mats = n_mats_data
        self.mats = mats[:self.num_mats]
        self.value = (val_re + 1j * val_im)[:self.num_mats]
        self.error = err[:self.num_mats]
        self.hartree = 0.

    def plot(self):
        input_data_plot(self.mats, self.value, self.error, self.data_type,
                        (-1 if self.data_type == "Green's function" else None),
                        self.hartree)


class OutputData(object):
    """Output data of the analytic continuation."""

    def __init__(self):
        pass

    def update_fname(self, fname):
        self.fname = fname
        print('Update output file name to: {}'.format(self.fname))

    def update(self, w, spec, input_data):
        """Update the output data with the latest input and results."""
        self.w = w
        self.spec = spec
        self.input_data = input_data

    def save(self, interpolate=False, n_reg=None):
        """Save output data to text file.

        Here we have to distinguish several different cases.
        * For a self-energy, we first do a Kramers-Kronig transformation
            to get also the real part.
            Then the Hartree term is added. Remember: when reading input data from a text file,
            the Hartree term has to be handeled manually, the program treats it as zero.
            The whole function, i.e. real and imaginary part of the self-energy, is saved.
        * In case of a Green's function, we save the spectral function,
            i.e. -Im[G(omega)] / pi. The real part is not computed.
        * If desired, an interpolation to a regular-spaced grid is done.

        Output format: text file.
        For spectral function: two colums: frequency, spectrum
        For self-energy: three columns: frequency, real part, imaginary part.
        """
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

        elif (self.input_data.data_type == "Green's function"
              or self.input_data.data_type == "bosonic"):
            if interpolate:
                spec_interp = self.interpolate(self.w, self.spec, n_reg)
                np.savetxt(self.fname, np.vstack((self.w_reg, spec_interp)).T)
            else:
                np.savetxt(self.fname, np.vstack((self.w, self.spec)).T)
        else:
            print("Unknown data type {}, don't know how to save.".format(self.input_data.data_type))


    def interpolate(self, original_grid, original_function, n_reg):
        """Spline interpolation of real-frequency data."""
        self.w_reg = np.linspace(np.amin(self.w), np.amax(self.w), num=n_reg, endpoint=True)
        interp_function = interp.InterpolatedUnivariateSpline(original_grid, original_function)(self.w_reg)
        return interp_function
