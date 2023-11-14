import numpy, h5py, pylab, pickle
import unittest
from . import mpi_launch, get_dataset
from circus.shared.utils import *
from circus.shared.parser import CircusParser

def get_performance(file_name, name):

    a, b            = os.path.splitext(os.path.basename(file_name))
    file_name, ext  = os.path.splitext(file_name)
    file_out        = os.path.join(os.path.abspath(file_name), a)
    result_name     = os.path.join(file_name, 'injected')

    pic_name        = file_name + '.pic'
    data            = pickle.load(open(pic_name))
    n_cells         = data['cells'] 
    n_point         = int(numpy.sqrt(len(n_cells)))
    amplitude       = data['amplitudes'][0:n_point]
    rate            = data['rates'][::n_point]
    sampling        = data['sampling']
    probe_file      = data['probe']
    sim_templates   = 1

    temp_file       = file_out + '.templates.hdf5'
    temp_x          = h5py.File(temp_file).get('temp_x')[:]
    temp_y          = h5py.File(temp_file).get('temp_y')[:]
    temp_data       = h5py.File(temp_file).get('temp_data')[:]
    temp_shape      = h5py.File(temp_file).get('temp_shape')[:]
    templates       = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(temp_shape[0]*temp_shape[1], temp_shape[2]))

    temp_file       = os.path.join(result_name, '%s.templates.hdf5' %a)
    temp_x          = h5py.File(temp_file).get('temp_x')[:]
    temp_y          = h5py.File(temp_file).get('temp_y')[:]
    temp_data       = h5py.File(temp_file).get('temp_data')[:]
    temp_shape      = h5py.File(temp_file).get('temp_shape')[:]
    inj_templates   = scipy.sparse.csc_matrix((temp_data, (temp_x, temp_y)), shape=(temp_shape[0]*temp_shape[1], temp_shape[2]))

    amplitudes      = h5py.File(file_out + '.templates.hdf5').get('limits')[:]

    n_tm            = inj_templates.shape[1]//2
    res             = numpy.zeros(len(n_cells))
    res2            = numpy.zeros(len(n_cells))
    res3            = numpy.zeros(len(n_cells))

    for gcount, temp_id in enumerate(range(n_tm - len(n_cells), n_tm)):
        source_temp = inj_templates[:, temp_id].toarray().flatten()
        similarity  = []
        temp_match  = None
        dmax        = 0
        for i in range(templates.shape[1]//2):
            d = numpy.corrcoef(templates[:, i].toarray().flatten(), source_temp)[0, 1]
            similarity += [d]
            if d >= dmax:
                temp_match = i
                dmax       = d
        res[gcount]  = numpy.max(similarity)
        res2[gcount] = numpy.sum(numpy.array(similarity) >= sim_templates)
        res3[gcount] = temp_match

    pylab.figure()


    pylab.subplot(121)
    pylab.imshow(res.reshape(n_point, n_point), aspect='auto', interpolation='nearest', origin='lower')
    cb = pylab.colorbar()
    cb.set_label('Correlation')
    pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(rate, 1))
    pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(amplitude, 1))
    pylab.ylabel('Rate [Hz]')
    pylab.xlabel('Relative Amplitude')
    pylab.xlim(-0.5, n_point-0.5)
    pylab.ylim(-0.5, n_point-0.5)

    pylab.subplot(122)
    pylab.imshow(res2.reshape(n_point, n_point).astype(numpy.int32), aspect='auto', interpolation='nearest', origin='lower')
    cb = pylab.colorbar()
    cb.set_label('Number of templates')
    pylab.yticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(rate, 1))
    pylab.xticks(numpy.linspace(0.5, n_point-0.5, 5), numpy.round(amplitude, 1))
    pylab.ylabel('Rate [Hz]')
    pylab.xlabel('Relative Amplitude')
    pylab.xlim(-0.5, n_point-0.5)
    pylab.ylim(-0.5, n_point-0.5)

    pylab.tight_layout()

    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    plot_path = os.path.join(plot_path, 'plots')
    plot_path = os.path.join(plot_path, 'clustering')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    output = os.path.join(plot_path, '%s.pdf' %name)
    pylab.savefig(output)
    return templates, res2


class TestClustering(unittest.TestCase):

    def setUp(self):
        self.all_matches    = None
        self.all_templates  = None
        dirname             = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
        self.path           = os.path.join(dirname, 'synthetic')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name      = os.path.join(self.path, 'clustering.dat')
        self.source_dataset = get_dataset(self)
        if not os.path.exists(self.file_name):
            mpi_launch('benchmarking', self.source_dataset, 2, 0, 'False', self.file_name, 'clustering', 1)
            mpi_launch('whitening', self.file_name, 2, 0, 'False')

        self.parser = CircusParser(self.file_name)
        self.parser.write('clustering', 'max_elts', '1000')

    def test_clustering_one_CPU(self):
        mpi_launch('clustering', self.file_name, 1, 0, 'False')
        res = get_performance(self.file_name, 'one_CPU')
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]


    def test_clustering_two_CPU(self):
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        res = get_performance(self.file_name, 'two_CPU')
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]

    def test_clustering_pca(self):
        self.parser.write('clustering', 'extraction', 'median-pca')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        self.parser.write('clustering', 'extraction', 'median-raw')
        res = get_performance(self.file_name, 'median-pca')
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]

    def test_clustering_nb_passes(self):
        self.parser.write('clustering', 'nb_repeats', '1')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        self.parser.write('clustering', 'nb_repeats', '3')
        res = get_performance(self.file_name, 'nb_passes')
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]

    def test_clustering_sim_same_elec(self):
        self.parser.write('clustering', 'sim_same_elec', '5')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        self.parser.write('clustering', 'sim_same_elec', '3')
        res = get_performance(self.file_name, 'sim_same_elec')
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]

    def test_clustering_cc_merge(self):
        self.parser.write('clustering', 'cc_merge', '0.8')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        self.parser.write('clustering', 'cc_merge', '0.95')
        res = get_performance(self.file_name, 'cc_merge')
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]

    def test_remove_mixtures(self):
        self.parser.write('clustering', 'remove_mixtures', 'False')
        mpi_launch('clustering', self.file_name, 2, 0, 'False')
        self.parser.write('clustering', 'remove_mixtures', 'True')
        res = get_performance(self.file_name, 'cc_merge')
        if self.all_templates is None:
            self.all_templates = res[0]
            self.all_matches   = res[1]