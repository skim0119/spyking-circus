import numpy
import re
import sys
import logging
from circus.shared.messages import print_and_log
from .hdf5 import H5File

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


logger = logging.getLogger(__name__)


class KwdFile(H5File):

    description = "kwd"
    extension = [".kwd"]
    parallel_write = h5py.get_config().mpi
    is_streamable = ['multi-files', 'single-file']

    _required_fields = {
        'sampling_rate': float
    }

    _default_values = {
        'recording_number': 0,
        'dtype_offset': 'auto',
        'gain': 1.0
    }

    def set_streams(self, stream_mode):

        if stream_mode == 'single-file':

            sources = []
            to_write = []
            count = 0
            params = self.get_description()
            my_file = h5py.File(self.file_name, mode='r')
            all_matches = list(my_file.get('recordings').keys())
            all_streams = []
            for m in all_matches:
                all_streams += [int(m)]

            idx = numpy.argsort(all_streams)

            for count in range(len(all_streams)):
                params['recording_number'] = all_streams[idx[count]]
                new_data = type(self)(self.file_name, params)
                sources += [new_data]
                to_write += ['We found the datafile %s with t_start %d and duration %d' % (new_data.file_name, new_data.t_start, new_data.duration)]

            print_and_log(to_write, 'debug', logger)

            return sources

        elif stream_mode == 'multi-files':
            return H5File.set_streams(stream_mode)

    def _read_from_header(self):

        self.params['h5_key'] = 'recordings/%s/data' % self.params['recording_number']

        self.__check_valid_key__(self.h5_key)

        self._open()

        header = {}
        header['data_dtype'] = self.my_file.get(self.h5_key).dtype
        self.compression = self.my_file.get(self.h5_key).compression

        self._check_compression()

        self.size = self.my_file.get(self.h5_key).shape

        if self.size[0] > self.size[1]:
            self.time_axis = 0
            self._shape = (self.size[0], self.size[1])
        else:
            self.time_axis = 1
            self._shape = (self.size[1], self.size[0])

        header['nb_channels'] = self._shape[1]
        mykey = 'recordings/%s/application_data' % self.params['recording_number']
        header['gain'] = dict(list(self.my_file.get(mykey).attrs.items()))['channel_bit_volts']
        self._t_start = dict(list(self.my_file.get(mykey).attrs.items()))['start_time']

        self._close()

        return header
