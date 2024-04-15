import pathlib
# import re
import sys
# import os
import numpy as np
import time as tm
from .datafile import DataFile, comm


class RawBinaryFile(DataFile):

    description = "raw_binary"
    extension = []
    parallel_write = True
    is_writable = True

    _required_fields = {
        'data_dtype': str,
        'sampling_rate': float,
        'nb_channels': int
    }

    _default_values = {
        'dtype_offset': 'auto',
        'data_offset': 0,
        'gain': 1.0
    }

    def _read_from_header(self):
        self._open()
        self.size = len(self.data)
        self._shape = (self.size // self.nb_channels, int(self.nb_channels))
        self._close()
        return {}

    def allocate(self, shape, data_dtype=None):
        if data_dtype is None:
            data_dtype = self.data_dtype
        #self._shape = shape
        #self.size = np.prod(shape) // self.nb_channels

        if self.is_master:
            print(f"{data_dtype=}, {shape=}")
            print(f"allocating {(np.prod(shape) / (2**20)) * np.dtype(data_dtype).itemsize / (2**10):.03f} Gbyte file: {self.file_name}.")
            self.data = np.memmap(self.file_name, offset=self.data_offset, dtype=data_dtype, mode='w+', shape=shape)
            pathlib.Path(self.file_name).touch()
        comm.Barrier()
        self._read_from_header()
        #del self.data

    def read_chunk(self, idx, chunk_size, padding=(0, 0), nodes=None):

        t_start, t_stop = self._get_t_start_t_stop(idx, chunk_size, padding)
        local_shape = t_stop - t_start

        self._open()

        do_slice = nodes is not None
        local_chunk = self.data[t_start*self.nb_channels:t_stop*self.nb_channels]
        local_chunk = local_chunk.reshape(local_shape, self.nb_channels)

        if do_slice:
            local_chunk = np.take(local_chunk, nodes, axis=1)

        self._close()

        return self._scale_data_to_float32(local_chunk)

    def _write_chunk(self, time, data):
        data = self._unscale_data_from_float32(data)
        data = data.ravel()
        itemsize = np.dtype(self.data_dtype).itemsize
        offset = itemsize * self.nb_channels * time + self.data_offset

        stime = tm.time()
        self._open(mode='r+', offset=offset, shape=(len(data),))
        self.data[:len(data)] = data
        #self._open(mode='r+')
        #self.data[self.nb_channels*time:self.nb_channels*time+len(data)] = data
        self._close()

        print(f"[{comm.Get_rank()}/{comm.Get_size()}] writing: offset={offset/(2**30):.03f}Gb, {len(data)=} ({len(data)*itemsize/(2**30):.03f}Gb), isize {itemsize} byte, {tm.time() - stime:.02f}sec")
        sys.stdout.flush()

    def write_chunk(self, time, data):
        # p2p write
        n_writer = 1
        assert n_writer <= comm.Get_size()
        n_group = comm.Get_size() // n_writer
        rank = comm.Get_rank()

        # Circular topology
        group = int(rank / n_group)
        group_lead_rank = group * n_group
        rank_in_group = rank % n_group
        next_rank = group_lead_rank + (rank_in_group + 1) % n_group
        prev_rank = group_lead_rank + (rank_in_group - 1) % n_group
        if next_rank >= comm.Get_size():
            next_rank = group_lead_rank
        if prev_rank >= comm.Get_size():
            prev_rank = comm.Get_size() - 1

        is_group_lead = rank_in_group == 0

        if not (is_group_lead and self._starter):
            _ = comm.recv(source=prev_rank, tag=rank)
        else:
            self._starter = False
        self._write_chunk(time, data)

        if not self._ender:
            comm.send(1, dest=next_rank, tag=next_rank)

    def _open(self, mode='r', offset=None, shape=None):
        if offset is None:
            offset = self.data_offset
        self.data = np.memmap(self.file_name, offset=offset, dtype=self.data_dtype, mode=mode, shape=shape)

    def _close(self):
        #self.data.flush()
        #pathlib.Path(self.file_name).touch()
        del self.data
        self.data = None
