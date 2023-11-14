Supported File Formats
======================

To get the list of supported file format, you need to do::

	>> spyking-circus help -i
	-------------------------  Informations  -------------------------
	| The file formats that are supported are:
	| 
	| -- RAW_BINARY (read/parallel write)
	|       Extensions       : 
	|       Supported streams: multi-files
	| -- MCS_RAW_BINARY (read/parallel write)
	|       Extensions       : .raw, .dat
	|       Supported streams: multi-files
	| -- HDF5 (read/write)
	|       Extensions       : .h5, .hdf5
	|       Supported streams: multi-files
	| -- OPENEPHYS (read/parallel write)
	|       Extensions       : .openephys
	|       Supported streams: multi-folders
	| -- KWD (read/write)
	|       Extensions       : .kwd
	|       Supported streams: multi-files, single-file
	| -- NWB (read/write)
	|       Extensions       : .nwb, .h5, .hdf5
	|       Supported streams: multi-files
	| -- NIX (read/write)
	|       Extensions       : .nix, .h5, .hdf5
	|       Supported streams: multi-files
	| -- ARF (read/write)
	|       Extensions       : .arf, .hdf5, .h5
	|       Supported streams: multi-files, single-file
	| -- BRW (read/write)
	|       Extensions       : .brw
	|       Supported streams: multi-files
	| -- NUMPY (read/parallel write)
	|       Extensions       : .npy
	|       Supported streams: multi-files
	| -- RHD (read/parallel write)
	|       Extensions       : .rhd
	|       Supported streams: multi-files
	| -- NEURALYNX (read/parallel write)
	|       Extensions       : .ncs
	|       Supported streams: multi-files, multi-folders
	| -- BLACKROCK (read only)
	|       Extensions       : .ns1, .ns2, .nss3, .ns4, .ns5, .ns6
	|       Supported streams: multi-files
	| -- MDA (read/parallel write)
	|       Extensions       : .mda
	|       Supported streams: multi-files
	------------------------------------------------------------------

This list will tell you what are the wrappers available, and you need to specify one in your configuration file with the ``file_format`` parameter in the ``[data]`` section. To know more about the mandatory/optional parameters for a given file format, you should do::

	>> spyking-circus raw_binary -i
	-------------------------  Informations  -------------------------
	| The parameters for RAW_BINARY file format are:
	| 
	| -- sampling_rate -- <type 'float'> [** mandatory **]
	| -- data_dtype -- <type 'str'> [** mandatory **]
	| -- nb_channels -- <type 'int'> [** mandatory **]
	| 
	| -- data_offset -- <type 'int'> [default is 0]
	| -- dtype_offset -- <type 'str'> [default is auto]
	| -- gain -- <type 'int'> [default is 1]
	------------------------------------------------------------------

.. note:: 
	
	Depending on the file format, the parameters needed in the ``[data]`` section of the parameter file can vary. Some file format are self-contained, while some others need extra parameters to reconstruct the data. For all the needed parameters, you need to add in the ``[data]`` section of the parameter file a line with ``parameter = value``


.. warning::

	As said after, only file format derived from  ``raw_binary``, and without streams are currently supported by the phy and MATLAB GUI, if you want to see the raw data. All other views, that do not depend on the raw data, will stay the same, so you can still sort your data.


Neuroshare support
------------------

Some of the file formats (plexon, ...) can be accessed only if you have the neuroshare_ library installed. Note that despite a great simplicity of use, this library provides only very slow read access and no write access to the file formats. Therefore, this is not an efficient wrapper, and it may slow down considerably the code. Feel free to contribute if you have better ideas about what to do!

Multi-Channel support
---------------------

To be able to read efficiently native mcd files, you must have the pymcstream_ python package installed. This is a cross-platform packages (Windows/Mac/Linux) and the installation procedure can be found on the webwsite.

HDF5-like file
--------------

This should be easy to implement any HDF5-like file format. Some are already available, feel free to add yours. Note that to allow parallel write with HDF5, you must have a version of HDF5 compiled with the MPI option activated. This means that you need to do a :doc:`manual install <../introduction/hdf5>`.


Raw binary File
---------------

The simplest file format is the raw_binary one. Suppose you have *N* channels 

.. math::

   c_0, c_1, ... , c_N

And if you assume that :math:`c_i(t)` is the value of channel :math:`c_i` at time *t*, then your datafile should be a raw file with values

.. math::

   c_0(0), c_1(0), ... , c_N(0), c_0(1), ..., c_N(1), ... c_N(T)


This is simply the flatten version of your recordings matrix, with size *N* x *T* 

.. note::

    The values can be saved in your own format (``int16``, ``uint16``, ``int8``, ``float32``). You simply need to specify that to the code


As you can see by typing::

	>> spyking-circus raw_binary -i
	-------------------------  Informations  -------------------------
	| The parameters for RAW_BINARY file format are:
	| 
	| -- sampling_rate -- <type 'float'> [** mandatory **]
	| -- data_dtype -- <type 'str'> [** mandatory **]
	| -- nb_channels -- <type 'int'> [** mandatory **]
	| 
	| -- data_offset -- <type 'int'> [default is 0]
	| -- dtype_offset -- <type 'str'> [default is auto]
	| -- gain -- <type 'int'> [default is 1]
	------------------------------------------------------------------

There are some extra and required parameters for the raw_binary file format. For example, you must specify the sampling rate ``sampling_rate``, the data_dtype (``int16``, ``float32``, ...) and also the number of channels ``nb_channels``. The remaining parameters are optional, i.e. if not provided, default values written there will be used. So the ``mydata.params`` file for a ``mydata.dat`` raw binary file will have the following params in the ``[data]`` section::

	file_format   = raw_binary
	sampling_rate = XXXX
	data_dtype    = XXXX # should be int16,uint16,float32,...
	nb_channels   = XXXX # as it can not be guessed from the file, it has to be specified
	data_offset   = XXXX # Optional, if a header with a fixed size is present
	gain          = XXXX # Optional, if you want a non unitary gain for the channels


.. warning::

	The ``raw_binary`` file format is the default one used internally by SpyKING CIRCUS when the flag ``overwrite`` is set to ``False``. This means several things

		* data are saved as ``float32``, so storage can be large
		* we can not handle properly t_start parameters if there are streams in the original data. Times will be continuous
		* this is currently the **only** file format properly supported by phy and MATLAB GUIs, if you want to see the raw data

.. _neuroshare: https://pythonhosted.org/neuroshare/
.. _pymcstream: https://bitbucket.org/galenea/pymcstream/src