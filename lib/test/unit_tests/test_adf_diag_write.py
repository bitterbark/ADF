"""
Collection of python unit tests for "write_netcdf_synchronous"
in adf_diag.py, which writes the time series and lat-lon remapped
files.

Threaded dask writes hung intermittently on large files, so these
tests check that every chunk is computed in the calling thread, that
the file only appears under its final name once the write succeeds,
and that a failed write leaves no file behind. Each test writes small
synthetic files to a temporary directory.

NOTE: adf_diag needs xarray, dask and netCDF4, so these tests are
skipped in CI, which installs only PyYAML and pytest.  They run in a
full ADF environment.
"""

#+++++++++++++++++++++++
#Import required modules
#+++++++++++++++++++++++

import unittest
import sys
import os
import os.path
import shutil
import tempfile
import threading

import pytest

pytest.importorskip("xarray")
pytest.importorskip("dask")
pytest.importorskip("netCDF4")

# pylint: disable=wrong-import-position
import numpy as np
import xarray as xr
import dask.array as da

#Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

#Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

#Import function being tested
from adf_diag import write_netcdf_synchronous, save_to_nc
# pylint: enable=wrong-import-position


def _make_dataset(record=None):
    """Small dask-backed dataset with 4 time chunks.

    If record is a list, the thread id that computes each chunk is appended to it.
    """
    values = np.arange(4 * 3 * 5, dtype="f4").reshape(4, 3, 5)
    data = da.from_array(values, chunks=(1, 3, 5))
    if record is not None:
        def _note_thread(block):
            record.append(threading.get_ident())
            return block
        data = data.map_blocks(_note_thread, dtype=data.dtype)
    ds = xr.Dataset({"PRECT": (("time", "lat", "lon"), data)},
                    coords={"time": np.arange(4.0), "lat": np.arange(3.0), "lon": np.arange(5.0)})
    return ds, values


class WriteNetcdfSynchronousTestRoutine(unittest.TestCase):

    """
    Runs all of the unit tests for write_netcdf_synchronous.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.outname = os.path.join(self.tmpdir, "test.PRECT.nc")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_chunks_computed_in_calling_thread(self):
        """Every chunk is computed in the calling thread (no dask thread pool)."""
        threads = []
        ds, _ = _make_dataset(record=threads)
        write_netcdf_synchronous(ds.to_netcdf, self.outname)
        self.assertGreaterEqual(len(threads), 4)
        self.assertEqual(set(threads), {threading.get_ident()})

    def test_output_matches_input(self):
        """The file is written under its final name, unchanged, with no .tmp left."""
        ds, values = _make_dataset()
        write_netcdf_synchronous(lambda path: save_to_nc(ds, path), self.outname)
        self.assertTrue(os.path.exists(self.outname))
        self.assertFalse(os.path.exists(self.outname + ".tmp"))
        with xr.open_dataset(self.outname) as out:
            np.testing.assert_array_equal(out["PRECT"].values, values)

    def test_failed_write_leaves_no_file(self):
        """A failed write creates no final file and removes its .tmp."""
        def _fail(path):
            with open(path, "w", encoding="utf-8") as fil:
                fil.write("partial")
            raise RuntimeError("simulated write failure")
        with self.assertRaises(RuntimeError):
            write_netcdf_synchronous(_fail, self.outname)
        self.assertFalse(os.path.exists(self.outname))
        self.assertFalse(os.path.exists(self.outname + ".tmp"))

    def test_failed_write_keeps_existing_file(self):
        """A failed write leaves an existing file at the final name untouched."""
        with open(self.outname, "w", encoding="utf-8") as fil:
            fil.write("original")
        def _fail(path):
            raise RuntimeError("simulated write failure")
        with self.assertRaises(RuntimeError):
            write_netcdf_synchronous(_fail, self.outname)
        with open(self.outname, encoding="utf-8") as fil:
            self.assertEqual(fil.read(), "original")


#++++++++++++++++++++++++++++++++++++++++++++++++
#Run unit tests if this script is called directly
#++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    unittest.main()
