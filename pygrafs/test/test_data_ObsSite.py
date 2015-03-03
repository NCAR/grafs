import unittest

from pygrafs.libs.data import ObsSite


class TestDataObsSite(unittest.TestCase):
    def setUp(self):
        obs_file = "test_data/int_obs.20141215.nc"
        self.obs = ObsSite(obs_file)
        return

    def test_meta_file(self):
        print self.obs.meta_data.shape
        self.assertEqual(self.obs.meta_data.shape[1],8)

    def tearDown(self):
        self.obs.close()
        return
