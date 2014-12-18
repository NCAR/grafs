import unittest
from datetime import datetime,timedelta
import numpy as np
from pygrafs.data.ModelGrid import ModelGrid


class TestDataModelGrid(unittest.TestCase):
    def setUp(self):
        self.filename = "test_data/int_fcst_grid.20141102.11.nc"
        self.model_grid = ModelGrid(self.filename)

    def test_file_loading(self):
        self.assertIsNotNone(self.model_grid.x_var,"Does not have x grid")
        self.assertIsNotNone(self.model_grid.y_var,"Does not have y grid")
        self.assertEqual(len(self.model_grid.x.shape),2,"X-grid not 2-dimensional")
        self.assertEqual(len(self.model_grid.y.shape),2,"Y-grid not 2-dimensional")
        self.assertGreaterEqual(self.model_grid.x.min(),-180,"X-grid less than lower bound")
        self.assertLessEqual(self.model_grid.x.max(),180,"X-grid greater than upper bound")
        self.assertGreaterEqual(self.model_grid.y.min(),-90,"Y-grid less than lower bound")
        self.assertLessEqual(self.model_grid.y.max(),90,"Y-grid greater than upper bound")
        self.assertEqual(self.model_grid.x.size,self.model_grid.y.size,"X and Y grids not same size")

    def test_load_full(self):
        with self.assertRaises(KeyError):
            self.model_grid.load_full("cpe")
        var = 'av_dswrf_sfc'
        self.model_grid.load_full(var)
        self.assertEqual(self.model_grid.data[var][0].size, self.model_grid.x.size,
                         "Variable size does not match coordinate grid size.")
        min_val = np.min(self.model_grid.data[var][10])
        max_val = np.nanmax(self.model_grid.data[var])
        self.assertLessEqual(max_val, 10000, "Max value is %0.2f" % max_val)
        self.assertGreaterEqual(min_val, 0, "Min value is %0.2f" % min_val)


        self.assertNotIn("cpe",self.model_grid.data.keys())

    def test_load_subset(self):
        variable = "av_dswrf_sfc"
        time_subset = (datetime(2014, 11, 02, 18),datetime(2014, 11, 03, 6))
        x_subset = (-121.1, -119.5)
        y_subset = (37.1, 39.5)
        subset_obj = self.model_grid.load_subset(variable,time_subset,y_subset, x_subset,
                                    time_subset_type='coordinate',
                                    space_subset_type='coordinate')
        self.assertTrue(np.all(np.array(subset_obj.data.shape) > 0),
                        "At least 1 dimension has 0 length")

    def test_coordinate_to_index(self):
        input_i = 105
        input_j = 323
        input_x = self.model_grid.x[input_i, input_j]
        input_y = self.model_grid.y[input_i, input_j]
        out_i, out_j = self.model_grid.coordinate_to_index(input_x, input_y)
        self.assertEqual(input_i, out_i, "First index does not match")
        self.assertEqual(input_j, out_j, "Second index does not match")

    def tearDown(self):
        self.model_grid.close()

