import unittest
import numpy as np
import pandas as pd
from forecast import build_features, ridge_predict, load_source, fit_and_evaluate

class ForecastTests(unittest.TestCase):
    def test_future_target_cannot_change_past_features(self):
        data,_=load_source(); modified=data.copy(); modified.loc[500:, 'cnt']*=50
        pd.testing.assert_frame_equal(build_features(data).iloc[:500],build_features(modified).iloc[:500])
    def test_rolling_mean_excludes_current_target(self):
        data,_=load_source(); features=build_features(data)
        self.assertAlmostEqual(features.loc[40,'mean_7'],data.loc[33:39,'cnt'].mean())
    def test_ridge_constant_column_and_nonnegative_output(self):
        x=np.ones((20,3)); y=np.full(20,7.)
        self.assertAlmostEqual(ridge_predict(x,y,np.ones(3),1.),7.)
    def test_test_outcomes_do_not_change_model_selection(self):
        data,_=load_source(); before,_,_=fit_and_evaluate(data,build_features(data))
        modified=data.copy(); modified.loc[modified.dteday >= pd.Timestamp('2012-07-01'),'cnt']*=10
        after,_,_=fit_and_evaluate(modified,build_features(modified))
        self.assertEqual(before['selected_method'],after['selected_method'])
        self.assertEqual(before['validation_scores'],after['validation_scores'])

if __name__=='__main__': unittest.main()
