import dateutil
import pandas as pd
import sys

sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk')
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk_metadata')

from nilmtk import DataSet, HDFDataStore
from nilmtk.metrics import error_in_assigned_energy, f1_score, \
    mean_normalized_error_power, tp_fp_fn_tn

sys.path.insert(0, '/Users/jp/Documents/FIng/nilm-scripts')
from patternsimilarities_disaggregator import PatternSimilaritiesDisaggregator

### Load the data
data_train = DataSet(
    './synthetic_dataset_1YEAR_UKDALE_house1-datetimes.h5',
    format='HDF'
)
data_train.clear_cache()
data_test = DataSet(
    './synthetic_dataset_1YEAR_UKDALE_house1-datetimes.h5',
    format='HDF'
)
data_test.clear_cache()
filename = "./testing_patternssimilarities_results_house1_datetimes.h5"

TRAIN_START = dateutil.parser.parse("2013-04-07 00:00:00")
# TRAIN_END = dateutil.parser.parse("2012-12-13 00:00:00")
# TRAIN_END = dateutil.parser.parse("2013-05-11 00:00:00")  # primeros 100mil
TRAIN_END = dateutil.parser.parse("2013-05-07 00:00:00")
# SAMPLE_START = dateutil.parser.parse("2012-12-14 00:00:00")
# SAMPLE_END = dateutil.parser.parse("2012-12-16 00:00:00")
TEST_START = TRAIN_END
# TEST_END = dateutil.parser.parse("2014-04-08 00:00:00")
TEST_END = dateutil.parser.parse("2013-06-07 00:00:00")
# TEST_END = dateutil.parser.parse("2012-12-15 00:00:00")

data_train.set_window(start=TRAIN_START, end=TRAIN_END)
data_test.set_window(start=TEST_START, end=TEST_END)

# SAMPLE_PERIOD = 60
# SAMPLE_PERIOD = 300  # 5min
# SAMPLE_PERIOD = 600  # 10min
SAMPLE_PERIOD = 900  # 15min
b = 1

delta = 150  # neighbourhood
d = 40  # time interval
H = 500  # separates high from low consumption
phi = 250  # tolerance to differences

# TRAIN
print("> TRAINING")
alg_clf = PatternSimilaritiesDisaggregator()
alg_clf.train(
    data_train.buildings[b].elec,
    neighbourhood=delta,
    time_interval_neighbourhood=d,
    H=H,
    tolerance=phi,
    sample_period=SAMPLE_PERIOD
)

# TEST
print("> TESTING")
prediction_alg = HDFDataStore(filename, mode='w')
alg_clf.disaggregate(data_test.buildings[b].elec, prediction_alg)
prediction_alg.close()
del prediction_alg

"""METRiCS"""
predicted_data = DataSet(filename, format='HDF')
predicted_data.clear_cache()
predicted = predicted_data.buildings[b].elec
gt_data = data_test.buildings[b].elec
results = {}
metric_funcs = (
    error_in_assigned_energy,
    f1_score,
    # fraction_energy_assigned_correctly,
    mean_normalized_error_power,
    tp_fp_fn_tn
)
for m_func in metric_funcs:
    metric_func_name = m_func.__name__.replace('_', ' ')
    print("> METRIC {}".format(metric_func_name))
    result = m_func(predicted, gt_data)
    result.index = predicted.get_labels(result.index)
    # print("RESULTS OF METRIC {}:\n{}".format(metric_func_name, result))
    if isinstance(result, pd.DataFrame):
        for i, col in enumerate(result.columns):
            results[col] = result.iloc[:, i]
            results[col].index = result.index
    else:
        results[metric_func_name] = result

metric_results = pd.DataFrame(data=results)
ixs = metric_results.index.tolist()
ixs[0] = 'Aggregated'
metric_results.index = ixs

print(metric_results.to_string())

predicted_data.store.close()

print("> END.")

