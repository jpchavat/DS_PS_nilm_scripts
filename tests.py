from datetime import datetime
import dateutil
import sys
import pandas as pd

sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk')
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk_metadata')

from nilmtk import DataSet, HDFDataStore
from nilmtk.metrics import error_in_assigned_energy, f1_score, \
    mean_normalized_error_power, tp_fp_fn_tn

sys.path.insert(0, '/Users/jp/Documents/FIng/nilm-scripts')
from patternsimilarities_disaggregator import PatternSimilaritiesDisaggregator


def train_model(model, data):
    print("> TRAINING")
    model.train(
        data,
        neighbourhood=delta,
        time_interval_neighbourhood=d,
        H=H,
        tolerance=phi,
        sample_period=SAMPLE_PERIOD
    )


def test_model(model, data, predictions_fname):
    print("> TESTING")
    prediction_alg = HDFDataStore(predictions_fname, mode='w')
    model.disaggregate(data, prediction_alg)
    prediction_alg.close()
    del prediction_alg


def calc_metrics(predictions_fname, gt_data):
    """METRiCS"""
    predicted_data = DataSet(predictions_fname, format='HDF')
    predicted_data.clear_cache()
    predicted = predicted_data.buildings[b].elec
    results = {}
    metric_funcs = (
        error_in_assigned_energy,
        f1_score,
        # fraction_energy_assigned_correctly,
        mean_normalized_error_power,
        tp_fp_fn_tn,
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

    predicted_data.store.close()

    return metric_results


if __name__ == "__main__":
    """ LOAD DATA"""
    data_filename = './datasets_h5/synthetic_dataset_1YEAR_UKDALE_house1_2019_08.h5'
    prediction_filename = "./predictions/PRED_{}_{}.h5".format(
        datetime.now().isoformat().replace(':', '').replace('.', '').replace('-', '_'),
        data_filename.replace('.', '').replace('/', '').replace('-', '_')
    )

    data_train = DataSet(data_filename, format='HDF')
    data_train.clear_cache()
    data_test = DataSet(data_filename, format='HDF')
    data_test.clear_cache()

    TRAIN_START = dateutil.parser.parse("2013-04-15 00:00:00")
    TRAIN_END = dateutil.parser.parse("2013-04-30 00:00:00")
    TEST_START = TRAIN_END
    TEST_END = dateutil.parser.parse("2013-05-15 00:00:00")

    data_train.set_window(start=TRAIN_START, end=TRAIN_END)
    data_test.set_window(start=TEST_START, end=TEST_END)

    # SAMPLE_PERIOD = 60
    SAMPLE_PERIOD = 300  # 5min
    # SAMPLE_PERIOD = 600  # 10min
    # SAMPLE_PERIOD = 900  # 15min
    b = 1

    delta = 150  # neighbourhood
    d = 40  # time interval
    H = 500  # separates high from low consumption
    phi = 250  # tolerance to differences

    patternssim_model = PatternSimilaritiesDisaggregator()
    """Train the model"""
    train_model(patternssim_model, data=data_train.buildings[b].elec)
    """Test the model"""
    test_model(
        patternssim_model,
        data=data_test.buildings[b].elec,
        predictions_fname=prediction_filename
    )

    prediction_filename = "./predictions/PRED_2019_08_22T011853352366_datasets_h5synthetic_dataset_1YEAR_UKDALE_house1_2019_08h5.h5"
    metric_results = calc_metrics(
        prediction_filename,
        gt_data=data_test.buildings[b].elec
    )
    print(metric_results.to_string())
    print("> END.")
