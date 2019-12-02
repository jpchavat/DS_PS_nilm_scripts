from datetime import datetime
from time import time

import warnings
warnings.filterwarnings("ignore")

import dateutil
import sys
import pandas as pd

sys.path.insert(0, "/Users/jp/Documents/FIng/nilmtk")
sys.path.insert(0, "/Users/jp/Documents/FIng/nilmtk_metadata")

from nilmtk import DataSet, HDFDataStore
from nilmtk.metrics import (
    error_in_assigned_energy,
    f1_score,
    mean_normalized_error_power,
    tp_fp_fn_tn,
)

sys.path.insert(0, "/Users/jp/Documents/FIng/nilm-scripts")
from patternsimilarities_disaggregator import PatternSimilaritiesDisaggregator


def train_model(model, data):
    start_training = time()
    print("-> TRAINING")
    model.train(
        data,
        neighbourhood=delta,
        time_interval_neighbourhood=d,
        H=H,
        tolerance=phi,
        sample_period=SAMPLE_PERIOD,
    )
    print("-> END TRAINING in {} seconds".format(time() - start_training))


def test_model(model, data, predictions_fname):
    start_testing = time()
    print("-> TESTING")
    prediction_alg = HDFDataStore(predictions_fname, mode="w")
    model.disaggregate(data, prediction_alg)
    prediction_alg.close()
    del prediction_alg
    print("-> END TESTING in {} seconds".format(time() - start_testing))


def correct_prediction_model(
    model, predictions_fname, corrected_prediction_fname, lower_limit, upper_limit
):
    start_correction = time()
    print("-> CORRECTION")

    # corrected_predictions_data = HDFDataStore(corrected_prediction_fname, mode="w")
    predicted_data = DataSet(predictions_fname, format="HDF")
    predicted_data.clear_cache()

    # TODO: generalizar a otro buildings
    predicted = predicted_data.buildings[b].elec.meters

    """Find consumption constant value for fridge and HTPC"""
    max_fridge = model.get_power_series_from_meter(
        model.metergroup.meters[1], clean_cache_before=True
    ).max()
    max_htpc = model.get_power_series_from_meter(
        model.metergroup.meters[-1], clean_cache_before=True
    ).max()

    """Calculate dates where consumption is between limits"""
    indexes_to_change = model.get_power_series_from_meter(
        predicted[0], clean_cache_before=True
    )[lambda x: x >= lower_limit][lambda x: x <= upper_limit].index

    """Replace fridge and htpc by their maximums, and the other appliances by zero"""
    # Max to Fridge
    predicted_fridge = model.get_power_series_from_meter(
        predicted[1], clean_cache_before=True
    )
    predicted_fridge[indexes_to_change] = max_fridge
    # Max to HTPC
    predicted_htpc = model.get_power_series_from_meter(
        predicted[-1], clean_cache_before=True
    )
    predicted_htpc[indexes_to_change] = max_htpc
    # Zero to the rest of the appliances
    for meter in predicted[1:-2]:
        predicted_meter = model.get_power_series_from_meter(
            meter, clean_cache_before=True
        )
        predicted_meter[indexes_to_change] = 0

    """Save the corrected predictions"""
    corrected_predicted_data = HDFDataStore(corrected_prediction_fname, mode="w")
    # Write main
    model.write_main_to_output(predicted[0], corrected_predicted_data)
    mains_pred_name = model.get_power_series_from_meter(predicted[0]).name
    # Write appliances
    for meter in predicted[1:]:
        b_num = meter.building()
        m_num = meter.instance()
        df = pd.DataFrame(model.get_power_series_from_meter(meter))
        df.columns = pd.MultiIndex.from_tuples([mains_pred_name])
        model.write_meterdf_to_output(df, corrected_predicted_data, b_num, m_num)

    corrected_predicted_data.close()

    # predicted_data.close()
    del corrected_predicted_data
    del predicted_data
    print("-> END CORRECTION in {} seconds".format(time() - start_correction))


def calc_metrics(predictions_fname, gt_data):
    """METRiCS"""
    start_metrics = time()
    print("-> METRICS")
    predicted_data = DataSet(predictions_fname, format="HDF")
    predicted_data.clear_cache()
    predicted = predicted_data.buildings[b].elec
    results = {}
    metric_funcs = (
        error_in_assigned_energy,
        f1_score,
        mean_normalized_error_power,
        tp_fp_fn_tn,
    )
    for m_func in metric_funcs:
        metric_func_name = m_func.__name__.replace("_", " ")
        print(" > METRIC {}".format(metric_func_name))
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
    ixs[0] = "Aggregated"
    metric_results.index = ixs

    predicted_data.store.close()

    print("-> END METRICS in {} seconds".format(time() - start_metrics))

    return metric_results


if __name__ == "__main__":
    """ LOAD DATA"""
    start_exp = time()
    # data_filename = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_2019-08-25T190826647710.h5"
    # data_filename = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_CONSTVALUES_2019-08-26T004307788622.h5"
    # data_train_fname = "./datasets_h5/SYNT_1Y_UKDALE_H1_UTC_CONSTVAL_250_2000_2500_2500_80_2019-08-26T004307788622.h5"
    data_train_fname = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_NOISE-3.h5"
    # data_test_fname = "./datasets_h5/SYNT_1Y_UKDALE_H1_UTC_CONSTVAL_260_2000_2400_2600_70_2019-08-26T164245394143.h5"
    data_test_fname = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_NOISE-3.h5"
    prediction_filename = "./predictions/PRED_{}_{}.h5".format(
        datetime.now().isoformat().replace(":", "").replace(".", "").replace("-", "_"),
        data_test_fname.replace(".", "").replace("/", "").replace("-", "_"),
    )

    data_train = DataSet(data_train_fname, format="HDF")
    data_train.clear_cache()
    data_test = DataSet(data_test_fname, format="HDF")
    data_test.clear_cache()

    TRAIN_START = dateutil.parser.parse("2013-01-01 00:00:00")
    # TRAIN_END = dateutil.parser.parse("2013-01-01 12:00:00")
    TRAIN_END = dateutil.parser.parse("2013-07-01 00:00:00")
    TEST_START = TRAIN_END
    # TEST_START = dateutil.parser.parse("2013-01-01 00:00:00")
    # TEST_END = dateutil.parser.parse("2013-01-10 00:00:00")
    TEST_END = dateutil.parser.parse("2013-12-31 23:59:59")
    # TEST_END = dateutil.parser.parse("2013-01-30 00:00:00")

    data_train.set_window(start=TRAIN_START, end=TRAIN_END)
    data_test.set_window(start=TEST_START, end=TEST_END)

    # SAMPLE_PERIOD = 60
    SAMPLE_PERIOD = 300  # 5min
    # SAMPLE_PERIOD = 600  # 10min
    # SAMPLE_PERIOD = 900  # 15min
    # SAMPLE_PERIOD = 1200  # 20min
    # SAMPLE_PERIOD = 2400  # 40min
    b = 1

    delta = 100  # neighbourhood
    d = 10  # time interval
    H = 500  # separates high from low consumption
    phi = 250  # tolerance to differences

    # lower_limit_L1 = 270
    lower_limit_L1 = 190
    # upper_limit_L2 = 500
    upper_limit_L2 = 230

    print(
        "============================================================================\n"
        "\n"
        "EXPERIMENT PS {}\n".format(datetime.now().isoformat()) +
        "\n"
        " Dataset train filename: {}\n".format(data_train_fname) +
        " Dataset test filename: {}\n".format(data_test_fname) +
        " Predictions filename: {}\n".format(prediction_filename) +
        "\n"
        " Train datetime range from {} to {}\n".format(TRAIN_START, TRAIN_END) +
        " Test datetime range from {} to {}\n".format(TEST_START, TEST_END) +
        "\n"
        "Parameters:\n"
        " Sample period: {}\n".format(SAMPLE_PERIOD) +
        " Delta: {}\n".format(delta) +
        " Time interval (d): {}\n".format(d) +
        " Low consumption threshold (H): {}\n".format(H) +
        " Tolerance in diff. (phi): {}".format(phi)
    )

    ELEC_TYPES = ["fridge", "washer dryer", "kettle", "dish washer", "HTPC"]

    patternssim_model = PatternSimilaritiesDisaggregator()

    """Train the model"""
    TRAIN_ELECTS = data_train.buildings[b].elec.from_list(
        [m.identifier for m in data_train.buildings[b].elec.select_using_appliances(
            type=ELEC_TYPES
        ).meters] + [data_train.buildings[b].elec.mains().identifier]
    )
    train_model(patternssim_model, data=TRAIN_ELECTS)

    """Test the model"""
    TEST_ELECTS = data_test.buildings[b].elec.from_list(
        [m.identifier for m in data_test.buildings[b].elec.select_using_appliances(
            type=ELEC_TYPES
        ).meters] + [data_test.buildings[b].elec.mains().identifier]
    )
    test_model(
        patternssim_model,
        data=TEST_ELECTS,
        predictions_fname=prediction_filename,
    )

    # prediction_filename = './predictions/PRED_2019_11_14T191402232251_datasets_h5synthetic_1YEAR_UKDALE_house1_UTC_articulo_NOISE_2h5.h5'
    metric_results = calc_metrics(
        prediction_filename, gt_data=TEST_ELECTS
    )
    METRIC_COLS = ["Fridge", "Washer dryer", "Kettle", "Dish washer", "HTPC"]
    metric_results = metric_results.T.iloc[[0, 2, 7, 8, 1], 1:].loc[:,METRIC_COLS]
    metric_results.index = ["TEE (kW)", "NEAP", "precision", "recall", "F-score"]
    metric_results = metric_results.round(4)
    metric_results.iloc[0, :] = (metric_results.iloc[0, :] / 1000).round(2)
    print(metric_results.to_string())
    print("CSV FORMAT...")
    metric_results.to_csv(sys.stdout, sep="&", line_terminator="\\\\\n")

    print(
        "> END experiment in {} seconds.\n"
        "\n"
        "============================================================================"
        .format(time() - start_exp)
    )
