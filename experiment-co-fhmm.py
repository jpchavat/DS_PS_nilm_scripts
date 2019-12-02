from datetime import datetime
from time import time

import warnings
warnings.filterwarnings("ignore")

import dateutil
import pandas as pd

import sys
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk')
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk_metadata')

from nilmtk import DataSet, HDFDataStore
from nilmtk.disaggregate import (
    combinatorial_optimisation,
    fhmm_exact,
)
from nilmtk.metrics import (
    error_in_assigned_energy,
    f1_score,
    mean_normalized_error_power,
    tp_fp_fn_tn
)


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
    # ixs = metric_results.index.tolist()
    # ixs[0] = "Aggregated"
    # metric_results.index = ixs

    predicted_data.store.close()

    print("-> END METRICS in {} seconds".format(time() - start_metrics))

    return metric_results


if __name__ == "__main__":
    """ LOAD DATA"""
    start_exp = time()
    # data_filename = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_2019-08-25T190826647710.h5"
    # data_filename = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_CONSTVALUES_2019-08-26T004307788622.h5"
    # data_filename = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_CONSTVALUES_2_2019-08-26T120856188525.h5"
    # data_train_fname = "./datasets_h5/SYNT_1Y_UKDALE_H1_UTC_CONSTVAL_250_2000_2500_2500_80_2019-08-26T004307788622.h5"
    data_train_fname = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_NOISE-3.h5"
    # data_test_fname = "./datasets_h5/SYNT_1Y_UKDALE_H1_UTC_CONSTVAL_260_2000_2400_2600_70_2019-08-26T164245394143.h5"
    data_test_fname = "./datasets_h5/synthetic_1YEAR_UKDALE_house1_UTC_articulo_NOISE-3.h5"
    prediction_filename_co = './predictions/PRED_CO_{}_{}.h5'.format(
        datetime.now().isoformat().replace(":", "").replace(".", "").replace("-", "_"),
        data_train_fname.replace(".", "").replace("/", "").replace("-", "_")
    )
    prediction_filename_fhmm = './predictions/PRED_FHMM_{}_{}.h5'.format(
        datetime.now().isoformat().replace(":", "").replace(".", "").replace("-", "_"),
        data_train_fname.replace(".", "").replace("/", "").replace("-", "_")
    )

    data_train = DataSet(data_train_fname, format="HDF")
    data_train.clear_cache()
    data_test = DataSet(data_test_fname, format="HDF")
    data_test.clear_cache()

    TRAIN_START = dateutil.parser.parse("2013-01-01 00:00:00")
    # TRAIN_END = dateutil.parser.parse("2013-01-01 12:00:00")
    TRAIN_END = dateutil.parser.parse("2013-07-01 00:00:00")
    TEST_START = TRAIN_END
    TEST_END = dateutil.parser.parse("2013-12-31 23:59:59")
    # TEST_END = dateutil.parser.parse("2013-01-30 00:00:00")

    data_train.set_window(start=TRAIN_START, end=TRAIN_END)
    data_test.set_window(start=TEST_START, end=TEST_END)

    # SAMPLE_PERIOD = 60
    SAMPLE_PERIOD = 300  # 5min
    # SAMPLE_PERIOD = 600  # 10min
    # SAMPLE_PERIOD = 900  # 15min
    # SAMPLE_PERIOD = 1200  # 20min
    b = 1

    print(
        "============================================================================\n"
        "\n"
        "EXPERIMENT CO-FHMM {}\n".format(datetime.now().isoformat()) +
        "\n"
        " Dataset train filename: {}\n".format(data_train_fname) +
        " Dataset test filename: {}\n".format(data_test_fname) +
        " Predictions CO filename: {}\n".format(prediction_filename_co) +
        " Predictions FHMM filename: {}\n".format(prediction_filename_fhmm) +
        "\n"
        " Train datetime range from {} to {}\n".format(TRAIN_START, TRAIN_END) +
        " Test datetime range from {} to {}\n".format(TEST_START, TEST_END) +
        "\n"
        "Parameters:\n"
        " Sample period: {}\n".format(SAMPLE_PERIOD)
    )

    predictions = {
        "CO": {
            "file": prediction_filename_co,
            "alg": combinatorial_optimisation.CombinatorialOptimisation
        },
        "FHMM": {
            "file": prediction_filename_fhmm,
            "alg": fhmm_exact.FHMM
        },
    }

    ELEC_TYPES = ["fridge", "washer dryer", "kettle", "dish washer", "HTPC"]
    TRAIN_ELECTS = data_train.buildings[b].elec.from_list(
        [m.identifier for m in data_train.buildings[b].elec.select_using_appliances(
            type=ELEC_TYPES
        ).meters] + [data_train.buildings[b].elec.mains().identifier]
    )
    TEST_ELECTS = data_test.buildings[b].elec.from_list(
        [m.identifier for m in data_test.buildings[b].elec.select_using_appliances(
            type=ELEC_TYPES
        ).meters] + [data_test.buildings[b].elec.mains().identifier]
    )

    for p, info in predictions.items():
        alg = info['alg']
        pred_filename = info['file']

        # TRAIN
        print("> TRAINING {}".format(p))
        data_train.set_window(start=TRAIN_START, end=TRAIN_END)
        alg_clf = alg()
        alg_clf.train(TRAIN_ELECTS, sample_period=SAMPLE_PERIOD)

        # TEST
        print("> TESTING {}".format(p))
        prediction_alg = HDFDataStore(pred_filename, mode='w')
        data_test.set_window(start=TEST_START, end=TEST_END)
        alg_clf.disaggregate(
            # data_test.buildings[b].elec,
            TEST_ELECTS,
            prediction_alg,
            sample_period=SAMPLE_PERIOD)
        prediction_alg.close()
        del prediction_alg

        # METRICS
        metric_results = calc_metrics(pred_filename, gt_data=TEST_ELECTS)
        METRIC_COLS = ["Fridge", "Washer dryer", "Kettle", "Dish washer", "HTPC"]
        metric_results = metric_results.T.iloc[[0, 2, 7, 8, 1], :].loc[:, METRIC_COLS]
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
