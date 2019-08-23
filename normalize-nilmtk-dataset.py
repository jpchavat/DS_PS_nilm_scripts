from __future__ import print_function
import yaml
import os
import pandas as pd
import numpy as np
from os.path import join

import matplotlib
# Comment to see plots in UI
matplotlib.use('Agg')  # avoid using tkinter (not installed in cluster.uy)

import sys
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk')
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk_metadata')
sys.path.insert(0, '/clusteruy/home/jpchavat/nilmtk')  # Clusteruy
sys.path.insert(0, '/clusteruy/home/jpchavat/nilm_metadata')  # Clusteruy
from nilmtk import DataSet, version, MeterGroup

print("NILMTK version {}".format(version.version))


def calc_stats(activations):
    # type: (list) -> dict
    maxs_act = [act.max() for act in activations]
    avg = np.average(maxs_act)
    median = np.median(maxs_act)
    std = np.std(maxs_act)
    max_max = np.max(maxs_act)
    min_max = np.min(maxs_act)

    lim_inf = max(0, median - 2 * std)
    lim_sup = median + 2 * std
    acts_std_2 = [a for a in maxs_act if lim_inf < a < lim_sup]
    avg_std_2 = np.average(acts_std_2)
    median_std_2 = np.median(acts_std_2)

    lim_inf = max(0, median - std)
    lim_sup = median + std
    acts_std_1 = [a for a in acts_std_2 if lim_inf < a < lim_sup]
    avg_std_1 = np.average(acts_std_1)
    median_std_1 = np.median(acts_std_1)

    return {
        "n_act": len(activations),
        "n_act_std_1": len(acts_std_1),
        "n_act_std_2": len(acts_std_2),
        "max_max": max_max,
        "min_max": min_max,
        "std": std,
        "avg": avg,
        "median": median,
        "avg_std_1": avg_std_1,
        "avg_std_2": avg_std_2,
        "median_std_1": median_std_1,
        "median_std_2": median_std_2,
    }


def calc_elec_stats(elecs, print_results=False):
    # type: (MeterGroup, bool) -> dict
    """Calculate stats for each appliance in `elecs`."""
    stats_str = (
        "Max of maxs: {max_max}\n"
        "Min of maxs: {min_max}\n"
        "Std: {std}\n"
        "\n"
        "Number of activations: {n_act}\n"
        "Average max: {avg}\n"
        "Median max: {median}\n"
        "\n"
        "Number of activations (in 2*std): {n_act_std_2}\n"
        "Average max (in 2*std): {avg_std_2}\n"
        "Median max (in 2*std): {median_std_2}\n"
        "\n"
        "Number of activations (in std): {n_act_std_1}\n"
        "Average max (in std): {avg_std_1}\n"
        "Median max (in std): {median_std_1}\n"
    )
    elecs_stats = {}
    for e in ELEC_NAMES:
        activations = elecs[e].get_activations()
        stats = calc_stats(activations)
        elecs_stats[e] = stats
        print(e, end="..")
        if print_results:
            print(stats_str.format(**stats))

    return elecs_stats


def plot_elecs(elecs, elecs_stats):
    # type: (MeterGroup, dict) -> None
    import matplotlib as plt

    # Plot the calculated stats and visualize best value to normalize
    cols = 1
    rows = int(np.ceil(len(elecs_stats.keys()) / cols))

    NUM_ACTIVATIONS_TO_PLOT = 6

    fig, axs = plt.subplots(int(np.ceil(rows)), cols)

    for i, (e, stats) in enumerate(elecs_stats.items()):  # <-- estoy LIMITANDO AQUI
        ax = axs[i % rows, i % cols] if cols > 1 else axs[i]
        ax.set_title(e, fontsize=24)

        print("Loading activations for {}...".format(e))
        activations = elecs[e].get_activations()
        print("ploting {} activations for {}".format(NUM_ACTIVATIONS_TO_PLOT, e))
        for i in range(NUM_ACTIVATIONS_TO_PLOT):
            activations[i].plot(ax=ax)

        ax.axhline(y=stats["median"], color="red")
        ax.axhline(y=stats["median_std_2"], color="yellow")
        ax.axhline(y=stats["median_std_1"], color="green")

    plt.show()


def create_normalized_consumption(elecs, stats):
    # type: (MeterGroup, dict) -> pd.DataFrame
    """Normalize ON values for each appliance with media value"""
    new_appliance_data = {}

    for e in ELEC_NAMES:
        # Initialize with zeros
        new_appliance_data[e] = pd.Series(
            index=elecs[e].power_series_all_data().index, data=0.0
        )
        # Replace zeros by median where consumption is greater than MIN_CONSUMPTION
        new_appliance_data[e][
            elecs[e].power_series_all_data() > MIN_CONSUMPTION
        ] = stats[e]["median_std_1"]

    # Creates the DataFrame with index datetime and columns each appliance
    # normalized consumption
    new_data = pd.DataFrame(data=new_appliance_data).fillna(value=0.0)
    new_data = pd.concat([new_data], axis=1, keys=["power"])
    new_data.index.name = "datetime"

    return new_data


def resample_norm_cons(norm_data):
    # type: (pd.DataFrame) -> pd.DataFrame
    """Resample the series and unify data into bins of 6 seconds"""
    new_data_resampled = norm_data.resample(
        "6S"
    ).max()  # Takes just one record / appliance
    new_data_resampled["power", "active"] = new_data_resampled.sum(axis=1)
    new_data_resampled = new_data_resampled.fillna(value=0.0)
    # new_data_resampled.head()

    return new_data_resampled


def create_metadata_directories(base_dir):
    # type: (str) -> str
    """Creates directory structures for metadata"""

    base_dir_metadata = join(base_dir, "metadata")
    base_dir_building = join(base_dir, "building{}")
    base_dir_building_elec = join(base_dir, "building{}", "elec")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(base_dir_metadata, exist_ok=True)
    for b in BUILDINGS:
        os.makedirs(base_dir_building.format(b), exist_ok=True)
        os.makedirs(base_dir_building_elec.format(b), exist_ok=True)

    return base_dir


def create_metadata_files(base_dir, data):
    # type: (str, DataSet) -> None
    """Creates the metadata files in the metadata directory structure"""

    # Creates a yaml metadata for the dataset
    dataset_metadata = {
        "name": "SYNTHETIC-NORM",
        "long_name": "Synthetic metadata from normalized UK-DALE (?)",
        "description": "Created by taking some appliances and assigning constant (median in std-1) consumption when state is ON",
        "schema": "https://github.com/nilmtk/nilm_metadata/tree/v0.2",
        "geo_location": data.metadata.get("geo_location"),
        "date": data.metadata.get("date"),
        "timezone": data.metadata.get("timezone"),
    }
    with open(join(base_dir, "metadata", "dataset.yaml"), "w") as f:
        yaml.dump(dataset_metadata, f, default_flow_style=False)

    # Creates a yaml metadata for the meter devices
    meter_device = """synthetic_monitor:
      model: synthetic-monitor
      sample_period: 6   # the interval between samples. In seconds.
      max_sample_period: 6   # Max allowable interval between samples. Seconds.
      measurements:
      - physical_quantity: power   # power, voltage, energy, current?
        type: active   # active (real power), reactive or apparent?
        upper_limit: 5000
        lower_limit: 0"""
    with open(join(BASE_DIR, "metadata", "meter_devices.yaml"), "w") as f:
        f.write(meter_device)

    # Create a yaml metadata for the building
    building_metadata = {
        "instance": 1,
        "elec_meters": {
            i + 1: {"site_meter": True, "device_model": "synthetic_monitor"}
            if i == 0
            else {"submeter_of": 1, "device_model": "synthetic_monitor"}
            for i, _ in enumerate([0] + ELEC_NAMES)
        },
        "appliances": [
            {"type": e, "instance": 1, "meters": [i + 2]}
            for i, e in enumerate(ELEC_NAMES)
        ],
    }
    with open(join(BASE_DIR, "metadata", "building{}.yaml").format(b), "w") as f:
        yaml.dump(building_metadata, f, default_flow_style=False)


def create_data_files(norm_cons, building, dest_format="HDF"):
    # type: (pd.DataFrame, int, str) -> None
    """Creates the CSV/H5 file(s), depending on format selected (CSV/HDF),
    with the data of the meters/appliances of building `building`."""
    if dest_format == "CSV":
        """CSV CREATION"""
        # Creates ONE CSV per appliance, simulating a meter
        norm_cons["power"].to_csv(
            path_or_buf=join(
                BASE_DIR, "building{}".format(building), "elec", "meter1.csv"
            ),
            columns=["active"],
            index=True,
        )
        for i, e in enumerate(ELEC_NAMES):
            norm_cons["power"].to_csv(
                path_or_buf=join(
                    BASE_DIR,
                    "building{}".format(building),
                    "elec",
                    "meter{}.csv".format(i + 2),
                ),
                columns=[e],
                index=True,
            )
    elif dest_format == "HDF":
        raise NotImplementedError("Destination formato not implemented")
        # Open DataStore
        # store = get_datastore(BASE_DIR + "h5", "HDF", mode="w")
    else:
        raise NotImplementedError("Destination formato not implemented")


INPUT_DATASET_FILE = "/Users/jp/Documents/FIng/PruebasNILM/ukdale.h5"
BUILDINGS = [1]
BASE_DIR = "/Users/jp/Documents/FIng/PruebasNILM/synthetic_dataset_1YEAR_UKDALE_house1"
ELEC_NAMES = ["fridge", "washer dryer", "kettle", "dish washer", "HTPC"]

MIN_CONSUMPTION = 5.0

if __name__ == "__main__":
    # Load the dataset
    data = DataSet(INPUT_DATASET_FILE)
    data.clear_cache()
    # Set datetime range
    data.set_window(start="2013-04-07 00:00:00", end="2014-04-08 23:59:59")

    for b in BUILDINGS:
        print("Start processing building {}...".format(b))
        data.buildings[b].elec.select_using_appliances(type=ELEC_NAMES)
        elecs = data.buildings[b].elec

        print("> Calculating stats...", end='')
        elecs_stats = calc_elec_stats(elecs=elecs, print_results=False)
        print("FINISH")

        print("> Normalizing consumptions...", end='')
        norm_cons = create_normalized_consumption(elecs, stats=elecs_stats)
        print("FINISH")

        print("> Resampling consumptions...", end='')
        norm_cons_resampled = resample_norm_cons(norm_cons)
        print("FINISH")

        print("> Creating metadata directories...", end='')
        create_metadata_directories(base_dir=BASE_DIR)
        print("FINISH")

        print("> Creating metadata files...", end='')
        create_metadata_files(base_dir=BASE_DIR, data=data)
        print("FINISH")

        print("> Creating data files...", end='')
        create_data_files(norm_cons_resampled, building=b, dest_format="CSV")
        print("FINISH")

        print("END")
