import glob
import os
from datetime import datetime

import pandas as pd
from os.path import join
from sys import stdout

from nilmtk import DataStore
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_datastore, get_module_directory, check_directory_exists
from nilm_metadata import save_yaml_to_datastore


def convert_data(input_path: str, output_dir: str = "./") -> None:
    """
    Parameters
    ----------
    input_path : str
        The root path of the CSV files, e.g. House1.csv
    output_dir : str
        The destination directory.
    """

    # Open DataStore
    store = get_datastore(output_dir, "HDF", mode="w")

    # Convert raw data to DataStore
    _convert(input_path, store)

    # Add metadata
    yaml_dir = join(get_module_directory(), input_path, "metadata")
    save_yaml_to_datastore(yaml_dir=yaml_dir, store=store)

    store.close()

    print("Done converting SYNTHEDIC DATA IN CSV to HDF5!")


def _convert(input_path: str, store: DataStore) -> None:
    """
    Parameters
    ----------
    input_path : str
        The root path of the REFIT dataset.
    store : DataStore
        The NILMTK DataStore object.
    """

    check_directory_exists(input_path)
    sub_dirs = list(os.walk(input_path))[0][1]

    # Iterate though all houses and channels

    for house_id, house_dir in enumerate(
        d for d in sub_dirs if d.startswith("building")
    ):
        house_id += 1  # NILMTK starts in 1, not in 0
        print("Loading house {house_id}".format(house_id=house_id), end="...")
        stdout.flush()

        for chan_id, chann_file in enumerate(
            glob.glob(join(input_path, house_dir, "elec", "meter*.csv"))
        ):
            chan_id += 1  # NILMTK start in 1, not in 0
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=house_id, meter=chan_id)

            csv_filename = join(input_path, house_dir, "elec", chann_file)

            chan_df = pd.read_csv(
                csv_filename,
                index_col="datetime",
                # parse_dates=["datetime"],
                # date_parser=pd.to_datetime,
                # dtype=[datetime, float]
            )
            chan_df.index = pd.to_datetime(chan_df.index)
            chan_df = chan_df.sort_index()
            chan_df.columns = pd.MultiIndex.from_tuples([("power", "active")])

            # Modify the column labels to reflect the power measurements recorded.
            chan_df.columns.set_names(LEVEL_NAMES, inplace=True)

            store.put(str(key), chan_df)

        print("..END")


if __name__ == "__main__":
    convert_data(
        input_path='/Users/jp/Documents/FIng/PruebasNILM/synthetic_dataset_1YEAR_UKDALE_house1',
        output_dir='/Users/jp/Documents/FIng/PruebasNILM/synthetic_dataset_1YEAR_UKDALE_house1-datetimes.h5'
    )
