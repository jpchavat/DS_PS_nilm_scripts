import glob
import os
import re

import matplotlib
# Comment to see plots in UI
matplotlib.use('Agg')  # avoid using tkinter (not installed in cluster.uy)

import pandas as pd
from os.path import join
from sys import stdout

import sys
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk')
sys.path.insert(0, '/Users/jp/Documents/FIng/nilmtk_metadata')
sys.path.insert(0, '/clusteruy/home/jpchavat/nilmtk')
sys.path.insert(0, '/clusteruy/home/jpchavat/nilm_metadata')

from nilmtk import DataStore
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import get_datastore, get_module_directory, check_directory_exists
from nilm_metadata import save_yaml_to_datastore
from nilm_metadata.convert_yaml_to_hdf5 import _load_file


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

    yaml_dir = join(get_module_directory(), input_path, "metadata")
    # metadata = _load_file(yaml_dir, 'dataset.yaml')

    # Load timezone, if exists, if not, use UTC
    # tz = metadata.get('timezone', 'UTC')
    # print("Timezone to be used: {}".format(tz))

    # Convert raw data to DataStore
    _convert(input_path, store)

    # Add metadata
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
    tz : str
        String with timezone to use in index parser
    """

    check_directory_exists(input_path)
    sub_dirs = list(os.walk(input_path))[0][1]

    # Iterate though all houses and channels

    for house_id, house_dir in enumerate(
        d for d in sub_dirs if d.startswith("building")
    ):
        house_id += 1  # NILMTK starts in 1, not in 0
        print("> Loading house {house_id}".format(house_id=house_id))
        stdout.flush()

        general_index = None

        for chann_file in glob.glob(join(input_path, house_dir, "elec", "meter*.csv")):
            chan_id = int(re.findall(".*meter(\d+)\.csv$", chann_file)[0])
            # chan_id += 1  # NILMTK start in 1, not in 0
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=house_id, meter=chan_id)

            csv_filename = join(input_path, house_dir, "elec", chann_file)

            print("Loading CSV file {}".format(csv_filename), end="...")
            stdout.flush()
            chan_df = pd.read_csv(
                csv_filename,
                index_col="datetime",
                # parse_dates=[0],
            )
            print("END")
            stdout.flush()
            # chan_df.index = chan_df.index.tz_localize('UTC')
            # chan_df.index = chan_df.index.tz_localize(
            #     tz=tz,
            #     # infer_dst=True,
            #     # ambiguous='infer'
            # )

            print("Setting index properties...", end="...")
            stdout.flush()

            if general_index is None:
                chan_df.index = pd.to_datetime(chan_df.index, utc=True)
                # general_index = chan_df.index.copy()  # inmutable ?
                general_index = chan_df.index
            else:
                # chan_df.index = general_index.copy()  # inmutable ?
                chan_df.index = general_index

            chan_df = chan_df.sort_index()
            chan_df.columns = pd.MultiIndex.from_tuples([("power", "active")])

            print("END")
            stdout.flush()

            # Modify the column labels to reflect the power measurements recorded.
            chan_df.columns.set_names(LEVEL_NAMES, inplace=True)

            store.put(str(key), chan_df)

        print("..END")


if __name__ == "__main__":
    convert_data(
        # input_path='/Users/jp/Documents/FIng/Maestria/nilm-scripts/datasets_csv_metadata/synthetic_1YEAR_UKDALE_house1_UTC_articulo_CONSTVALUES',
        input_path='/Users/jp/Documents/FIng/Maestria/nilm-scripts/datasets_csv_metadata/synthetic_1YEAR_UKDALE_house1_UTC_articulo_NOISE-3',
        # output_dir='/Users/jp/Documents/FIng/PruebasNILM/synthetic_dataset_1YEAR_UKDALE_house1_UTC_CONSTVALUES.h5'.format(
        output_dir='/Users/jp/Documents/FIng/PruebasNILM/synthetic_1YEAR_UKDALE_house1_UTC_articulo_NOISE-3.h5'
    )
