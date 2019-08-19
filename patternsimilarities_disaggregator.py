from __future__ import print_function, division

import random
from datetime import datetime
from sys import stdout

import pandas as pd

from nilmtk.disaggregate import Disaggregator
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk import ElecMeter


class PatternSimilaritiesDisaggregator(Disaggregator):
    """Provides a common interface to all disaggregation classes.

    See https://github.com/nilmtk/nilmtk/issues/271 for discussion, and
    nilmtk/docs/manual/development_guide/writing_a_disaggregation_algorithm.md
    for the development guide.

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.

    MODEL_NAME : string
        A short name for this type of model.
        e.g. 'CO' for combinatorial optimisation.
        Used by self._save_metadata_for_disaggregation.
    """

    def __init__(self):
        self.MODEL_NAME = "PatternSimilaritiesUDELAR"

    def train(
        self,
        metergroup,
        neighbourhood,
        time_interval_neighbourhood,
        H,
        tolerance,
        sample_period=None,
    ):
        """Trains the model given a metergroup containing appliance meters
        (supervised) or a site meter (unsupervised).  Will have a
        default implementation in super class.  Can be overridden for
        simpler in-memory training, or more complex out-of-core
        training.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        :param neighbourhood: defines an energy consumption neighbourhood
        :param time_interval_neighbourhood: define a time interval neighbourhood
        :param H: parameter that separates high from low energy consumption
        :param tolerance: tolerance parameter
        :param sample_period:
        """
        self.metergroup = metergroup
        self.neighbourhood = neighbourhood
        self.time_interval_neighbourhood = time_interval_neighbourhood
        self.H = H
        self.tolerance = tolerance
        self.sample_period = sample_period
        self.meter_series = {}

        d = self.time_interval_neighbourhood

        # Lines 1-9 of algorithm
        self.M_z = {}
        # Evaluate for each main of each building
        mains = self._get_train_mains()
        for main in mains:
            b_num = main.building()
            aggregated_meter = self._get_power_series_all_data(main)
            amount_train_mains_records = aggregated_meter.size
            self.M_z[b_num] = []
            # Fill the beginning with zeros (as many as d)
            for i in range(d):
                self.M_z[b_num].append(0)

            for i in range(1 + d, amount_train_mains_records - d):
                self.M_z[b_num].append(
                    # Count how many neightbour consumption are greater
                    # Lines 3-8 of algorithm
                    aggregated_meter[i - d : i + d][
                        aggregated_meter > (aggregated_meter[i] - self.tolerance)
                    ].size
                )

            # Fill the end with zeros (as many as d)
            for i in range(d):
                self.M_z[b_num].append(0)

    def train_on_chunk(self, chunk, meter):
        """Signature is fine for site meter dataframes (unsupervised
        learning). Would need to be called for each appliance meter
        along with appliance identifier for supervised learning.
        Required to be overridden to provide out-of-core
        disaggregation.

        Parameters
        ----------
        chunk : pd.DataFrame where each column represents a
            disaggregated appliance
        meter : ElecMeter for this chunk
        """
        raise NotImplementedError("User 'train' method. Are equal.")

    def _get_train_mains(self):
        mains = self.metergroup.mains()
        # method mains() can return a MeterGroup (which is iterable) or directly
        # the unique ElecMeter that is main
        if isinstance(mains, ElecMeter):
            mains = [mains]

        return mains

    def _get_power_series_all_data(self, meter):
        """Given a meter, load the serie or take it from the memory cache"""

        b_series = self.meter_series.get(meter.building())
        if b_series is not None:
            serie = b_series.get(meter.instance())
        else:
            self.meter_series[meter.building()] = {}
            serie = None

        if serie is not None:
            return serie
        else:
            self.meter_series[meter.building()][meter.instance()] = meter.power_series_all_data(
                sample_period=self.sample_period
            )
            return self.meter_series[meter.building()][meter.instance()]

    def disaggregate(self, mains, output_datastore):
        """Passes each chunk from mains generator to disaggregate_chunk() and
        passes the output to _write_disaggregated_chunk_to_datastore()
        Will have a default implementation in super class.  Can be
        overridden for more simple in-memory disaggregation, or more
        complex out-of-core disaggregation.

        Parameters
        ----------
        mains : nilmtk.ElecMeter (single-phase) or
            nilmtk.MeterGroup (multi-phase)
        output_datastore : instance of nilmtk.DataStore or str of
            datastore location
        """
        d = self.time_interval_neighbourhood

        # Lines 1-9 of algorithm are part of the training phase

        # Load the data (serie of aggregated) to be disaggregated
        mains_to_dissag = self._get_power_series_all_data(mains)
        amount_mains_to_dissag_records = mains_to_dissag.size

        # Lines 10-18 of algorithm
        M_x = []
        # Fill the beginning with zeros (as many as d)
        for i in range(d):
            M_x.append(0)
        for i in range(1 + d, amount_mains_to_dissag_records - d):
            M_x.append(
                # Count how many neightbour consumption are greater
                # Lines 12-17 of algorithm
                mains_to_dissag[i - d : i + d][
                    mains_to_dissag > (mains_to_dissag[i] - self.tolerance)
                ].size
            )
        # Fill the end with zeros (as many as d)
        for i in range(d):
            M_x.append(0)

        # Indicates if metadaca must be saved
        data_is_available = False

        # TODO: datos para construir datastore de salida, reubicar
        # OJO! esto es asi pq no lo he generalizado a varios buildings
        # timeframes = []
        building_path = "/building{}".format(mains.building())
        mains_data_location = building_path + "/elec/meter1"
        cols = pd.MultiIndex.from_tuples([mains_to_dissag.name])

        train_mains = self._get_train_mains()
        # Lines 19 to end
        for i in range(1 + d, amount_mains_to_dissag_records - d):
            I = set()
            for main in train_mains:
                b_num = main.building()
                aggregated_meter = self._get_power_series_all_data(main)
                aggregated_meter_amount = aggregated_meter.size

                for j in range(d, aggregated_meter_amount - d):
                    if (
                        mains_to_dissag[i] > self.H
                        and abs(aggregated_meter[j] - mains_to_dissag[i])
                        <= self.neighbourhood
                    ):
                        I.add((b_num, j))

            # Lines 26 to 34
            # I is a set with tuples (building number <int>, position of record <int>)
            I = list(I)  # to assure that keeps order
            if I:
                diff_similarities = [
                    (j_b, j_i, abs(self.M_z[j_b][j_i] - M_x[i]))
                    for j_b, j_i in I
                ]
            else:
                diff_similarities = []
                for main in self._get_train_mains():
                    main_serie = self._get_power_series_all_data(main)
                    b_num = main.building()

                    diff_similarities += [
                        (b_num, index, abs(value - mains_to_dissag[i]))
                        for index, value in enumerate(main_serie)
                    ]
            del I
            min_diff = min(diff_similarities, key=lambda x: x[2])[2]
            J = [(b, i) for b, i, val in diff_similarities if val == min_diff]
            del diff_similarities
            # This two values are the index of the training record that is similar to
            # the current moment of dissagregation. Prediction consist in taking the
            # values of the appliances at this moment as the prediction.
            build_pred, positin_pred = random.choice(J)
            del J

            main = None
            for main in self._get_train_mains():
                if main.building() == build_pred:
                    break
            # aggregated_of_pred = self._get_power_series_all_data(
            #     main
            # )[positin_pred]
            datetime_pred = self._get_power_series_all_data(
                main
            ).index[positin_pred]

            # Datetime of the moment most similar to the one we are desegregating
            # datetime_pred = aggregated_of_pred.index

            for meter in (
                m for m in self.metergroup.all_meters()
                if m.building() == build_pred
                and not m.is_site_meter()
            ):
                data_is_available = True
                meter_instance = meter.instance()
                predicted_power = self._get_power_series_all_data(
                    meter
                )[datetime_pred:datetime_pred]
                output_df = pd.DataFrame(
                    data=predicted_power.values,
                    index=mains_to_dissag.index[i:i+1]  # Replace train index by data to dessagregate
                )
                output_df.columns = pd.MultiIndex.from_tuples([mains_to_dissag.name])
                # output_df.index = mains_to_dissag.index[i:i+1]  # Replace train index by data to dessagregate
                key = "{}/elec/meter{}".format(building_path, meter_instance)
                output_datastore.append(key, output_df)

            if i % 100 == 0:
                print("{}...".format(i), end="")
                stdout.flush()

        print("END TESTING")
        stdout.flush()
        # Copy mains data to disag output
        output_datastore.append(
            key=mains_data_location, value=pd.DataFrame(mains_to_dissag, columns=cols)
        )
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=self.sample_period,
                measurement=mains_to_dissag.name,
                timeframes=[mains.get_timeframe()],
                building=mains.building(),
                meters=self.metergroup.all_meters()
            )

    def disaggregate_chunk(self, mains):
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.DataFrame

        Returns
        -------
        appliances : pd.DataFrame where each column represents a
            disaggregated appliance
        """
        raise NotImplementedError()

    def _pre_disaggregation_checks(self, load_kwargs):
        if not self.model:
            raise RuntimeError(
                "The model needs to be instantiated before"
                " calling `disaggregate`.  For example, the"
                " model can be instantiated by running `train`."
            )

        if "resample_seconds" in load_kwargs:
            DeprecationWarning(
                "'resample_seconds' is deprecated."
                "  Please use 'sample_period' instead."
            )
            load_kwargs["sample_period"] = load_kwargs.pop("resample_seconds")

        return load_kwargs

    def _save_metadata_for_disaggregation(
        self,
        output_datastore,
        sample_period,
        measurement,
        timeframes,
        building,
        meters=None,
        num_meters=None,
        supervised=True,
    ):
        """Add metadata for disaggregated appliance estimates to datastore.

        This method returns nothing.  It sets the metadata
        in `output_datastore`.

        Note that `self.MODEL_NAME` needs to be set to a string before
        calling this method.  For example, we use `self.MODEL_NAME = 'CO'`
        for Combinatorial Optimisation.

        Parameters
        ----------
        output_datastore : nilmtk.DataStore subclass object
            The datastore to write metadata into.
        sample_period : int
            The sample period, in seconds, used for both the
            mains and the disaggregated appliance estimates.
        measurement : 2-tuple of strings
            In the form (<physical_quantity>, <type>) e.g.
            ("power", "active")
        timeframes : list of nilmtk.TimeFrames or nilmtk.TimeFrameGroup
            The TimeFrames over which this data is valid for.
        building : int
            The building instance number (starting from 1)
        supervised : bool, defaults to True
            Is this a supervised NILM algorithm?
        meters : list of nilmtk.ElecMeters, optional
            Required if `supervised=True`
        num_meters : int
            Required if `supervised=False`
        """

        # TODO: `preprocessing_applied` for all meters
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        building_path = "/building{}".format(building)
        mains_data_location = building_path + "/elec/meter1"

        meter_devices = {
            self.MODEL_NAME: {
                "model": self.MODEL_NAME,
                "sample_period": sample_period,
                "max_sample_period": sample_period,
                "measurements": [
                    {"physical_quantity": measurement[0], "type": measurement[1]}
                ],
            },
            "mains": {
                "model": "mains",
                "sample_period": sample_period,
                "max_sample_period": sample_period,
                "measurements": [
                    {"physical_quantity": measurement[0], "type": measurement[1]}
                ],
            },
        }

        merged_timeframes = merge_timeframes(timeframes, gap=sample_period)
        total_timeframe = TimeFrame(
            merged_timeframes[0].start, merged_timeframes[-1].end
        )

        date_now = datetime.now().isoformat().split(".")[0]
        dataset_metadata = {
            "name": self.MODEL_NAME,
            "date": date_now,
            "meter_devices": meter_devices,
            "timeframe": total_timeframe.to_dict(),
        }
        output_datastore.save_metadata("/", dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                "device_model": "mains",
                "site_meter": True,
                "data_location": mains_data_location,
                "preprocessing_applied": {},  # TODO
                "statistics": {"timeframe": total_timeframe.to_dict()},
            }
        }

        def update_elec_meters(meter_instance):
            elec_meters.update(
                {
                    meter_instance: {
                        "device_model": self.MODEL_NAME,
                        "submeter_of": 1,
                        "data_location": (
                            "{}/elec/meter{}".format(building_path, meter_instance)
                        ),
                        "preprocessing_applied": {},  # TODO
                        "statistics": {"timeframe": total_timeframe.to_dict()},
                    }
                }
            )

        # Appliances and submeters:
        appliances = []
        if supervised:
            for meter in meters:
                meter_instance = meter.instance()
                update_elec_meters(meter_instance)

                for app in meter.appliances:
                    appliance = {
                        "meters": [meter_instance],
                        "type": app.identifier.type,
                        "instance": app.identifier.instance
                        # TODO this `instance` will only be correct when the
                        # model is trained on the same house as it is tested on
                        # https://github.com/nilmtk/nilmtk/issues/194
                    }
                    appliances.append(appliance)

                # Setting the name if it exists
                if meter.name:
                    if len(meter.name) > 0:
                        elec_meters[meter_instance]["name"] = meter.name
        else:  # Unsupervised
            # Submeters:
            # Starts at 2 because meter 1 is mains.
            for chan in range(2, num_meters + 2):
                update_elec_meters(meter_instance=chan)
                appliance = {
                    "meters": [chan],
                    "type": "unknown",
                    "instance": chan - 1
                    # TODO this `instance` will only be correct when the
                    # model is trained on the same house as it is tested on
                    # https://github.com/nilmtk/nilmtk/issues/194
                }
                appliances.append(appliance)

        building_metadata = {
            "instance": building,
            "elec_meters": elec_meters,
            "appliances": appliances,
        }

        output_datastore.save_metadata(building_path, building_metadata)

    def _write_disaggregated_chunk_to_datastore(self, chunk, datastore):
        """ Writes disaggregated chunk to NILMTK datastore.
        Should not need to be overridden by sub-classes.

        Parameters
        ----------
        chunk : pd.DataFrame representing a single appliance
            (chunk needs to include metadata)
        datastore : nilmtk.DataStore
        """
        raise NotImplementedError()

    def import_model(self, filename):
        """Loads learned model from file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to load model from
        """
        raise NotImplementedError()

    def export_model(self, filename):
        """Saves learned model to file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to save model to
        """
        raise NotImplementedError()


if __name__ == "__main__":

    """Test the dissagregator"""
    print("Start test.")

    from nilmtk import HDFDataStore, ElecMeter


    class ElecStub(object):
        """
                TRAIN DATA
                6|          +
                5|          + +
                4|          + +
                3|    +     + +
                2|    + + + + + + +
                1|    + + + + + + +
                0|=================
                  0 1 2 3 4 5 6 7 8
                """

        data_train_main = pd.Series(
            {0: 0, 1: 0, 2: 3, 3: 2, 4: 2, 5: 6, 6: 5, 7: 2, 8: 2}
        )
        """
        2|
        1|    +     +      
        0|=================
          0 1 2 3 4 5 6 7 8
        """
        data_train_elec1 = pd.Series(
            {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0}
        )
        """
        4|
        3|          + +
        2|          + +
        1|          + +    
        0|=================
          0 1 2 3 4 5 6 7 8
        """
        data_train_elec2 = pd.Series(
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 3, 6: 3, 7: 0, 8: 0}
        )
        """
        3|
        2|    + + + + + + +
        1|    + + + + + + +
        0|=================
          0 1 2 3 4 5 6 7 8
        """
        data_train_elec3 = pd.Series(
            {0: 0, 1: 0, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}
        )
        elecs = [data_train_elec1, data_train_elec2, data_train_elec3]

        def __iter__(self):
            self._iter_elec = 0
            return self

        def __next__(self):
            if self._iter_elec < len(self.elecs):
                self._iter_elec += 1
                return self.elecs[self._iter_elec]
            else:
                raise StopIteration

        def mains(self):
            return self

        def power_series_all_data(self):
            return self.data_train_main

    class MeterGroupStub(object):
        """Stub class for testing"""

        """
        6|          
        5|          + +
        4|          + +
        3|      +   + +
        2|+ + + + + + + + +
        1|+ + + + + + + + +
        0|=================
          0 1 2 3 4 5 6 7 8
        """
        data_testing = pd.Series({0: 2, 1: 2, 2: 2, 3: 3, 4: 2, 5: 5, 6: 5, 7: 2, 8: 2})

        def __init__(self, is_train_meter):
            # self.buildings = {1: self}  # metergroup no tiene buildings
            if is_train_meter:
                self.data = ElecStub.data_train_main
                self.elec = ElecStub()
            else:
                self.data = self.data_testing
                self.elec = self

        def power_series_all_data(self):
            return self.data

        def mains(self):
            return self

        def building(self):
            return 1

    delta = 3  # neighbourhood
    d = 2  # time interval
    H = 0  # separates high from low consumption
    phi = 2  # tolerance to differences

    PS = PatternSimilaritiesDisaggregator()

    print("Train model.")
    PS.train(
        MeterGroupStub(is_train_meter=True),
        neighbourhood=delta,
        time_interval_neighbourhood=d,
        H=H,
        tolerance=phi,
    )
    filename = "./testing_patterns_similarities_disaggregator.h5"
    prediction_alg = HDFDataStore(filename, mode="w")
    print("Dissagregate.")
    PS.disaggregate(MeterGroupStub(is_train_meter=False), prediction_alg)
    prediction_alg.close()
    del prediction_alg

    print("End test.")