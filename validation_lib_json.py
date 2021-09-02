import os
import pandas as pd
import pyarrow as pa
import apache_beam as beam

import tempfile
import tensorflow as tf

from typing import Text, Optional
from tensorflow_data_validation.api import validation_api
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.statistics import stats_impl
from tensorflow_data_validation.statistics import stats_options as options
from tensorflow_data_validation.utils import stats_util
from tensorflow_metadata.proto.v0 import schema_pb2

@beam.ptransform_fn
@beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(pa.RecordBatch)
def decodeJSON(self, data_location: Text, orient: Text, schema: Optional[schema_pb2.Schema] = None):
    """
    Decodes JSON files into PyArrow RecordBatches.
    
    Takes the data_location, orientation and the schema (optional) of a JSON file. The
    file is read and converted to a Pandas dataframe, then into RecordBatches with an
    optional schema.

    Args:
      data_location: The location of the input JSON file.
      orient: The orientation of the JSON file. Can be:
        'split' : dict like {index -> [index], columns -> [columns], data -> [values]}
        'records' : list like [{column -> value}, ... , {column -> value}]
        'index' : dict like {index -> {column -> value}}
        'columns' : dict like {column -> {index -> value}}
        'values' : just the values array
      schema: Schema of the data in the JSON file. Optional.

    Returns:
      RecordBatches of the JSON file.
    """
    df = pd.read_json(data_location, orient=orient)
    rb = pa.RecordBatch.from_pandas(df, schema=schema)

    return rb

def validate_examples_in_JSON(data_location: Text, orient: Text, stats_options: options.StatsOptions, output_path: Optional[Text] = None):
    """
    Validate examples in JSON.

    Uses a beam pipeline to validate the data in a JSON file with the given StatsOptions.
    Generates summary statistics and writes it on to a file on the output_path. If not found, the program creates a file and writes the stats onto it.

    Args:
      data_location: The location of the input JSON file.
      orient: The orientation of the JSON file. Can be:
        'split' : dict like {index -> [index], columns -> [columns], data -> [values]}
        'records' : list like [{column -> value}, ... , {column -> value}]
        'index' : dict like {index -> {column -> value}}
        'columns' : dict like {column -> {index -> value}}
        'values' : just the values array
      stats_options: StatsOptions for the data to be compared and validated against.
      output_path: The location of the output tfrecords file for the statistics to be written on. Optional.

    Returns:
      A DatasetFeatureStatisticsList proto with the stats of the JSON file.
    """

    if stats_options.schema is None:
        raise ValueError('The specified stats_options must include a schema.')
    if output_path is None:
        output_path = os.path.join(tempfile.mkdtemp(), 'anomaly_stats.tfrecord')
    output_dir_path = os.path.dirname(output_path)
    if not tf.io.gfile.exists(output_dir_path):
        tf.io.gfile.makedirs(output_dir_path)

    with beam.Pipeline() as p:
        _ = (
            p
            | 'Read and Decode JSON' >> decodeJSON(data_location=data_location, orient=orient, schema=stats_options.schema if stats_options.infer_type_from_schema else None)
            | 'Detect Anomalies' >> validation_api.IdentifyAnomalousExamples(stats_options)
            | 'GenerateSummaryStatistics' >> stats_impl.GenerateSlicedStatisticsImpl(stats_options, is_slicing_enabled=True)
            | 'WriteStatsOutput' >> stats_api.WriteStatisticsToTFRecord(output_path)
        )

    return stats_util.load_statistics(output_path)
