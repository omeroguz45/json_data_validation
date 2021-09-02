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
from tensorflow_data_validation.utils import stats_util
from tensorflow_metadata.proto.v0 import schema_pb2

@beam.ptransform_fn
@beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(pa.RecordBatch)
def decodeJSON(self, data_location, schema: Optional[schema_pb2.Schema] = None):
    df = pd.read_json(data_location, orient='records')
    rb = pa.RecordBatch.from_pandas(df, schema=schema)

    return rb

def validate_examples_in_JSON(data_location, stats_options, output_path: Optional[Text] = None):
    """Validate examples in JSON.
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
            | 'Read and Decode JSON' >> decodeJSON(data_location=data_location, schema=stats_options.schema if stats_options.infer_type_from_schema else None)
            | 'Detect Anomalies' >> validation_api.IdentifyAnomalousExamples(stats_options)
            | 'GenerateSummaryStatistics' >> stats_impl.GenerateSlicedStatisticsImpl(stats_options, is_slicing_enabled=True)
            | 'WriteStatsOutput' >> stats_api.WriteStatisticsToTFRecord(output_path)
        )

    return stats_util.load_statistics(output_path)
