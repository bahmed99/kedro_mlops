"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from new_project.pipelines import data_processing as dp
from new_project.pipelines import modeling as ds
def register_pipelines() -> Dict[str, Pipeline]:
    data_processing_pipeline = dp.create_pipeline()
    modeling_pipeline = ds.create_pipeline()
    return {  "__default__": data_processing_pipeline+ modeling_pipeline,
        "dp": data_processing_pipeline, "ds": modeling_pipeline}
