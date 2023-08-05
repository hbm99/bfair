from autogoal.sampling import Sampler

from bfair.methods.autogoal.ensembling.sampling import SampleModel
from bfair.sensors.handler import SensorHandler
from bfair.sensors.image.clip.base import ClipBasedSensor

def optimize():
    pass

def generate(sampler: Sampler) -> SampleModel:
    """
    Generates a new SampleModel object with the given Sampler.

    Args:
        sampler (Sampler): The Sampler to use for generating samples.

    Returns:
        SampleModel: A new SampleModel object with the given Sampler.
    """
    handler = SensorHandler([ClipBasedSensor()], merge=None)
    return SampleModel(sampler, handler)

