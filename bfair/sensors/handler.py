from typing import Sequence, List, Callable
from bfair.sensors.base import Sensor
from autogoal.kb import SemanticType


class SensorHandler:
    def __init__(
        self,
        sensors: Sequence[Sensor],
        merge: Callable[[List[Sequence[str]]], Sequence[str]] = None,
    ):
        self.sensors = sensors
        self.merge = merge if merge else UnionMerge()

    def __call__(
        self,
        item,
        attributes: List[str],
        attr_cls: str,
        *,
        stype: SemanticType = None,
        use_all_sensors_if_inference_fails=True,
    ):
        stype = self._try_to_infer(item, stype=stype)

        if stype is not None:
            return self.annotate(item, stype, attributes, attr_cls)
        elif use_all_sensors_if_inference_fails:
            return self.forward(item, self.sensors, attributes, attr_cls)
        raise ValueError(
            "Unable to infer semantic type. Either pass it explicitly using `stype=...` or set `use_all_sensors_if_inference_fails=True`"
        )

    def annotate(self, data, stype: SemanticType, attributes: List[str], attr_cls: str):
        sensors = self.select(stype, attr_cls)
        final = self.forward(data, sensors, attributes, attr_cls)
        return final

    def select(self, stype: SemanticType, attr_cls: str) -> List[Sensor]:
        return [sensor for sensor in self.sensors if sensor.match(stype, attr_cls)]

    def forward(
        self,
        data,
        sensors: List[Sensor],
        attributes: List[str],
        attr_cls: str,
    ):
        annotations = [sensor(data, attributes, attr_cls) for sensor in sensors]
        final = self.merge(annotations)
        return final

    def _try_to_infer(self, item, *, stype: SemanticType = True):
        if stype is not None:
            return stype
        try:
            return SemanticType.infer(item)
        except ValueError:
            return None


class UnionMerge:
    def __call__(self, annotations):
        return set([attr for attributes in annotations for attr in attributes])


class IntersectionMerge:
    def __call__(self, annotations):
        return set.intersection(*(set(attributes) for attributes in annotations))
