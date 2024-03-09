from typing import Dict, List

from typing_extensions import Literal

MetricComponent = str
CalibrationMethod = Literal["isotonic", "sigmoid"]
MetricComponents = List[MetricComponent]
Details = Dict[MetricComponent, float]