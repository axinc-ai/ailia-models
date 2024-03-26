from typing import Dict, List, Literal

# from typing_extensions import Literal

MetricComponent = str
CalibrationMethod = Literal["isotonic", "sigmoid"]
MetricComponents = List[MetricComponent]
Details = Dict[MetricComponent, float]