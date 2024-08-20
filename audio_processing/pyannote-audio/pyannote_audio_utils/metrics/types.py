from typing import Dict, List, Literal


MetricComponent = str
CalibrationMethod = Literal["isotonic", "sigmoid"]
MetricComponents = List[MetricComponent]
Details = Dict[MetricComponent, float]