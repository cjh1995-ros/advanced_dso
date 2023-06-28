from dataclasses import dataclass, field
import numpy as np

@dataclass
class Pixel:
    u:          float
    v:          float
    ui:         int
    vi:         int
    intensity:  float

@dataclass
class Point:
    pixel:      Pixel
    i_depth:    float
    x:          float
    y:          float
    z:          float
    xyz:        np.ndarray = field(init=False)