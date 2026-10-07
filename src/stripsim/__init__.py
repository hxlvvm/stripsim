"""stripsim: lateral-flow assay simulator, from binding kinetics to phone photos to limit of detection."""
from . import assay, image, readers
from .assay import AssayResult, Strip, run
from .image import HARSH, LAB, PHONE, Conditions, reflectance, render
from .readers import READERS, lod95, matched_filter, tc_ratio

__all__ = ["assay", "image", "readers", "Strip", "AssayResult", "run", "Conditions", "LAB", "PHONE", "HARSH",
           "reflectance", "render", "READERS", "tc_ratio", "matched_filter", "lod95"]
__version__ = "0.1.0"
