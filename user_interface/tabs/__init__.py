# TTS UI Tabs Package
from .inference_tab import create_inference_tab
from .data_tab import create_data_tab
from .training_tab import create_training_tab

__all__ = [
    'create_inference_tab',
    'create_data_tab', 
    'create_training_tab'
] 