import psutil
import logging

logger = logging.getLogger("BitNet.PowerMatrix")

class ThermalSubstrate:
@staticmethod
def query_thermal_state() -> str:
"""Dynamic Apple Silicon Thermal Query. Downshifts precision if >85%."""
try:
if psutil.cpu_percent(interval=0.1) > 85.0:
return "a4"
return "a8"

except Exception:
return "a8"
