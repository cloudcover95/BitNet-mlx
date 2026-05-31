# src/bitnet_mlx/utils/hardware.py
import psutil
import subprocess
import logging
from typing import Tuple
from pydantic_settings import BaseSettings
from pydantic import Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("OmniGovernor")

class OmniSovereignConfig(BaseSettings):
    thermal_throttle_limit: float = Field(default=90.0, description="CPU thermal threshold for degradation")
    performance_profile: str = Field(default="AUTO", description="AUTO, OFFGRID_48V, UNRESTRICTED")
    class Config:
        env_prefix = "JC_OMNI_"

settings = OmniSovereignConfig()

class HardwareGovernor:
    """Detects Apple Silicon topology and enforces safe edge-node envelopes (JuniorHome)."""
    
    @staticmethod
    def get_soc_topology() -> str:
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        except Exception:
            return "Apple Silicon (Unknown Topology)"

    @staticmethod
    def get_system_load() -> Tuple[float, float, float]:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return cpu, mem.available / (1024 ** 3), swap.used / (1024 ** 3)

    @staticmethod
    def calculate_quantization_ceiling() -> float:
        """Dials bounds dynamically based on active telemetry."""
        if settings.performance_profile == "UNRESTRICTED":
            return 15.0 # Maximize precision (near FP4)
            
        cpu_load, _, swap_used = HardwareGovernor.get_system_load()
        if settings.performance_profile == "OFFGRID_48V":
            return 3.0 if cpu_load > 60.0 else 7.0
            
        soc = HardwareGovernor.get_soc_topology()
        is_heavy_iron = any(tier in soc for tier in ["Max", "Ultra"])
        
        if cpu_load > settings.thermal_throttle_limit or swap_used > 2.0:
            return 3.0  # Degrade matrix bounds to cool 48V power package
        elif is_heavy_iron and cpu_load < 50.0:
            return 15.0 # Maximize precision for unloaded Ultra nodes
        return 7.0      # Optimal 1.58-bit equilibrium