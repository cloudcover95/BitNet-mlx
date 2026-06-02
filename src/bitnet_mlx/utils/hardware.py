# src/bitnet_mlx/utils/hardware.py
import psutil
import subprocess
import logging
from typing import Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OmniGovernor")

class EdgePowerConfig(BaseSettings):
    throttle_limit: float = Field(default=85.0, description="CPU threshold before matrix degradation")
    performance_profile: str = Field(default="AUTO", description="AUTO, OFFGRID_48V, UNRESTRICTED")
    model_config = SettingsConfigDict(env_prefix="JC_HW_", extra="ignore")

settings = EdgePowerConfig()

class HardwareGovernor:
    """Configures edge logic limits based on power availability and thermals."""
    @staticmethod
    def get_soc_topology() -> Tuple[str, str]:
        try:
            hw_model = subprocess.check_output(["sysctl", "-n", "hw.model"], stderr=subprocess.DEVNULL).decode().strip()
            hw_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL).decode().strip()
            return hw_model, hw_brand
        except Exception:
            return "Unknown", "Apple Silicon"

    @staticmethod
    def get_system_load() -> Tuple[float, float, float]:
        return psutil.cpu_percent(interval=0.1), psutil.virtual_memory().available / (1024 ** 3), psutil.swap_memory().used / (1024 ** 3)

    @staticmethod
    def calculate_quantization_ceiling() -> float:
        if settings.performance_profile == "UNRESTRICTED": 
            return 15.0
            
        cpu_load, _, swap_used = HardwareGovernor.get_system_load()
        hw_model, hw_brand = HardwareGovernor.get_soc_topology()
        
        if "iPhone" in hw_model or "iPad" in hw_model or cpu_load > settings.throttle_limit or swap_used > 2.0:
            return 3.0  
        elif any(t in hw_brand for t in ["Max", "Ultra", "M4"]) and cpu_load < 50.0:
            return 15.0 
        return 7.0