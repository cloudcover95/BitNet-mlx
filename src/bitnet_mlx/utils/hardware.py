# src/bitnet_mlx/utils/hardware.py
import psutil
import subprocess
import logging
import re
from typing import Tuple
from pydantic_settings import BaseSettings
from pydantic import Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JuniorHomeGovernor")

class OmniSovereignConfig(BaseSettings):
    thermal_throttle_limit: float = Field(default=85.0, description="Thermal limit before AMX degradation")
    performance_profile: str = Field(default="AUTO", description="AUTO, OFFGRID_48V, UNRESTRICTED, MOBILE_A_SERIES")
    powermetrics_enabled: bool = Field(default=True, description="Attempt sudo powermetrics polling")
    max_shard_size_gb: float = Field(default=2.0, description="Max safetensor chunk size for mobile mesh nodes")

    class Config:
        env_prefix = "JC_OMNI_"

settings = OmniSovereignConfig()

class HardwareGovernor:
    """Monitors Apple Silicon topology (A-Series / M-Series) for 48V JuniorHome thermal routing."""
    
    @staticmethod
    def get_soc_topology() -> Tuple[str, str]:
        try:
            hw_model = subprocess.check_output(["sysctl", "-n", "hw.model"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            hw_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            return hw_model, hw_brand
        except Exception:
            try:
                hw_machine = subprocess.check_output(["sysctl", "-n", "hw.machine"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
                return hw_machine, "A-Series (Mobile)"
            except Exception:
                return "Unknown", "Apple Silicon (Unverified)"

    @staticmethod
    def _poll_powermetrics() -> float:
        if not settings.powermetrics_enabled: return -1.0
        try:
            out = subprocess.check_output(
                ["sudo", "-n", "powermetrics", "-n", "1", "--samplers", "smc"], 
                stderr=subprocess.DEVNULL, timeout=2.0
            ).decode("utf-8")
            match = re.search(r"CPU thermal level:\s*(\d+)", out)
            return float(match.group(1)) if match else -1.0
        except Exception:
            return -1.0

    @staticmethod
    def get_system_load() -> Tuple[float, float, float]:
        cpu_load = psutil.cpu_percent(interval=0.1)
        thermal_level = HardwareGovernor._poll_powermetrics()
        if thermal_level >= 0:
            cpu_load = max(cpu_load, thermal_level)
            
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return cpu_load, mem.available / (1024 ** 3), swap.used / (1024 ** 3)

    @staticmethod
    def calculate_quantization_ceiling() -> float:
        """Dynamic dial for execution bounds guarding against off-grid 48V dropout."""
        if settings.performance_profile == "UNRESTRICTED": return 15.0
            
        cpu_load, _, swap_used = HardwareGovernor.get_system_load()
        hw_model, hw_brand = HardwareGovernor.get_soc_topology()
        
        is_mobile = "iPhone" in hw_model or "iPad" in hw_model or settings.performance_profile == "MOBILE_A_SERIES"
        is_heavy_iron = any(tier in hw_brand for tier in ["Max", "Ultra"])
        
        if is_mobile or cpu_load > settings.thermal_throttle_limit or swap_used > 2.0:
            return 3.0  
        elif is_heavy_iron and cpu_load < 50.0:
            return 15.0 
        return 7.0