from rich.console import Console
from ..evaluation.junior_tda import JuniorCloudTDAEvaluator
from ..utils.hardware import HardwareGovernor

console = Console()

class JuniorSandboxSuite:
    @staticmethod
    def execute_validation_matrix() -> bool:
        console.print("[bold yellow][*] Booting JuniorCloud Omni-Sovereign Sandbox...[/bold yellow]")
        hw_model, hw_brand = HardwareGovernor.get_soc_topology()
        cpu, mem, swap = HardwareGovernor.get_system_load()
        q_ceiling = HardwareGovernor.calculate_quantization_ceiling()
        
        console.print("\n[bold cyan]--- JuniorHome Edge Telemetry ---[/bold cyan]")
        console.print(f"Hardware: [white]{hw_model} ({hw_brand})[/white]")
        console.print(f"Compute/Thermal Load: {cpu}% | RAM Free: {mem:.2f}GB | Swap: {swap:.2f}GB")
        console.print(f"Active TDA Ceiling (q_max): [bold green]{q_ceiling}[/bold green]")
        
        console.print("\n[bold cyan]--- JuniorQuant TDA Math Check ---[/bold cyan]")
        success = JuniorCloudTDAEvaluator.run_validation_suite()
        if success:
            console.print("\n[bold green][+] JuniorCoach backend matrix is fully stable and sovereign.[/bold green]")
        return success
