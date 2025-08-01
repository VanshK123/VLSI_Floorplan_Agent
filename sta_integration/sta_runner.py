"""Runs external STA tool with real-time feedback integration."""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class STARunner:
    """Interface to invoke an STA binary with real-time feedback."""

    def __init__(self, binary_path: Path, timeout: int = 30, batch_size: int = 16) -> None:
        self.binary_path = binary_path
        self.timeout = timeout
        self.batch_size = batch_size
        
        # Performance tracking
        self.total_runs = 0
        self.successful_runs = 0
        self.average_runtime = 0.0
        self.critical_path_delays = []
        
        # Real-time feedback parameters
        self.violation_threshold = 0.1
        self.adaptation_rate = 0.05
        self.lyapunov_stability = True
        
        # Batch processing
        self.pending_evaluations = []
        self.batch_results = {}
        
        logging.info(f"STA Runner initialized with timeout={timeout}s, batch_size={batch_size}")

    def run(self, design_dir: Path) -> Path:
        """Execute STA and return path to report."""
        if not design_dir.exists():
            raise FileNotFoundError(f"Design directory not found: {design_dir}")
        
        start_time = time.time()
        
        try:
            # Create STA command
            cmd = self._create_sta_command(design_dir)
            
            # Execute STA with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Check for errors
            if result.returncode != 0:
                logging.error(f"STA failed with return code {result.returncode}")
                logging.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"STA execution failed: {result.stderr}")
            
            # Parse results
            report_path = self._parse_sta_output(result.stdout, design_dir)
            
            # Update performance metrics
            runtime = time.time() - start_time
            self._update_metrics(runtime, True)
            
            return report_path
            
        except subprocess.TimeoutExpired:
            logging.warning(f"STA evaluation timed out after {self.timeout}s")
            self._update_metrics(self.timeout, False)
            raise TimeoutError(f"STA evaluation timed out")
        except Exception as e:
            self._update_metrics(time.time() - start_time, False)
            raise

    async def run_async(self, design_dir: Path) -> Path:
        """Asynchronous STA execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, design_dir)

    def run_batch(self, design_dirs: List[Path]) -> List[Path]:
        """Execute STA on multiple designs in batch."""
        if not design_dirs:
            return []
        
        results = []
        batch_start = time.time()
        
        # Process in batches
        for i in range(0, len(design_dirs), self.batch_size):
            batch = design_dirs[i:i + self.batch_size]
            
            # Execute batch
            batch_results = []
            for design_dir in batch:
                try:
                    result = self.run(design_dir)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Batch evaluation failed for {design_dir}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        batch_time = time.time() - batch_start
        logging.info(f"Batch evaluation completed: {len(results)} designs in {batch_time:.2f}s")
        
        return results

    def _create_sta_command(self, design_dir: Path) -> List[str]:
        """Create STA command with appropriate arguments."""
        # This would be customized for the specific STA tool
        # For now, using a generic command structure
        
        cmd = [
            str(self.binary_path),
            "-design", str(design_dir / "design.def"),
            "-library", str(design_dir / "library.lib"),
            "-constraints", str(design_dir / "constraints.sdc"),
            "-output", str(design_dir / "sta_report.rpt"),
            "-format", "json"
        ]
        
        return cmd

    def _parse_sta_output(self, stdout: str, design_dir: Path) -> Path:
        """Parse STA output and extract timing information."""
        try:
            # Parse JSON output
            data = json.loads(stdout)
            
            # Extract critical path delay
            if 'timing' in data and 'critical_path' in data['timing']:
                critical_delay = data['timing']['critical_path']['delay']
                self.critical_path_delays.append(critical_delay)
                
                # Real-time feedback analysis
                self._analyze_timing_feedback(critical_delay)
            
            # Extract slack information
            if 'timing' in data and 'slack' in data['timing']:
                slack_info = data['timing']['slack']
                self._analyze_slack_violations(slack_info)
            
            # Return report path
            report_path = design_dir / "sta_report.rpt"
            return report_path
            
        except json.JSONDecodeError:
            logging.warning("Failed to parse STA output as JSON, using fallback parsing")
            return self._fallback_parse_sta_output(stdout, design_dir)

    def _fallback_parse_sta_output(self, stdout: str, design_dir: Path) -> Path:
        """Fallback parsing for non-JSON STA output."""
        # Extract timing information using regex or line parsing
        lines = stdout.split('\n')
        critical_delay = None
        
        for line in lines:
            if 'critical path delay' in line.lower():
                try:
                    critical_delay = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
        
        if critical_delay is not None:
            self.critical_path_delays.append(critical_delay)
            self._analyze_timing_feedback(critical_delay)
        
        return design_dir / "sta_report.rpt"

    def _analyze_timing_feedback(self, critical_delay: float):
        """Analyze timing feedback for closed-loop control."""
        if not self.lyapunov_stability:
            return
        
        # Calculate Lyapunov function value
        if len(self.critical_path_delays) >= 2:
            recent_delays = self.critical_path_delays[-10:]
            delay_variance = np.var(recent_delays)
            
            # Check for timing violations
            if critical_delay > self.violation_threshold:
                logging.warning(f"Timing violation detected: {critical_delay:.3f}ns")
                self._adapt_parameters_for_violation()
            
            # Stability analysis
            if delay_variance > 0.1:
                logging.info("High delay variance detected, adjusting parameters")
                self._adapt_parameters_for_stability()

    def _analyze_slack_violations(self, slack_info: Dict[str, Any]):
        """Analyze slack violations for DRC iteration reduction."""
        if 'negative_slack' in slack_info:
            negative_slack_count = len(slack_info['negative_slack'])
            if negative_slack_count > 0:
                logging.warning(f"Found {negative_slack_count} slack violations")
                
                # Track for DRC iteration optimization
                self._update_drc_iteration_metrics(negative_slack_count)

    def _adapt_parameters_for_violation(self):
        """Adapt parameters when timing violations are detected."""
        # Increase adaptation rate for violations
        self.adaptation_rate *= 1.2
        self.adaptation_rate = min(self.adaptation_rate, 0.2)
        
        logging.info(f"Adapted parameters for violation: rate={self.adaptation_rate:.3f}")

    def _adapt_parameters_for_stability(self):
        """Adapt parameters for stability improvement."""
        # Decrease adaptation rate for stability
        self.adaptation_rate *= 0.9
        self.adaptation_rate = max(self.adaptation_rate, 0.01)
        
        logging.info(f"Adapted parameters for stability: rate={self.adaptation_rate:.3f}")

    def _update_drc_iteration_metrics(self, violation_count: int):
        """Update metrics for DRC iteration optimization."""
        # This would track DRC iteration patterns
        # For now, just log the information
        logging.info(f"DRC iteration metrics: {violation_count} violations")

    def _update_metrics(self, runtime: float, success: bool):
        """Update performance metrics."""
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        
        # Update average runtime
        if self.total_runs == 1:
            self.average_runtime = runtime
        else:
            self.average_runtime = (self.average_runtime * (self.total_runs - 1) + runtime) / self.total_runs

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        success_rate = self.successful_runs / self.total_runs if self.total_runs > 0 else 0.0
        
        return {
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'success_rate': success_rate,
            'average_runtime': self.average_runtime,
            'critical_path_delays': self.critical_path_delays[-10:] if self.critical_path_delays else [],
            'adaptation_rate': self.adaptation_rate
        }

    def create_design_file(self, placement: Dict[str, np.ndarray], 
                          output_dir: Path) -> Path:
        """Create design file from placement for STA."""
        # Create DEF file with placement
        def_content = self._generate_def_content(placement)
        
        def_file = output_dir / "design.def"
        def_file.write_text(def_content)
        
        # Create library file
        lib_content = self._generate_lib_content()
        lib_file = output_dir / "library.lib"
        lib_file.write_text(lib_content)
        
        # Create constraints file
        sdc_content = self._generate_sdc_content()
        sdc_file = output_dir / "constraints.sdc"
        sdc_file.write_text(sdc_content)
        
        return output_dir

    def _generate_def_content(self, placement: Dict[str, np.ndarray]) -> str:
        """Generate DEF file content from placement."""
        x = placement['x']
        y = placement['y']
        width = placement.get('width', np.ones(len(x)) * 10)
        height = placement.get('height', np.ones(len(x)) * 10)
        
        def_lines = [
            "VERSION 5.8 ;",
            "DIVIDERCHAR \"/\" ;",
            "BUSBITCHARS \"[]\" ;",
            "DESIGN design ;",
            "UNITS DISTANCE MICRONS 1000 ;",
            "",
            "DIEAREA ( 0 0 ) ( 10000 10000 ) ;",
            ""
        ]
        
        # Add cell instances
        for i in range(len(x)):
            cell_name = f"cell_{i}"
            def_lines.append(f"   - {cell_name}")
            def_lines.append(f"     + PLACED ( {int(x[i])} {int(y[i])} ) N ;")
        
        def_lines.extend([
            "",
            "END COMPONENTS",
            "",
            "END DESIGN"
        ])
        
        return "\n".join(def_lines)

    def _generate_lib_content(self) -> str:
        """Generate library file content."""
        return """
library (library_name) {
    cell (AND2) {
        area : 10.0;
        pin (A) { direction : input; }
        pin (B) { direction : input; }
        pin (Y) { direction : output; }
    }
    cell (OR2) {
        area : 12.0;
        pin (A) { direction : input; }
        pin (B) { direction : input; }
        pin (Y) { direction : output; }
    }
}
"""

    def _generate_sdc_content(self) -> str:
        """Generate SDC constraints file content."""
        return """
# Clock constraints
create_clock -name clk -period 10.0 [get_ports clk]

# Input delays
set_input_delay -clock clk 1.0 [all_inputs]

# Output delays
set_output_delay -clock clk 1.0 [all_outputs]

# Load constraints
set_load 0.1 [all_outputs]
"""


class AsyncSTARunner(STARunner):
    """Asynchronous STA runner with enhanced batching."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semaphore = asyncio.Semaphore(4)  # Limit concurrent STA runs
        
    async def run_concurrent_batch(self, design_dirs: List[Path]) -> List[Path]:
        """Run STA on multiple designs concurrently."""
        async def run_single(design_dir: Path) -> Optional[Path]:
            async with self.semaphore:
                try:
                    return await self.run_async(design_dir)
                except Exception as e:
                    logging.error(f"Concurrent STA failed for {design_dir}: {e}")
                    return None
        
        # Run all designs concurrently
        tasks = [run_single(design_dir) for design_dir in design_dirs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        return valid_results
