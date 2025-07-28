# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import pandas as pd

from tqdm import tqdm
from numba import njit
from typing import List
from pathlib import Path

from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor


@PREPROCESSORS.register()
class UCLPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        raw_file = Path(parentdir) / 'UCL_Battery_Dataset.csv'
        
        if not raw_file.exists():
            raise FileNotFoundError(f'UCL dataset file not found: {raw_file}')
        
        # UCL dataset contains only one cell, so process it as a single battery
        cell_name = 'UCL_EIL-MJ1-015'
        
        # Check if file should be skipped
        whether_to_skip = self.check_processed_file(cell_name)
        if whether_to_skip:
            return 0, 1
        
        process_batteries_num = 0
        skip_batteries_num = 0
        
        if not self.silent:
            print(f'Loading {raw_file.name}...')
        
        try:
            battery = load_ucl_battery(raw_file, cell_name)
            if battery and len(battery.cycle_data) > 0:
                self.dump_single_file(battery)
                process_batteries_num = 1
                
                if not self.silent:
                    print(f'File: {battery.cell_id} dumped to pkl file')
        except Exception as e:
            if not self.silent:
                print(f'Error processing {cell_name}: {e}')
            skip_batteries_num = 1
        
        return process_batteries_num, skip_batteries_num


def load_ucl_battery(file_path: Path, cell_name: str) -> BatteryData:
    """Load UCL battery data from CSV file"""
    
    # Load the CSV file
    df = pd.read_csv(file_path, low_memory=False)
    
    return create_ucl_battery_data(df, cell_name)


def create_ucl_battery_data(df: pd.DataFrame, cell_name: str) -> BatteryData:
    """Create BatteryData from UCL dataset"""
    
    # Column block mapping based on the notebook structure
    block_map = {
        "": {
            "test_time": "Test Time",
            "cycle": "Cycle Number", 
            "voltage": "Cell Potential",
            "temp": "Temp",
            "capacity": "Capacity",
        },
        ".1": {
            "test_time": "Test Time.1",
            "cycle": "Cycle Number.1",
            "voltage": "Cell Potential.1", 
            "temp": "Temp.1",
            "capacity": "Capacity.1",
        },
        ".2": {
            "test_time": "Test Time.2",
            "cycle": "Cycle Number.2",
            "voltage": "Cell Potential.2",
            "temp": "Temp.2", 
            "capacity": "Capacity.2",
        },
        ".3": {
            "test_time": "Test Time.3",
            "cycle": "Cycle Number.3",
            "voltage": "Cell Potential.3",
            "temp": "Temp.3",
            "capacity": "Capacity.3", 
        },
        ".4": {
            "test_time": "Test Time.4",
            "cycle": "Cycle Number.4",
            "voltage": "Cell Potential.4",
            "temp": "Temp.4",
            "capacity": "Capacity.4",
        },
    }
    
    # Remove header row (contains units like "Hrs", "Deg C", etc.)
    df_clean = df[df["Cycle"] != "-"].copy()
    
    # Extract cycle summary data from first 4 columns
    cycle_summary = df_clean[["Cycle", "Charge Capacity", "Discharge Capacity", "Discharge/Charge"]].copy()
    for col in cycle_summary.columns:
        cycle_summary[col] = pd.to_numeric(cycle_summary[col], errors='coerce')
    cycle_summary = cycle_summary.dropna()
    
    # Collect all cycle data from different column blocks
    all_cycle_data = {}
    
    for suffix, cols in block_map.items():
        cycle_col = cols["cycle"]
        test_time_col = cols["test_time"] 
        voltage_col = cols["voltage"]
        temp_col = cols["temp"]
        capacity_col = cols["capacity"]
        
        # Skip if columns don't exist
        if not all(col in df.columns for col in [cycle_col, test_time_col, voltage_col, temp_col, capacity_col]):
            continue
            
        # Get data for this block
        block_df = df[[test_time_col, cycle_col, temp_col, capacity_col, voltage_col]].copy()
        
        # Convert to numeric, skip header row
        for col in block_df.columns:
            block_df[col] = pd.to_numeric(block_df[col], errors='coerce')
        
        # Remove NaN rows
        block_df = block_df.dropna(subset=[cycle_col, voltage_col])
        
        # Group by cycle number
        for cycle_num, cycle_df in block_df.groupby(cycle_col):
            cycle_num = int(cycle_num)
            if cycle_num not in all_cycle_data:
                all_cycle_data[cycle_num] = {
                    'test_time': [],
                    'voltage': [], 
                    'temp': [],
                    'capacity': []
                }
            
            # Sort by test time
            cycle_df = cycle_df.sort_values(test_time_col)
            
            all_cycle_data[cycle_num]['test_time'].extend(cycle_df[test_time_col].tolist())
            all_cycle_data[cycle_num]['voltage'].extend(cycle_df[voltage_col].tolist())
            all_cycle_data[cycle_num]['temp'].extend(cycle_df[temp_col].tolist())
            all_cycle_data[cycle_num]['capacity'].extend(cycle_df[capacity_col].tolist())
    
    # Convert to CycleData objects
    cycles = []
    for cycle_num in sorted(all_cycle_data.keys()):
        cycle_data = all_cycle_data[cycle_num]
        
        if len(cycle_data['voltage']) < 2:
            continue
            
        # Skip cycles where all capacity values are essentially zero or the same
        capacity_array = np.array(cycle_data['capacity'])
        if len(capacity_array) == 0 or (np.max(capacity_array) - np.min(capacity_array)) < 0.001:
            continue
            
        # Convert time from hours to seconds
        time_in_s = np.array(cycle_data['test_time']) * 3600
        voltage_in_V = np.array(cycle_data['voltage'])
        temperature_in_C = np.array(cycle_data['temp'])
        capacity_in_Ah = np.array(cycle_data['capacity'])
        
        # Calculate current from capacity change (dQ/dt) - more carefully
        current_in_A = np.zeros_like(capacity_in_Ah)
        for i in range(1, len(capacity_in_Ah)):
            dt = time_in_s[i] - time_in_s[i-1] 
            if dt > 0.001:  # Avoid very small time differences
                dQ = capacity_in_Ah[i] - capacity_in_Ah[i-1]
                current_calc = dQ * 3600 / dt  # Convert Ah/s to A
                # Clamp unrealistic current values
                current_in_A[i] = np.clip(current_calc, -10.0, 10.0)
        
        # For UCL data, capacity appears to be cumulative discharge capacity
        # Use the actual capacity values more intelligently
        # Determine charge vs discharge based on capacity trend
        capacity_diff = np.diff(capacity_in_Ah, prepend=capacity_in_Ah[0])
        
        # Initialize arrays
        charge_capacity_in_Ah = np.zeros_like(capacity_in_Ah)
        discharge_capacity_in_Ah = np.zeros_like(capacity_in_Ah)
        
        # Build cumulative capacities based on whether capacity is increasing (charging) or decreasing (discharging)
        cumulative_charge = 0.0
        cumulative_discharge = 0.0
        
        for i in range(len(capacity_in_Ah)):
            if i > 0:
                if capacity_diff[i] > 0:  # Capacity increasing = charging
                    cumulative_charge += abs(capacity_diff[i])
                elif capacity_diff[i] < 0:  # Capacity decreasing = discharging  
                    cumulative_discharge += abs(capacity_diff[i])
            
            charge_capacity_in_Ah[i] = cumulative_charge
            discharge_capacity_in_Ah[i] = cumulative_discharge
        
        cycle = CycleData(
            cycle_number=cycle_num,
            voltage_in_V=voltage_in_V.tolist(),
            current_in_A=current_in_A.tolist(), 
            time_in_s=time_in_s.tolist(),
            charge_capacity_in_Ah=charge_capacity_in_Ah.tolist(),
            discharge_capacity_in_Ah=discharge_capacity_in_Ah.tolist(),
            temperature_in_C=temperature_in_C.tolist()
        )
        
        cycles.append(cycle)
    
    if not cycles:
        return None
        
    # Use the specified nominal capacity for UCL dataset
    nominal_capacity = 3.5  # UCL dataset specifies 3.5Ah capacity
    
    # Create battery data with actual UCL test protocol
    battery = BatteryData(
        cell_id=cell_name,
        cycle_data=cycles,
        form_factor='cylindrical_18650',  # 18650 form factor
        anode_material='graphite',  # NCA/graphite chemistry
        cathode_material='NCA',  # Nickel Cobalt Aluminum oxide
        nominal_capacity_in_Ah=nominal_capacity,
        charge_protocol=[
            # CC phase: 1.5A CC to 4.2V
            CyclingProtocol(
                current_in_A=1.5,
                rate_in_C=1.5 / nominal_capacity,  # ~0.43C charge rate
                voltage_in_V=4.2,
                start_soc=0.0,
                end_soc=0.8  # Approximate end of CC phase
            ),
            # CV phase: CV at 4.2V until cutoff 100mA
            CyclingProtocol(
                voltage_in_V=4.2,
                current_in_A=0.1,  # 100mA cutoff
                start_soc=0.8,
                end_soc=1.0
            )
        ],
        discharge_protocol=CyclingProtocol(
            current_in_A=4.0,  # 4.0A discharge current
            rate_in_C=4.0 / nominal_capacity,  # ~1.14C discharge rate
            start_soc=1.0,
            end_voltage_in_V=2.5
        ),
        max_voltage_limit_in_V=4.2,
        min_voltage_limit_in_V=2.5,
        max_current_limit_in_A=4.0,  # Maximum discharge current: 4.0A
        min_current_limit_in_A=-1.5,  # Maximum charge current: 1.5A (negative for charge)
        reference='UCL Battery Dataset',
        description='18650 3.5Ah NCA/graphite cell tested on Maccor 4200 at 24°C, 400 cycles, '
                   'Charge: 1.5A CC to 4.2V + CV (cutoff 100mA), Discharge: 4.0A to 2.5V'
    )
    
    return battery