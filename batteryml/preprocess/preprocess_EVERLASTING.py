# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import zipfile
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
class EVERLASTINGPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        raw_file = Path(parentdir) / 'EVERLASTING_Battery_Dataset.zip'
        
        if not raw_file.exists():
            raise FileNotFoundError(f'EVERLASTING zip file not found: {raw_file}')
        
        # Extract the zip file to temporary directory
        extract_dir = raw_file.parent / 'EVERLASTING_extracted'
        if not extract_dir.exists():
            if not self.silent:
                print(f'Extracting {raw_file.name}...')
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        
        # Find all CSV files
        csv_files = list(extract_dir.glob('*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f'No CSV files found in {extract_dir}')
        
        csv_files = sorted(csv_files)
        
        if not self.silent:
            csv_files = tqdm(csv_files, desc='Processing EVERLASTING batteries')
        
        process_batteries_num = 0
        skip_batteries_num = 0
        
        for csv_file in csv_files:
            # Parse cell information from filename
            cell_info = parse_everlasting_filename(csv_file.stem)
            cell_name = f'EVERLASTING_{csv_file.stem}'
            
            if hasattr(csv_files, 'set_description'):
                csv_files.set_description(f'Processing {cell_name}')
            
            # Check if file should be skipped
            whether_to_skip = self.check_processed_file(cell_name)
            if whether_to_skip:
                skip_batteries_num += 1
                continue
            
            try:
                battery = load_everlasting_battery(csv_file, cell_name, cell_info)
                if battery and len(battery.cycle_data) > 0:
                    self.dump_single_file(battery)
                    process_batteries_num += 1
                    
                    if not self.silent:
                        tqdm.write(f'File: {battery.cell_id} dumped to pkl file')
            except Exception as e:
                if not self.silent:
                    tqdm.write(f'Error processing {cell_name}: {e}')
                continue
        
        # Clean up extracted files
        import shutil
        shutil.rmtree(extract_dir)
        
        return process_batteries_num, skip_batteries_num


def parse_everlasting_filename(filename: str) -> dict:
    """Parse EVERLASTING filename to extract test conditions"""
    # Example: Cycl_T0_SOC10-90_Dch1.5C_Ch0.5C_TUV_Cell25_01
    # Or: DrivingAgeing_T0_SOC10-90_TUV_Cell116_01
    
    parts = filename.split('_')
    info = {
        'test_type': parts[0],  # Cycl or DrivingAgeing
        'temperature': int(parts[1][1:]),  # T0 -> 0°C
        'soc_range': parts[2][3:],  # SOC10-90 -> 10-90
        'cell_id': f"{parts[-2]}_{parts[-1]}",  # Cell25_01
        'discharge_rate': None,
        'charge_rate': None
    }
    
    # Extract charge/discharge rates for cycling tests
    if info['test_type'] == 'Cycl':
        for part in parts:
            if part.startswith('Dch'):
                info['discharge_rate'] = float(part[3:-1])  # Dch1.5C -> 1.5
            elif part.startswith('Ch'):
                info['charge_rate'] = float(part[2:-1])  # Ch0.5C -> 0.5
    
    return info


def load_everlasting_battery(csv_file: Path, cell_name: str, cell_info: dict) -> BatteryData:
    """Load EVERLASTING battery data from CSV file"""
    
    # Load CSV with proper delimiter (semicolon)
    df = pd.read_csv(csv_file, sep=';', low_memory=False)
    
    # Different column mapping for different test types
    if cell_info['test_type'] == 'Cycl':
        # Cycling test columns
        column_mapping = {
            'Cyc#': 'cycle_number',
            'Test (Hr)': 'time_in_hr',
            'Amp-hr': 'capacity_in_Ah',
            'Amps': 'current_in_A',
            'Volts': 'voltage_in_V',
            'EV Temp C': 'temperature_in_C'
        }
        df = df.rename(columns=column_mapping)
        df['time_in_s'] = df['time_in_hr'] * 3600
        
    elif cell_info['test_type'] == 'DrivingAgeing':
        # DrivingAgeing test columns
        column_mapping = {
            'Time, s': 'time_in_s',
            'I, A': 'current_in_A', 
            'U, V': 'voltage_in_V',
            'KAPA_CC, Ah': 'capacity_in_Ah',
            'Temp[1], C': 'temperature_in_C'
        }
        df = df.rename(columns=column_mapping)
        
        # For DrivingAgeing, create artificial cycle numbers based on time segments
        # Group data into cycles based on capacity resets or time intervals
        df['cycle_number'] = create_driving_cycles(df)
    else:
        raise ValueError(f"Unknown test type: {cell_info['test_type']}")
    
    
    # Group by cycle number
    cycles = []
    for cycle_num, cycle_df in df.groupby('cycle_number'):
        cycle_df = cycle_df.sort_values('time_in_s')
        
        # Skip cycles with insufficient data
        if len(cycle_df) < 2:
            continue
        
        # Extract basic measurements
        time_in_s = cycle_df['time_in_s'].values
        voltage_in_V = cycle_df['voltage_in_V'].values
        current_in_A = cycle_df['current_in_A'].values
        capacity_in_Ah = cycle_df['capacity_in_Ah'].values
        
        # Extract temperature if available
        temperature_in_C = None
        if 'temperature_in_C' in cycle_df.columns:
            temperature_in_C = cycle_df['temperature_in_C'].values
        
        # Calculate charge and discharge capacities
        charge_capacity_in_Ah, discharge_capacity_in_Ah = calculate_capacities(
            current_in_A, time_in_s
        )
        
        cycle_data = CycleData(
            cycle_number=int(cycle_num),
            voltage_in_V=voltage_in_V.tolist(),
            current_in_A=current_in_A.tolist(),
            time_in_s=time_in_s.tolist(),
            charge_capacity_in_Ah=charge_capacity_in_Ah.tolist(),
            discharge_capacity_in_Ah=discharge_capacity_in_Ah.tolist(),
            temperature_in_C=temperature_in_C.tolist() if temperature_in_C is not None else None
        )
        
        cycles.append(cycle_data)
    
    if not cycles:
        return None
    
    # Sort cycles by cycle number
    cycles.sort(key=lambda x: x.cycle_number)
    
    # Estimate nominal capacity from first cycle
    first_cycle_discharge = max([max(cycle.discharge_capacity_in_Ah) for cycle in cycles[:3] if cycle.discharge_capacity_in_Ah])
    nominal_capacity = max(first_cycle_discharge, 2.5)  # Assume at least 2.5Ah
    
    # Create charge and discharge protocols based on cell info
    charge_protocol, discharge_protocol = create_everlasting_protocols(cell_info, nominal_capacity)
    
    # Create battery data
    battery = BatteryData(
        cell_id=cell_name,
        cycle_data=cycles,
        form_factor='cylindrical_18650',  # Assumed based on typical automotive cells
        anode_material='graphite',
        cathode_material='NMC',  # Common for automotive applications
        nominal_capacity_in_Ah=nominal_capacity,
        charge_protocol=charge_protocol,
        discharge_protocol=discharge_protocol,
        max_voltage_limit_in_V=4.2,
        min_voltage_limit_in_V=2.5,
        reference='EVERLASTING Battery Dataset',
        description=f'EVERLASTING {cell_info["test_type"]} test at {cell_info["temperature"]}°C, '
                   f'SOC range: {cell_info["soc_range"]}%, Cell: {cell_info["cell_id"]}'
    )
    
    return battery


def create_everlasting_protocols(cell_info: dict, nominal_capacity: float):
    """Create charge and discharge protocols based on cell info"""
    
    if cell_info['test_type'] == 'Cycl' and cell_info['charge_rate'] and cell_info['discharge_rate']:
        # For cycling tests with specified rates
        charge_protocol = [
            CyclingProtocol(
                current_in_A=cell_info['charge_rate'] * nominal_capacity,
                rate_in_C=cell_info['charge_rate'],
                voltage_in_V=4.2,
                start_soc=0.0,
                end_soc=1.0
            )
        ]
        
        discharge_protocol = CyclingProtocol(
            current_in_A=cell_info['discharge_rate'] * nominal_capacity,
            rate_in_C=cell_info['discharge_rate'],
            start_soc=1.0,
            end_voltage_in_V=2.5
        )
    else:
        # For driving aging tests or unknown protocols
        charge_protocol = [
            CyclingProtocol(
                current_in_A=nominal_capacity * 0.5,  # 0.5C default
                rate_in_C=0.5,
                voltage_in_V=4.2,
                start_soc=0.0,
                end_soc=1.0
            )
        ]
        
        discharge_protocol = CyclingProtocol(
            current_in_A=nominal_capacity * 1.0,  # 1C default
            rate_in_C=1.0,
            start_soc=1.0,
            end_voltage_in_V=2.5
        )
    
    return charge_protocol, discharge_protocol


@njit
def calculate_capacities(current, time):
    """Calculate charge and discharge capacities from current and time"""
    charge_capacity = np.zeros_like(current)
    discharge_capacity = np.zeros_like(current)
    
    cumulative_charge = 0.0
    cumulative_discharge = 0.0
    
    for i in range(1, len(current)):
        dt = time[i] - time[i-1]
        if dt > 0:
            dQ = current[i] * dt / 3600  # Convert to Ah
            
            if current[i] > 0:  # Charging (positive current)
                cumulative_charge += dQ
            elif current[i] < 0:  # Discharging (negative current)
                cumulative_discharge += abs(dQ)
        
        charge_capacity[i] = cumulative_charge
        discharge_capacity[i] = cumulative_discharge
    
    return charge_capacity, discharge_capacity


def create_driving_cycles(df: pd.DataFrame) -> np.ndarray:
    """Create artificial cycle numbers for DrivingAgeing data based on time segments"""
    # For driving data, we'll create cycles based on time intervals
    # Each "cycle" represents a certain period (e.g., 1 hour or when capacity resets)
    
    time_data = df['time_in_s'].values
    capacity_data = df.get('capacity_in_Ah', pd.Series([0] * len(df))).values
    
    # Method 1: Time-based cycles (e.g., every hour = 3600 seconds)
    cycle_duration = 3600  # 1 hour per cycle
    cycles = (time_data / cycle_duration).astype(int)
    
    # Method 2: Also detect capacity resets (when capacity suddenly drops)
    if len(capacity_data) > 1:
        capacity_resets = np.where(np.diff(capacity_data) < -0.1)[0] + 1  # Capacity drops by >0.1Ah
        
        # Add cycle breaks at capacity resets
        for reset_idx in capacity_resets:
            cycles[reset_idx:] += 1
    
    return cycles