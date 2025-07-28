# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import scipy.io as sio
import numpy as np
import zipfile
import tempfile
import shutil

from tqdm import tqdm
from typing import List
from pathlib import Path

from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor
from batteryml import BatteryData, CycleData, CyclingProtocol


@PREPROCESSORS.register()
class NASAPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        # Look for NASA zip file
        raw_file = Path(parentdir) / 'NASA_Battery_Dataset.zip'
        
        if not raw_file.exists():
            raise FileNotFoundError(f'NASA zip file not found: {raw_file}')
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract main zip file
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                if not self.silent:
                    print(f'Extracting {raw_file.name}...')
                zip_ref.extractall(temp_path)
            
            # Find the battery data directory (should be "5. Battery Data Set")
            battery_dir = temp_path / "5. Battery Data Set"
            if not battery_dir.exists():
                # Fallback: search for any directory with .zip files
                for item in temp_path.rglob('*.zip'):
                    battery_dir = item.parent
                    break
            
            if not battery_dir.exists():
                raise FileNotFoundError('Battery data directory not found in extracted zip')
            
            # Extract all nested batch zip files
            batch_zips = list(battery_dir.glob('*.zip'))
            if not self.silent and batch_zips:
                batch_zips = tqdm(batch_zips, desc='Extracting batch files')
            
            for nested_zip in batch_zips:
                if hasattr(batch_zips, 'set_description'):
                    batch_zips.set_description(f'Extracting {nested_zip.name}')
                
                with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
                    zip_ref.extractall(battery_dir)
            
            # Find all NASA .mat files (including those from nested zips)
            raw_files = list(battery_dir.glob('B*.mat'))
            
            if not raw_files:
                raise FileNotFoundError(f'No NASA .mat files found in {battery_dir}')
            
            raw_files = sorted(raw_files, key=lambda x: int(x.stem[1:]))  # Sort by battery number
            data_batteries = []
            
            if not self.silent:
                raw_files = tqdm(raw_files, desc='Processing NASA batteries')

            for f in raw_files:
                if hasattr(raw_files, 'set_description'):
                    raw_files.set_description(f'Loading {f.stem}')

                try:
                    battery = load_nasa_battery(f)
                    if battery is not None:
                        data_batteries.append(battery)
                        self.dump_single_file(battery)
                        if not self.silent:
                            tqdm.write(f'File: {battery.cell_id} dumped to pkl file')
                except Exception as e:
                    if not self.silent:
                        tqdm.write(f'Error processing {f.stem}: {e}')
                    continue

        return len(data_batteries), 0


def load_nasa_battery(file_path: Path) -> BatteryData:
    """Load a single NASA battery .mat file"""
    data = sio.loadmat(str(file_path))
    
    # Get battery name from filename (e.g., B0005)
    battery_name = file_path.stem
    
    # Load battery data
    battery_data = data[battery_name]
    cycles = battery_data[0, 0]['cycle'][0]  # Shape is (1, 616) so we need [0] to get (616,)
    
    cycle_data = []
    cycle_number = 0
    
    for i, cycle in enumerate(cycles):
        # Get cycle type - handle different possible shapes
        cycle_type_raw = cycle['type']
        if cycle_type_raw.size > 0:
            cycle_type = cycle_type_raw.flat[0] if hasattr(cycle_type_raw, 'flat') else cycle_type_raw[0]
            if isinstance(cycle_type, bytes):
                cycle_type = cycle_type.decode('utf-8')
            elif isinstance(cycle_type, np.ndarray):
                cycle_type = str(cycle_type.flat[0])
            else:
                cycle_type = str(cycle_type)
        else:
            cycle_type = ''
        
        # Only process discharge cycles for capacity estimation
        if cycle_type == 'discharge':
            cycle_number += 1
            
            # Extract cycle data
            cycle_measurements = cycle['data'].flat[0] if hasattr(cycle['data'], 'flat') else cycle['data'][0, 0]
            
            # Get measurement arrays - handle different possible shapes
            voltage = cycle_measurements['Voltage_measured']
            if voltage.ndim > 1:
                voltage = voltage.flatten()
            
            current = cycle_measurements['Current_measured'] 
            if current.ndim > 1:
                current = current.flatten()
                
            temperature = cycle_measurements['Temperature_measured']
            if temperature.ndim > 1:
                temperature = temperature.flatten()
                
            time = cycle_measurements['Time']
            if time.ndim > 1:
                time = time.flatten()
            
            # Get capacity from NASA data (total capacity for this discharge cycle)
            total_capacity = cycle_measurements['Capacity']
            if total_capacity.ndim > 0:
                total_capacity_value = float(total_capacity.flat[0])
            else:
                total_capacity_value = float(total_capacity)
            
            # Calculate cumulative capacity during discharge
            # Assume linear capacity decrease during discharge
            if len(time) > 1:
                capacity = np.linspace(total_capacity_value, 0, len(time))
            else:
                capacity = [total_capacity_value]
            
            cycle_data.append(CycleData(
                cycle_number=cycle_number,
                voltage_in_V=voltage.tolist(),
                current_in_A=current.tolist(),
                temperature_in_C=temperature.tolist(),
                discharge_capacity_in_Ah=capacity.tolist(),
                charge_capacity_in_Ah=[0.0] * len(voltage),  # Not available in NASA data
                time_in_s=time.tolist(),
                internal_resistance_in_ohm=0.0  # Not directly available
            ))
    
    # Define discharge end voltage based on battery number
    # From NASA documentation: 2.0V, 2.2V, 2.5V, 2.7V for different battery groups
    battery_num = int(battery_name[1:])  # Extract number from B0005 -> 5
    
    if battery_num in [25, 26, 27, 28]:
        end_voltage = {25: 2.0, 26: 2.2, 27: 2.5, 28: 2.7}[battery_num]
    elif battery_num in [53, 54, 55, 56]:
        end_voltage = {53: 2.0, 54: 2.2, 55: 2.5, 56: 2.7}[battery_num]
    else:
        end_voltage = 2.0  # Default for other batteries
    
    # Define protocols based on NASA documentation
    # Discharge protocols vary by battery and temperature
    if battery_num in [53, 54, 55, 56]:
        # Cold temperature (4°C), 2A discharge
        discharge_protocol = CyclingProtocol(
            rate_in_C=1.0,  # 2A for 2Ah = 1C
            start_soc=1.0,
            end_voltage_in_V=end_voltage
        )
    else:
        # Room temperature (24°C), square wave discharge
        discharge_protocol = CyclingProtocol(
            rate_in_C=2.0,  # 4A amplitude = 2C
            start_soc=1.0,
            end_voltage_in_V=end_voltage
        )
    
    # Charge: CC-CV, 1.5A until 4.2V, then CV until 20mA
    charge_protocol = [CyclingProtocol(
        rate_in_C=0.75,  # 1.5A for 2Ah = 0.75C
        start_soc=0.0,
        end_soc=1.0
    )]
    
    return BatteryData(
        cell_id=f'NASA_{battery_name}',
        cycle_data=cycle_data,
        form_factor='cylindrical_18650',
        anode_material='graphite',
        cathode_material='LiCoO2',  # Based on NASA documentation
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=2.0,  # Typical 18650 capacity
        min_voltage_limit_in_V=end_voltage,
        max_voltage_limit_in_V=4.2   # From NASA documentation
    )