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


def get_nasa_battery_metadata(battery_num: int) -> dict:
    """Get hardcoded metadata for NASA batteries based on README files"""
    
    # Base metadata for all NASA batteries
    base_metadata = {
        'form_factor': 'cylindrical_18650',
        'anode_material': 'graphite', 
        'cathode_material': 'LiCoO2',
        'nominal_capacity_in_Ah': 2.0,
        'max_voltage_limit_in_V': 4.2,
        'reference': 'NASA Ames Prognostics Data Repository',
        'description': 'Li-ion battery aging dataset with charge/discharge/impedance cycles'
    }
    
    # Charge protocol (same for all batteries): CC-CV, 1.5A until 4.2V, then CV until 20mA  
    charge_protocol = CyclingProtocol(
        current_in_A=1.5,
        rate_in_C=0.75,  # 1.5A for 2Ah = 0.75C
        voltage_in_V=4.2,
        start_soc=0.0,
        end_soc=1.0
    )
    
    # Battery-specific metadata based on README files
    if battery_num in [5, 6, 7, 18]:
        # Room temperature, different discharge end voltages
        end_voltages = {5: 2.7, 6: 2.5, 7: 2.2, 18: 2.5}
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages[battery_num],
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,  # 2A for 2Ah = 1C
                start_soc=1.0,
                end_voltage_in_V=end_voltages[battery_num]
            ),
            'description': f'NASA battery #{battery_num}, room temperature, discharge to {end_voltages[battery_num]}V'
        }
        
    elif battery_num in [25, 26, 27, 28]:
        # Room temperature (24°C), 0.05Hz square wave discharge 4A amplitude, 50% duty cycle
        end_voltages = {25: 2.0, 26: 2.2, 27: 2.5, 28: 2.7}
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages[battery_num],
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=4.0,  # 4A amplitude
                rate_in_C=2.0,  # 4A for 2Ah = 2C
                start_soc=1.0,
                end_voltage_in_V=end_voltages[battery_num]
            ),
            'description': f'NASA battery #{battery_num}, 24°C, 0.05Hz square wave discharge to {end_voltages[battery_num]}V'
        }
        
    elif battery_num in [29, 30, 31, 32]:
        # Elevated temperature (43°C), 4A discharge
        end_voltages = {29: 2.0, 30: 2.2, 31: 2.5, 32: 2.7}
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages[battery_num],
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=4.0,
                rate_in_C=2.0,  # 4A for 2Ah = 2C
                start_soc=1.0,
                end_voltage_in_V=end_voltages[battery_num]
            ),
            'description': f'NASA battery #{battery_num}, 43°C elevated temperature, 4A discharge to {end_voltages[battery_num]}V'
        }
        
    elif battery_num in [33, 34, 36]:
        # Need to read the actual README for these
        end_voltages = {33: 2.0, 34: 2.2, 36: 2.5}  # Estimated based on pattern
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages.get(battery_num, 2.0),
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,
                start_soc=1.0,
                end_voltage_in_V=end_voltages.get(battery_num, 2.0)
            ),
            'description': f'NASA battery #{battery_num}'
        }
        
    elif battery_num in [38, 39, 40]:
        # Need to read the actual README for these
        end_voltages = {38: 2.0, 39: 2.2, 40: 2.5}  # Estimated based on pattern
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages.get(battery_num, 2.0),
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,
                start_soc=1.0,
                end_voltage_in_V=end_voltages.get(battery_num, 2.0)
            ),
            'description': f'NASA battery #{battery_num}'
        }
        
    elif battery_num in [41, 42, 43, 44]:
        # Need to read the actual README for these  
        end_voltages = {41: 2.0, 42: 2.2, 43: 2.5, 44: 2.7}  # Estimated based on pattern
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages.get(battery_num, 2.0),
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,
                start_soc=1.0,
                end_voltage_in_V=end_voltages.get(battery_num, 2.0)
            ),
            'description': f'NASA battery #{battery_num}'
        }
        
    elif battery_num in [45, 46, 47, 48]:
        # Need to read the actual README for these
        end_voltages = {45: 2.0, 46: 2.2, 47: 2.5, 48: 2.7}  # Estimated based on pattern
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages.get(battery_num, 2.0),
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,
                start_soc=1.0,
                end_voltage_in_V=end_voltages.get(battery_num, 2.0)
            ),
            'description': f'NASA battery #{battery_num}'
        }
        
    elif battery_num in [49, 50, 51, 52]:
        # Cold temperature (4°C), 2A discharge
        end_voltages = {49: 2.0, 50: 2.2, 51: 2.5, 52: 2.7}
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages[battery_num],
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,  # 2A for 2Ah = 1C
                start_soc=1.0,
                end_voltage_in_V=end_voltages[battery_num]
            ),
            'description': f'NASA battery #{battery_num}, 4°C cold temperature, 2A discharge to {end_voltages[battery_num]}V'
        }
        
    elif battery_num in [53, 54, 55, 56]:
        # Cold temperature (4°C), 2A discharge
        end_voltages = {53: 2.0, 54: 2.2, 55: 2.5, 56: 2.7}
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': end_voltages[battery_num],
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,  # 2A for 2Ah = 1C
                start_soc=1.0,
                end_voltage_in_V=end_voltages[battery_num]
            ),
            'description': f'NASA battery #{battery_num}, 4°C cold temperature, 2A discharge to {end_voltages[battery_num]}V'
        }
        
    else:
        # Default for unknown batteries
        metadata = {
            **base_metadata,
            'min_voltage_limit_in_V': 2.0,
            'charge_protocol': [charge_protocol],
            'discharge_protocol': CyclingProtocol(
                current_in_A=2.0,
                rate_in_C=1.0,
                start_soc=1.0,
                end_voltage_in_V=2.0
            ),
            'description': f'NASA battery #{battery_num}'
        }
    
    return metadata


def load_nasa_battery(file_path: Path) -> BatteryData:
    """Load a single NASA battery .mat file with proper charge/discharge/impedance handling"""
    data = sio.loadmat(str(file_path))
    
    # Get battery name from filename (e.g., B0005)
    battery_name = file_path.stem
    battery_num = int(battery_name[1:])  # Extract number from B0005 -> 5
    
    # Get metadata for this battery
    metadata = get_nasa_battery_metadata(battery_num)
    
    # Load battery data
    battery_data = data[battery_name]
    cycles = battery_data[0, 0]['cycle'][0]  # Shape is (1, 616) so we need [0] to get (616,)
    
    # Group cycles by charge-discharge-impedance triplets
    cycle_groups = group_nasa_cycles(cycles)
    
    cycle_data = []
    for cycle_num, cycle_group in enumerate(cycle_groups, 1):
        unified_cycle = create_unified_cycle_data(cycle_group, cycle_num)
        if unified_cycle:
            cycle_data.append(unified_cycle)
    
    return BatteryData(
        cell_id=f'NASA_{battery_name}',
        cycle_data=cycle_data,
        **metadata  # Unpack all metadata from README-based lookup
    )


def group_nasa_cycles(cycles):
    """Group NASA cycles into charge-discharge-impedance triplets"""
    cycle_groups = []
    current_group = {'charge': None, 'discharge': None, 'impedance': None}
    
    for i, cycle in enumerate(cycles):
        # Get cycle type
        cycle_type = get_cycle_type(cycle)
        
        if cycle_type in ['charge', 'discharge', 'impedance']:
            current_group[cycle_type] = cycle
            
            # If we have both charge and discharge, create a complete cycle
            # (impedance is optional)
            if current_group['charge'] and current_group['discharge']:
                cycle_groups.append(current_group.copy())
                current_group = {'charge': None, 'discharge': None, 'impedance': None}
    
    return cycle_groups


def get_cycle_type(cycle):
    """Extract cycle type from NASA cycle data"""
    cycle_type_raw = cycle['type']
    if cycle_type_raw.size > 0:
        cycle_type = cycle_type_raw.flat[0] if hasattr(cycle_type_raw, 'flat') else cycle_type_raw[0]
        if isinstance(cycle_type, bytes):
            cycle_type = cycle_type.decode('utf-8')
        elif isinstance(cycle_type, np.ndarray):
            cycle_type = str(cycle_type.flat[0])
        else:
            cycle_type = str(cycle_type)
        return cycle_type.lower()
    return ''


def create_unified_cycle_data(cycle_group, cycle_number):
    """Create unified CycleData from charge-discharge-impedance group"""
    charge_cycle = cycle_group.get('charge')
    discharge_cycle = cycle_group.get('discharge')
    impedance_cycle = cycle_group.get('impedance')
    
    if not discharge_cycle:
        return None  # Need at least discharge data
    
    # Extract data from both charge and discharge cycles
    all_voltage = []
    all_current = []
    all_temperature = []
    all_time = []
    all_charge_capacity = []
    all_discharge_capacity = []
    
    # Process charge cycle first (if available)
    if charge_cycle:
        try:
            charge_measurements = charge_cycle['data'].flat[0] if hasattr(charge_cycle['data'], 'flat') else charge_cycle['data'][0, 0]
            
            charge_voltage = charge_measurements['Voltage_measured']
            if charge_voltage.ndim > 1:
                charge_voltage = charge_voltage.flatten()
            
            charge_current = charge_measurements['Current_measured']
            if charge_current.ndim > 1:
                charge_current = charge_current.flatten()
            
            charge_temperature = charge_measurements['Temperature_measured']
            if charge_temperature.ndim > 1:
                charge_temperature = charge_temperature.flatten()
            
            charge_time = charge_measurements['Time']
            if charge_time.ndim > 1:
                charge_time = charge_time.flatten()
            
            # Calculate charge capacity using CALCE-style integration
            charge_capacity = calc_Q_nasa(charge_current, charge_time, is_charge=True)
            discharge_capacity_charge = np.zeros_like(charge_capacity)  # No discharge during charge
            
            # Add charge data
            all_voltage.extend(charge_voltage.tolist())
            all_current.extend(charge_current.tolist()) 
            all_temperature.extend(charge_temperature.tolist())
            all_time.extend(charge_time.tolist())
            all_charge_capacity.extend(charge_capacity.tolist())
            all_discharge_capacity.extend(discharge_capacity_charge.tolist())
            
        except Exception as e:
            print(f"Warning: Could not process charge cycle: {e}")
    
    # Process discharge cycle
    try:
        discharge_measurements = discharge_cycle['data'].flat[0] if hasattr(discharge_cycle['data'], 'flat') else discharge_cycle['data'][0, 0]
        
        discharge_voltage = discharge_measurements['Voltage_measured']
        if discharge_voltage.ndim > 1:
            discharge_voltage = discharge_voltage.flatten()
        
        discharge_current = discharge_measurements['Current_measured']
        if discharge_current.ndim > 1:
            discharge_current = discharge_current.flatten()
        
        discharge_temperature = discharge_measurements['Temperature_measured']
        if discharge_temperature.ndim > 1:
            discharge_temperature = discharge_temperature.flatten()
        
        discharge_time = discharge_measurements['Time']
        if discharge_time.ndim > 1:
            discharge_time = discharge_time.flatten()
        
        # Adjust discharge time to continue from charge time
        time_offset = all_time[-1] if all_time else 0
        discharge_time_adjusted = discharge_time + time_offset
        
        # Calculate discharge capacity using CALCE-style integration  
        discharge_capacity = calc_Q_nasa(discharge_current, discharge_time, is_charge=False)
        charge_capacity_discharge = np.zeros_like(discharge_capacity)  # No charge during discharge
        
        # Add discharge data
        all_voltage.extend(discharge_voltage.tolist())
        all_current.extend(discharge_current.tolist())
        all_temperature.extend(discharge_temperature.tolist())
        all_time.extend(discharge_time_adjusted.tolist())
        all_charge_capacity.extend(charge_capacity_discharge.tolist())
        all_discharge_capacity.extend(discharge_capacity.tolist())
        
    except Exception as e:
        print(f"Error processing discharge cycle: {e}")
        return None
    
    # Process impedance data
    internal_resistance = None
    impedance_data = {}
    
    if impedance_cycle:
        try:
            impedance_measurements = impedance_cycle['data'].flat[0] if hasattr(impedance_cycle['data'], 'flat') else impedance_cycle['data'][0, 0]
            
            # Extract impedance fields
            impedance_fields = [
                'Sense_current', 'Battery_current', 'Current_ratio',
                'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct'
            ]
            
            for field in impedance_fields:
                if field in impedance_measurements.dtype.names:
                    field_data = impedance_measurements[field]
                    if field_data.ndim > 1:
                        field_data = field_data.flatten()
                    impedance_data[field.lower()] = field_data.tolist()
            
            # Use rectified impedance mean as internal resistance
            if 'rectified_impedance' in impedance_data and len(impedance_data['rectified_impedance']) > 0:
                internal_resistance = float(np.mean(impedance_data['rectified_impedance']))
            elif 're' in impedance_data and len(impedance_data['re']) > 0:
                internal_resistance = float(np.mean(impedance_data['re']))
        except Exception as e:
            print(f"Warning: Could not process impedance cycle: {e}")
    
    # Create unified cycle data
    return CycleData(
        cycle_number=cycle_number,
        voltage_in_V=all_voltage,
        current_in_A=all_current,  
        temperature_in_C=all_temperature,
        discharge_capacity_in_Ah=all_discharge_capacity,
        charge_capacity_in_Ah=all_charge_capacity,
        time_in_s=all_time,
        internal_resistance_in_ohm=internal_resistance,
        **impedance_data  # Add impedance data as additional fields
    )


def calc_Q_nasa(I, t, is_charge):
    """Calculate capacity like CALCE dataset - similar to calc_Q function"""
    Q = np.zeros_like(I)
    for i in range(1, len(I)):
        if is_charge and I[i] > 0:  # Charging current (positive)
            Q[i] = Q[i-1] + I[i] * (t[i] - t[i-1]) / 3600
        elif not is_charge and I[i] < 0:  # Discharging current (negative)
            Q[i] = Q[i-1] - I[i] * (t[i] - t[i-1]) / 3600  # Note: -I[i] because current is negative
        else:
            Q[i] = Q[i-1]  # No change in capacity
    return Q