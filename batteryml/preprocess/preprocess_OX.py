# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import scipy.io

from tqdm import tqdm
from typing import List
from pathlib import Path

from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor


@PREPROCESSORS.register()
class OXPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        path = Path(parentdir)
        
        process_batteries_num = 0
        skip_batteries_num = 0
        
        # Process .mat files directly
        mat_files = list(path.glob('*.mat'))
        if mat_files:
            for mat_file in tqdm(mat_files, desc='Processing mat files'):
                try:
                    mat_data = scipy.io.loadmat(mat_file)
                    cell_keys = [k for k in mat_data.keys() if k.startswith('Cell')]
                    
                    for cell_key in tqdm(cell_keys, desc=f'Processing cells in {mat_file.name}', leave=False):
                        # Check if already processed
                        whether_to_skip = self.check_processed_file(cell_key)
                        if whether_to_skip:
                            skip_batteries_num += 1
                            continue
                            
                        battery = self.process_cell_from_mat(mat_data[cell_key][0, 0], cell_key)
                        if battery:
                            self.dump_single_file(battery)
                            process_batteries_num += 1
                            
                            if not self.silent:
                                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')
                                
                except Exception as e:
                    if not self.silent:
                        tqdm.write(f"Error processing {mat_file.name}: {e}")
        

        return process_batteries_num, skip_batteries_num

    def process_cell_from_mat(self, cell_data, cell_name):
        """Process a single cell directly from mat data"""
        try:
            cycle_data = []
            
            # Extract each cycle
            for cycle_field in tqdm(cell_data.dtype.names, desc=f'Processing cycles for {cell_name}', leave=False):
                cycle_num = int(cycle_field.replace('cyc', ''))
                cycle_data_raw = cell_data[cycle_field][0, 0]
                
                if len(cycle_data_raw) > 0:
                    # Combine charge and discharge phases
                    all_voltage = []
                    all_current = []
                    all_temperature = []
                    all_discharge_capacity = []
                    all_charge_capacity = []
                    all_time = []
                    
                    for phase_idx, phase in enumerate(cycle_data_raw):
                        phase_data = phase[0]
                        if len(phase_data) > 0:
                            t = phase_data['t'][0].flatten()
                            v = phase_data['v'][0].flatten()
                            q = phase_data['q'][0].flatten()
                            T = phase_data['T'][0].flatten()
                            
                            # Convert time from days to seconds
                            time_s = (t - t[0]) * 24 * 3600
                            all_time.extend(time_s)
                            all_voltage.extend(v)
                            all_temperature.extend(T)
                            
                            # Handle capacity and current
                            charge_cap = [max(0, qi) for qi in q]
                            discharge_cap = [abs(min(0, qi)) for qi in q]
                            all_charge_capacity.extend(charge_cap)
                            all_discharge_capacity.extend(discharge_cap)
                            
                            # Calculate current from capacity change
                            current = [0.0]  # First point has no previous point
                            for i in range(1, len(q)):
                                dt = time_s[i] - time_s[i-1] if time_s[i] != time_s[i-1] else 1
                                dq = q[i] - q[i-1]
                                current.append(dq * 3600 / dt)
                            all_current.extend(current)
                    
                    if all_voltage:
                        cycle_data.append(CycleData(
                            cycle_number=cycle_num,
                            voltage_in_V=all_voltage,
                            current_in_A=all_current,
                            temperature_in_C=all_temperature,
                            discharge_capacity_in_Ah=all_discharge_capacity,
                            charge_capacity_in_Ah=all_charge_capacity,
                            time_in_s=all_time
                        ))
            
            if cycle_data:
                # Charge Protocol is constant current
                charge_protocol = [CyclingProtocol(
                    rate_in_C=2.0, start_soc=0.0, end_soc=1.0
                )]
                discharge_protocol = [CyclingProtocol(
                    rate_in_C=1.0, start_soc=1.0, end_soc=0.0
                )]

                return BatteryData(
                    cell_id=cell_name,
                    cycle_data=cycle_data,
                    form_factor='pouch',
                    anode_material='graphite',
                    cathode_material='LCO',
                    discharge_protocol=discharge_protocol,
                    charge_protocol=charge_protocol,
                    nominal_capacity_in_Ah=0.72,
                    min_voltage_limit_in_V=2.7,
                    max_voltage_limit_in_V=4.2
                )
        except Exception as e:
            if not self.silent:
                tqdm.write(f"Error processing cell {cell_name}: {e}")
            return None
