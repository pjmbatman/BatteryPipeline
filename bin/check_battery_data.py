#!/usr/bin/env python3
"""
BatteryData 확인 스크립트
전처리된 pkl 파일의 BatteryData 구조를 확인합니다.
"""

import pickle
import sys
from pathlib import Path
from batteryml import BatteryData

def load_battery_data(pkl_path):
    """pkl 파일에서 BatteryData 로드"""
    with open(pkl_path, 'rb') as f:
        battery_data = pickle.load(f)
    return battery_data

def print_battery_info(battery_data):
    """BatteryData 정보 출력 (dict 또는 BatteryData 객체 모두 지원)"""
    # dict 형태인 경우 key로 접근, 객체인 경우 attribute로 접근
    def get_attr(data, key):
        if isinstance(data, dict):
            return data.get(key)
        else:
            return getattr(data, key, None)
    
    cell_id = get_attr(battery_data, 'cell_id')
    print(f"=== {cell_id} ===")
    print(f"Form Factor: {get_attr(battery_data, 'form_factor')}")
    print(f"Anode Material: {get_attr(battery_data, 'anode_material')}")
    print(f"Cathode Material: {get_attr(battery_data, 'cathode_material')}")
    print(f"Nominal Capacity: {get_attr(battery_data, 'nominal_capacity_in_Ah')} Ah")
    print(f"Voltage Limits: {get_attr(battery_data, 'min_voltage_limit_in_V')}V - {get_attr(battery_data, 'max_voltage_limit_in_V')}V")
    
    cycle_data = get_attr(battery_data, 'cycle_data')
    print(f"Total Cycles: {len(cycle_data) if cycle_data else 0}")
    
    # 방전 프로토콜 정보
    discharge_protocol = get_attr(battery_data, 'discharge_protocol')
    print(f"\nDischarge Protocol:")
    if discharge_protocol:
        if isinstance(discharge_protocol, list) and len(discharge_protocol) > 0:
            protocol = discharge_protocol[0]
            if hasattr(protocol, 'rate_in_C'):
                print(f"  Rate: {protocol.rate_in_C}C")
            if hasattr(protocol, 'end_voltage_in_V'):
                print(f"  End Voltage: {protocol.end_voltage_in_V}V")
    
    # 첫 번째와 마지막 사이클 정보
    if cycle_data:
        first_cycle = cycle_data[0]
        last_cycle = cycle_data[-1]
        
        def get_cycle_attr(cycle, key):
            if isinstance(cycle, dict):
                return cycle.get(key)
            else:
                return getattr(cycle, key, None)
        
        print(f"\nFirst Cycle (#{get_cycle_attr(first_cycle, 'cycle_number')}):")
        voltage_data = get_cycle_attr(first_cycle, 'voltage_in_V')
        current_data = get_cycle_attr(first_cycle, 'current_in_A')
        discharge_cap = get_cycle_attr(first_cycle, 'discharge_capacity_in_Ah')
        time_data = get_cycle_attr(first_cycle, 'time_in_s')
        
        if voltage_data:
            print(f"  Data Points: {len(voltage_data)}")
            print(f"  Voltage Range: {min(voltage_data):.2f}V - {max(voltage_data):.2f}V")
        if current_data:
            print(f"  Current Range: {min(current_data):.2f}A - {max(current_data):.2f}A")
        if discharge_cap:
            print(f"  Max Discharge Capacity: {max(discharge_cap):.3f}Ah")
        if time_data:
            print(f"  Duration: {max(time_data):.0f} seconds")
        
        print(f"\nLast Cycle (#{get_cycle_attr(last_cycle, 'cycle_number')}):")
        voltage_data = get_cycle_attr(last_cycle, 'voltage_in_V')
        current_data = get_cycle_attr(last_cycle, 'current_in_A')
        discharge_cap = get_cycle_attr(last_cycle, 'discharge_capacity_in_Ah')
        time_data = get_cycle_attr(last_cycle, 'time_in_s')
        
        if voltage_data:
            print(f"  Data Points: {len(voltage_data)}")
            print(f"  Voltage Range: {min(voltage_data):.2f}V - {max(voltage_data):.2f}V")
        if current_data:
            print(f"  Current Range: {min(current_data):.2f}A - {max(current_data):.2f}A")
        if discharge_cap:
            print(f"  Max Discharge Capacity: {max(discharge_cap):.3f}Ah")
        if time_data:
            print(f"  Duration: {max(time_data):.0f} seconds")
        
        # 용량 감소 추이
        capacities = []
        for cycle in cycle_data:
            discharge_cap = get_cycle_attr(cycle, 'discharge_capacity_in_Ah')
            if discharge_cap:
                capacities.append(max(discharge_cap))
        
        if capacities:
            print(f"\nCapacity Degradation:")
            print(f"  Initial: {capacities[0]:.3f}Ah")
            print(f"  Final: {capacities[-1]:.3f}Ah")
            print(f"  Capacity Fade: {((capacities[0] - capacities[-1]) / capacities[0] * 100):.1f}%")

def main():
    if len(sys.argv) > 1:
        # 특정 파일 지정
        pkl_path = Path(sys.argv[1])
        if not pkl_path.exists():
            print(f"Error: File not found: {pkl_path}")
            return
        
        battery_data = load_battery_data(pkl_path)
        print_battery_info(battery_data)
        
    else:
        # NASA 데이터 디렉토리에서 모든 파일 확인
        nasa_dir = Path("data/processed/NASA")
        
        if not nasa_dir.exists():
            print(f"Error: NASA data directory not found: {nasa_dir}")
            print("Please run preprocessing first:")
            print("uv run batteryml preprocess NASA data/raw/NASA data/processed/NASA")
            return
        
        pkl_files = list(nasa_dir.glob("NASA_B*.pkl"))
        
        if not pkl_files:
            print(f"No NASA pkl files found in {nasa_dir}")
            return
        
        pkl_files = sorted(pkl_files)
        print(f"Found {len(pkl_files)} NASA battery files\n")
        
        # 처음 3개 파일만 자세히 보기
        for i, pkl_file in enumerate(pkl_files[:3]):
            battery_data = load_battery_data(pkl_file)
            print_battery_info(battery_data)
            print("\n" + "="*50 + "\n")
        
        if len(pkl_files) > 3:
            print(f"... and {len(pkl_files) - 3} more files")
            print("\nTo check a specific battery file:")
            print("python check_nasa_data.py data/processed/NASA/NASA_B0005.pkl")

if __name__ == "__main__":
    main()