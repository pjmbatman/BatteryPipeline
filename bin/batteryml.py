#!/usr/bin/env python

# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import argparse
import pickle

from pathlib import Path

from batteryml.preprocess import (
    DOWNLOAD_LINKS, download_file, SUPPORTED_SOURCES
)
from batteryml.pipeline import Pipeline
from batteryml.builders import PREPROCESSORS


def main():
    parser = argparse.ArgumentParser('BatteryML command line utilities.')
    subparsers = parser.add_subparsers()

    # download command
    download_parser = subparsers.add_parser(
        "download", help="Download raw files for public datasets")
    download_parser.add_argument(
        "dataset", choices=list(DOWNLOAD_LINKS.keys()),
        help="Public dataset to download")
    download_parser.add_argument(
        "output_dir", help="Directory to save the raw data files")
    download_parser.set_defaults(func=download)

    # preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Organize the raw data files into BatteryData and save to disk")
    preprocess_parser.add_argument(
        "input_type", choices=[value for values in SUPPORTED_SOURCES.values() for value in values],
        help="Type of input raw files. For public datasets, specific "
             "preprocessor will be called. For standard battery test "
             "output files, the corresponding preprocessing logic "
             "will be applied.")
    preprocess_parser.add_argument(
        "--config", default="None",
        help="Path to the config file of Cycler.")
    preprocess_parser.add_argument(
        "raw_dir", help="Directory of raw input files.")
    preprocess_parser.add_argument(
        "output_dir", help="Directory to save the BatteryData files.")
    preprocess_parser.add_argument(
        "-q", "--quiet", "--silent", dest="silent",
        action="store_true", help="Suppress logs during preprocessing.")
    preprocess_parser.set_defaults(func=preprocess)

    # run command
    run_parser = subparsers.add_parser(
        "run", help="Run the given config for training or evaluation")
    run_parser.add_argument(
        "config", help="Path to the config file")
    run_parser.add_argument(
        "--workspace", type=str, default=None, help="Directory to save the checkpoints and predictions.")
    run_parser.add_argument(
        "--device", default="cpu", help="Running device")
    run_parser.add_argument(
        "--ckpt-to-resume", "--ckpt_to_resume", dest="ckpt_to_resume",
        help="path to the checkpoint to resume training or evaluation")
    run_parser.add_argument(
        "--train", action="store_true",
        help="Run training. Will skip training if this flag is not provided.")
    run_parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation. Will skip eval if this flag is not provided.")
    run_parser.add_argument(
        "--metric", default="RMSE,MAE,MAPE",
        help="Metrics for evaluation, seperated by comma")
    run_parser.add_argument(
        "--seed", type=int, default=0, help="random seed")
    run_parser.add_argument(
        "--epochs", type=int, help="number of epochs override")
    run_parser.add_argument(
        "--skip_if_executed", type=str, default='False', help="skip train/evaluate if the model executed")
    run_parser.set_defaults(func=run)

    # check command
    check_parser = subparsers.add_parser(
        "check", help="Check processed BatteryData pkl files")
    check_parser.add_argument(
        "data_path", nargs='?', 
        help="Path to pkl file or directory containing pkl files. If not provided, checks data/processed/")
    check_parser.add_argument(
        "--limit", type=int, default=3,
        help="Maximum number of files to show detailed info (default: 3)")
    check_parser.set_defaults(func=check)

    args = parser.parse_args()
    args.func(args)


def download(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    raw_dir = Path(args.output_dir)
    for f in DOWNLOAD_LINKS[args.dataset]:
        if len(f) == 2:
            (url, filename), total_length = f, None
        else:
            url, filename, total_length = f
        download_file(url, raw_dir / filename, total_length=total_length)


def preprocess(args):
    assert os.path.exists(
        args.raw_dir), f'Input path not exist: {args.raw_dir}'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_path = Path(args.config)
    input_path, output_path = Path(args.raw_dir), Path(args.output_dir)
    processor = PREPROCESSORS.build(dict(
        name=f'{args.input_type}Preprocessor',
        output_dir=output_path,
        silent=args.silent
    ))
    processor(input_path, config_path=config_path)


def run(args):
    # Convert skip_if_executed to boolean
    args.skip_if_executed = args.skip_if_executed.lower() in ['true', '1', 'yes']
    pipeline = Pipeline(args.config, args.workspace)
    model, dataset = None, None  # Reuse to save setup cost
    if args.train:
        model, dataset = pipeline.train(
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
            ckpt_to_resume=args.ckpt_to_resume,
            dataset=dataset,
            skip_if_executed=args.skip_if_executed)
    if args.eval:
        metric = args.metric.split(',')
        pipeline.evaluate(
            seed=args.seed,
            device=args.device,
            metric=metric,
            model=model,
            dataset=dataset,
            ckpt_to_resume=args.ckpt_to_resume,
            skip_if_executed=args.skip_if_executed
        )


def check(args):
    """Check processed BatteryData pkl files"""
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path("data/processed")
    
    if not data_path.exists():
        print(f"Error: Path not found: {data_path}")
        if not args.data_path:
            print("Please run preprocessing first or specify a valid path")
        return
    
    if data_path.is_file():
        # Single file
        if data_path.suffix == '.pkl':
            print_battery_info(load_battery_data(data_path))
        else:
            print(f"Error: {data_path} is not a pkl file")
    else:
        # Directory - find all pkl files
        pkl_files = list(data_path.rglob("*.pkl"))
        
        if not pkl_files:
            print(f"No pkl files found in {data_path}")
            return
        
        pkl_files = sorted(pkl_files)
        print(f"Found {len(pkl_files)} battery pkl files\n")
        
        # Show detailed info for first few files
        for i, pkl_file in enumerate(pkl_files[:args.limit]):
            battery_data = load_battery_data(pkl_file)
            print_battery_info(battery_data)
            print("\n" + "="*50 + "\n")
        
        if len(pkl_files) > args.limit:
            print(f"... and {len(pkl_files) - args.limit} more files")
            print(f"\nTo check specific files:")
            print(f"batteryml check {pkl_files[0]}")


def load_battery_data(pkl_path):
    """Load BatteryData from pkl file"""
    with open(pkl_path, 'rb') as f:
        battery_data = pickle.load(f)
    return battery_data


def print_battery_info(battery_data):
    """Print BatteryData information (supports both dict and object) - shows ALL attributes"""
    def get_attr(data, key):
        if isinstance(data, dict):
            return data.get(key)
        else:
            return getattr(data, key, None)
    
    def format_protocol_list(protocols):
        if not protocols:
            return "None"
        result = []
        for i, protocol in enumerate(protocols):
            protocol_info = []
            if hasattr(protocol, 'rate_in_C') and protocol.rate_in_C is not None:
                protocol_info.append(f"rate: {protocol.rate_in_C}C")
            if hasattr(protocol, 'current_in_A') and protocol.current_in_A is not None:
                protocol_info.append(f"current: {protocol.current_in_A}A")
            if hasattr(protocol, 'voltage_in_V') and protocol.voltage_in_V is not None:
                protocol_info.append(f"voltage: {protocol.voltage_in_V}V")
            if hasattr(protocol, 'power_in_W') and protocol.power_in_W is not None:
                protocol_info.append(f"power: {protocol.power_in_W}W")
            if hasattr(protocol, 'start_voltage_in_V') and protocol.start_voltage_in_V is not None:
                protocol_info.append(f"start_V: {protocol.start_voltage_in_V}V")
            if hasattr(protocol, 'start_soc') and protocol.start_soc is not None:
                protocol_info.append(f"start_soc: {protocol.start_soc}")
            if hasattr(protocol, 'end_voltage_in_V') and protocol.end_voltage_in_V is not None:
                protocol_info.append(f"end_V: {protocol.end_voltage_in_V}V")
            if hasattr(protocol, 'end_soc') and protocol.end_soc is not None:
                protocol_info.append(f"end_soc: {protocol.end_soc}")
            result.append(f"[{i+1}] {', '.join(protocol_info) if protocol_info else 'empty'}")
        return "\n    ".join(result)
    
    cell_id = get_attr(battery_data, 'cell_id')
    print(f"**************description of battery cell {cell_id}**************")
    
    # All BatteryData attributes
    print(f"cell_id: {get_attr(battery_data, 'cell_id')}")
    print(f"form_factor: {get_attr(battery_data, 'form_factor')}")
    print(f"anode_material: {get_attr(battery_data, 'anode_material')}")
    print(f"cathode_material: {get_attr(battery_data, 'cathode_material')}")
    print(f"electrolyte_material: {get_attr(battery_data, 'electrolyte_material')}")
    print(f"nominal_capacity_in_Ah: {get_attr(battery_data, 'nominal_capacity_in_Ah')}")
    print(f"depth_of_charge: {get_attr(battery_data, 'depth_of_charge')}")
    print(f"depth_of_discharge: {get_attr(battery_data, 'depth_of_discharge')}")
    print(f"already_spent_cycles: {get_attr(battery_data, 'already_spent_cycles')}")
    print(f"max_voltage_limit_in_V: {get_attr(battery_data, 'max_voltage_limit_in_V')}")
    print(f"min_voltage_limit_in_V: {get_attr(battery_data, 'min_voltage_limit_in_V')}")
    print(f"max_current_limit_in_A: {get_attr(battery_data, 'max_current_limit_in_A')}")
    print(f"min_current_limit_in_A: {get_attr(battery_data, 'min_current_limit_in_A')}")
    print(f"reference: {get_attr(battery_data, 'reference')}")
    print(f"description: {get_attr(battery_data, 'description')}")
    
    # Protocols
    charge_protocol = get_attr(battery_data, 'charge_protocol')
    print(f"charge_protocol:")
    if charge_protocol:
        print(f"    {format_protocol_list(charge_protocol)}")
    else:
        print(f"    None")
    
    discharge_protocol = get_attr(battery_data, 'discharge_protocol')
    print(f"discharge_protocol:")
    if discharge_protocol:
        print(f"    {format_protocol_list(discharge_protocol)}")
    else:
        print(f"    None")
    
    # Cycle data
    cycle_data = get_attr(battery_data, 'cycle_data')
    print(f"cycle length: {len(cycle_data) if cycle_data else 0}")
    
    # Show additional attributes that might exist from kwargs
    if isinstance(battery_data, dict):
        # For dict data, show any additional keys not in the standard attributes
        standard_attrs = {
            'cell_id', 'cycle_data', 'form_factor', 'anode_material', 'cathode_material',
            'electrolyte_material', 'nominal_capacity_in_Ah', 'depth_of_charge', 
            'depth_of_discharge', 'already_spent_cycles', 'charge_protocol', 
            'discharge_protocol', 'max_voltage_limit_in_V', 'min_voltage_limit_in_V',
            'max_current_limit_in_A', 'min_current_limit_in_A', 'reference', 'description'
        }
        additional_attrs = set(battery_data.keys()) - standard_attrs
        for attr in sorted(additional_attrs):
            print(f"{attr}: {battery_data[attr]}")
    else:
        # For object data, check for any additional attributes
        if hasattr(battery_data, '__dict__'):
            standard_attrs = {
                'cell_id', 'cycle_data', 'form_factor', 'anode_material', 'cathode_material',
                'electrolyte_material', 'nominal_capacity_in_Ah', 'depth_of_charge', 
                'depth_of_discharge', 'already_spent_cycles', 'charge_protocol', 
                'discharge_protocol', 'max_voltage_limit_in_V', 'min_voltage_limit_in_V',
                'max_current_limit_in_A', 'min_current_limit_in_A', 'reference', 'description'
            }
            additional_attrs = set(battery_data.__dict__.keys()) - standard_attrs
            for attr in sorted(additional_attrs):
                print(f"{attr}: {getattr(battery_data, attr)}")
    
    # Detailed cycle information if cycles exist
    if cycle_data and len(cycle_data) > 0:
        print(f"\n--- Cycle Data Details ---")
        first_cycle = cycle_data[0]
        last_cycle = cycle_data[-1]
        
        def get_cycle_attr(cycle, key):
            if isinstance(cycle, dict):
                return cycle.get(key)
            else:
                return getattr(cycle, key, None)
        
        def show_cycle_details(cycle, cycle_name):
            print(f"\n{cycle_name} (#{get_cycle_attr(cycle, 'cycle_number')}):")
            voltage_data = get_cycle_attr(cycle, 'voltage_in_V')
            current_data = get_cycle_attr(cycle, 'current_in_A')
            charge_cap = get_cycle_attr(cycle, 'charge_capacity_in_Ah')
            discharge_cap = get_cycle_attr(cycle, 'discharge_capacity_in_Ah')
            time_data = get_cycle_attr(cycle, 'time_in_s')
            temp_data = get_cycle_attr(cycle, 'temperature_in_C')
            resistance = get_cycle_attr(cycle, 'internal_resistance_in_ohm')
            
            def format_first_values(data, precision=3):
                """Format first 5 values of a list for display"""
                if not data or len(data) == 0:
                    return "[]"
                sample_size = min(5, len(data))
                if isinstance(data[0], (int, float)):
                    values = [f"{val:.{precision}f}" for val in data[:sample_size]]
                else:
                    values = [str(val) for val in data[:sample_size]]
                suffix = "..." if len(data) > sample_size else ""
                return f"[{', '.join(values)}{suffix}]"
            
            if voltage_data:
                first_values = format_first_values(voltage_data, 2)
                print(f"  voltage_in_V: {len(voltage_data)} points, range: {min(voltage_data):.2f}V - {max(voltage_data):.2f}V")
                print(f"    first 5 values: {first_values}")
            else:
                print(f"  voltage_in_V: None")
                
            if current_data:
                first_values = format_first_values(current_data, 2)
                print(f"  current_in_A: {len(current_data)} points, range: {min(current_data):.2f}A - {max(current_data):.2f}A")
                print(f"    first 5 values: {first_values}")
            else:
                print(f"  current_in_A: None")
                
            if charge_cap:
                first_values = format_first_values(charge_cap, 3)
                print(f"  charge_capacity_in_Ah: {len(charge_cap)} points, max: {max(charge_cap):.3f}Ah")
                print(f"    first 5 values: {first_values}")
            else:
                print(f"  charge_capacity_in_Ah: None")
                
            if discharge_cap:
                first_values = format_first_values(discharge_cap, 3)
                print(f"  discharge_capacity_in_Ah: {len(discharge_cap)} points, max: {max(discharge_cap):.3f}Ah")
                print(f"    first 5 values: {first_values}")
            else:
                print(f"  discharge_capacity_in_Ah: None")
                
            if time_data:
                first_values = format_first_values(time_data, 0)
                print(f"  time_in_s: {len(time_data)} points, duration: {max(time_data):.0f}s")
                print(f"    first 5 values: {first_values}")
            else:
                print(f"  time_in_s: None")
                
            if temp_data:
                first_values = format_first_values(temp_data, 1)
                print(f"  temperature_in_C: {len(temp_data)} points, range: {min(temp_data):.1f}°C - {max(temp_data):.1f}°C")
                print(f"    first 5 values: {first_values}")
            else:
                print(f"  temperature_in_C: None")
                
            print(f"  internal_resistance_in_ohm: {resistance}")
            
            # Show any additional cycle data
            if isinstance(cycle, dict):
                standard_cycle_attrs = {
                    'cycle_number', 'voltage_in_V', 'current_in_A', 'charge_capacity_in_Ah',
                    'discharge_capacity_in_Ah', 'time_in_s', 'temperature_in_C', 'internal_resistance_in_ohm'
                }
                additional_cycle_attrs = set(cycle.keys()) - standard_cycle_attrs
                for attr in sorted(additional_cycle_attrs):
                    attr_data = cycle[attr]
                    if isinstance(attr_data, list) and len(attr_data) > 0:
                        first_values = format_first_values(attr_data)
                        print(f"  {attr}: {len(attr_data)} points")
                        print(f"    first 5 values: {first_values}")
                    else:
                        print(f"  {attr}: {attr_data}")
            elif hasattr(cycle, 'additional_data') and cycle.additional_data:
                for key, val in cycle.additional_data.items():
                    if isinstance(val, list) and len(val) > 0:
                        first_values = format_first_values(val)
                        print(f"  {key}: {len(val)} points")
                        print(f"    first 5 values: {first_values}")
                    else:
                        print(f"  {key}: {val}")
        
        show_cycle_details(first_cycle, "First Cycle")
        if len(cycle_data) > 1:
            show_cycle_details(last_cycle, "Last Cycle")
        
        # Capacity degradation analysis
        capacities = []
        for cycle in cycle_data:
            discharge_cap = get_cycle_attr(cycle, 'discharge_capacity_in_Ah')
            if discharge_cap:
                capacities.append(max(discharge_cap))
        
        if capacities and len(capacities) > 1:
            print(f"\n--- Capacity Degradation ---")
            print(f"Initial capacity: {capacities[0]:.3f}Ah")
            print(f"Final capacity: {capacities[-1]:.3f}Ah")
            print(f"Capacity fade: {((capacities[0] - capacities[-1]) / capacities[0] * 100):.1f}%")


if __name__ == "__main__":
    main()
