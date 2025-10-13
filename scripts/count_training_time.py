#!/usr/bin/env python3
"""
Script to count the total training time from all log directories in the repository.

This script:
1. Recursively scans all log directories for TensorBoard event files
2. Extracts the start and end timestamps from each training run
3. Calculates the duration of each run
4. Aggregates the total training time across all runs
5. Provides detailed breakdown by date and run

Usage:
    python scripts/count_training_time.py
    python scripts/count_training_time.py --log-dir logs/sb3/Fre25-Isaaclabsym-Direct-v0
    python scripts/count_training_time.py --verbose
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import struct


def read_tfevents_timestamps(event_file_path):
    """
    Extract start and end timestamps from a TensorBoard event file.

    Args:
        event_file_path: Path to the event file

    Returns:
        tuple: (start_time, end_time) in seconds since epoch, or (None, None) if failed
    """
    try:
        # Try using TensorBoard's EventAccumulator (if available)
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            ea = EventAccumulator(str(event_file_path))
            ea.Reload()

            # Get all scalar tags
            tags = ea.Tags()

            # Use any scalar to get timestamps (they all share the same timeline)
            if 'scalars' in tags and len(tags['scalars']) > 0:
                # Get the first available scalar
                first_tag = tags['scalars'][0]
                events = ea.Scalars(first_tag)

                if len(events) > 0:
                    start_time = events[0].wall_time
                    end_time = events[-1].wall_time
                    return start_time, end_time
        except ImportError:
            # TensorBoard not available, fall back to manual parsing
            pass

        # Manual parsing of event file (fallback)
        # TensorFlow event file format: [record_length][crc][data][crc]
        # We'll just look at file timestamps as fallback
        with open(event_file_path, 'rb') as f:
            # Read file creation time from filesystem
            stat = os.stat(event_file_path)
            start_time = stat.st_ctime
            end_time = stat.st_mtime
            return start_time, end_time

    except Exception as e:
        print(f"[WARN] Failed to read {event_file_path}: {e}")
        return None, None


def format_duration(seconds):
    """Format duration in seconds to human-readable format."""
    if seconds is None:
        return "N/A"

    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or len(parts) == 0:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def scan_log_directory(log_root, verbose=False):
    """
    Recursively scan log directory for training runs and calculate durations.

    Args:
        log_root: Root directory to scan for logs
        verbose: Whether to print detailed information

    Returns:
        dict: Dictionary mapping run names to durations
    """
    log_root = Path(log_root)

    if not log_root.exists():
        print(f"[ERROR] Log directory does not exist: {log_root}")
        return {}

    run_durations = {}

    # Find all event files recursively
    print(f"[INFO] Scanning for TensorBoard event files...")
    event_files = list(log_root.rglob("events.out.tfevents.*"))

    print(f"[INFO] Found {len(event_files)} TensorBoard event files")

    for idx, event_file in enumerate(event_files):
        # Progress indicator every 50 files
        if (idx + 1) % 50 == 0 or idx == len(event_files) - 1:
            print(f"[INFO] Processing file {idx + 1}/{len(event_files)}...", end='\r')

        # Get the run directory (parent of event file)
        run_dir = event_file.parent
        run_name = run_dir.relative_to(log_root)

        # Extract timestamps
        start_time, end_time = read_tfevents_timestamps(event_file)

        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            run_durations[str(run_name)] = duration

            if verbose:
                start_dt = datetime.fromtimestamp(start_time)
                end_dt = datetime.fromtimestamp(end_time)
                print(f"\n  {run_name}")
                print(f"    Start:    {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    End:      {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Duration: {format_duration(duration)}")
        else:
            if verbose:
                print(f"\n  {run_name}")
                print(f"    [WARN] Failed to extract timestamps")

    print()  # New line after progress indicator

    return run_durations


def group_by_date(run_durations):
    """
    Group runs by date for better organization.

    Args:
        run_durations: Dictionary mapping run names to durations

    Returns:
        dict: Dictionary mapping dates to list of (run_name, duration) tuples
    """
    grouped = defaultdict(list)

    for run_name, duration in run_durations.items():
        # Extract date from run name (format: YYYY-MM-DD_HH-MM-SS or similar)
        parts = run_name.split('/')
        if len(parts) > 0:
            # Try to find a date-like string
            for part in parts:
                if len(part) >= 10 and part[:10].replace('-', '').replace('_', '').isdigit():
                    date = part[:10]
                    grouped[date].append((run_name, duration))
                    break
            else:
                # No date found, use "unknown"
                grouped["unknown"].append((run_name, duration))
        else:
            grouped["unknown"].append((run_name, duration))

    return grouped


def print_summary(run_durations, verbose=False):
    """
    Print summary of training times.

    Args:
        run_durations: Dictionary mapping run names to durations
        verbose: Whether to print detailed breakdown
    """
    if len(run_durations) == 0:
        print("\n[INFO] No training runs found!")
        return

    # Calculate total
    total_seconds = sum(run_durations.values())

    print("\n" + "=" * 80)
    print("TRAINING TIME SUMMARY")
    print("=" * 80)

    # Overall statistics
    print(f"\nTotal Runs:         {len(run_durations)}")
    print(f"Total Training Time: {format_duration(total_seconds)}")
    print(f"                     ({total_seconds/3600:.2f} hours)")
    print(f"                     ({total_seconds/86400:.2f} days)")

    if len(run_durations) > 0:
        avg_duration = total_seconds / len(run_durations)
        print(f"Average Run Time:    {format_duration(avg_duration)}")
        print(f"Longest Run:         {format_duration(max(run_durations.values()))}")
        print(f"Shortest Run:        {format_duration(min(run_durations.values()))}")

    # Group by date
    grouped = group_by_date(run_durations)

    if verbose or len(grouped) <= 20:
        print("\n" + "-" * 80)
        print("BREAKDOWN BY DATE")
        print("-" * 80)

        for date in sorted(grouped.keys()):
            runs = grouped[date]
            date_total = sum(duration for _, duration in runs)

            print(f"\n{date}: {len(runs)} runs, {format_duration(date_total)}")

            if verbose:
                for run_name, duration in sorted(runs):
                    print(f"  • {run_name}: {format_duration(duration)}")
    else:
        print(f"\n[INFO] Training spans {len(grouped)} dates")
        print(f"[INFO] Use --verbose to see breakdown by date")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Count total training time from log directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/sb3/Fre25-Isaaclabsym-Direct-v0",
        help="Root log directory to scan (default: logs/sb3/Fre25-Isaaclabsym-Direct-v0)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about each run"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file to save summary (optional)"
    )

    args = parser.parse_args()

    # Get the script directory and workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent

    # Resolve log directory path
    log_dir = workspace_root / args.log_dir

    print(f"[INFO] Scanning log directory: {log_dir}")

    # Scan directory
    run_durations = scan_log_directory(log_dir, verbose=args.verbose)

    # Print summary
    print_summary(run_durations, verbose=args.verbose)

    # Save to file if requested
    if args.output:
        output_path = workspace_root / args.output
        with open(output_path, 'w') as f:
            f.write("Training Time Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Runs: {len(run_durations)}\n")
            f.write(f"Total Training Time: {format_duration(sum(run_durations.values()))}\n")
            f.write(f"                     ({sum(run_durations.values())/3600:.2f} hours)\n")
            f.write(f"                     ({sum(run_durations.values())/86400:.2f} days)\n\n")

            grouped = group_by_date(run_durations)
            f.write("Breakdown by Date\n")
            f.write("-" * 80 + "\n\n")

            for date in sorted(grouped.keys()):
                runs = grouped[date]
                date_total = sum(duration for _, duration in runs)
                f.write(f"\n{date}: {len(runs)} runs, {format_duration(date_total)}\n")

                for run_name, duration in sorted(runs):
                    f.write(f"  • {run_name}: {format_duration(duration)}\n")

        print(f"\n[INFO] Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
