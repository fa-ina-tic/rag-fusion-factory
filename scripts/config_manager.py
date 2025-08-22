#!/usr/bin/env python3
"""Configuration management CLI tool for RAG Fusion Factory."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.utils import (
    validate_config,
    print_config_summary,
    export_config_to_file,
    create_user_config_from_template,
    update_user_config,
    get_environment,
    set_environment,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Fusion Factory Configuration Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    subparsers.add_parser("validate", help="Validate current configuration")
    
    # Summary command
    subparsers.add_parser("summary", help="Print configuration summary")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export configuration to file")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Output format")
    
    # Init command
    subparsers.add_parser("init", help="Initialize user configuration from template")
    
    # Environment command
    env_parser = subparsers.add_parser("env", help="Manage environment settings")
    env_parser.add_argument("--get", action="store_true", help="Get current environment")
    env_parser.add_argument("--set", help="Set environment")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update user configuration")
    update_parser.add_argument("key", help="Configuration key (dot notation)")
    update_parser.add_argument("value", help="Configuration value")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "validate":
            if validate_config():
                print("✅ Configuration is valid")
            else:
                print("❌ Configuration validation failed")
                sys.exit(1)
        
        elif args.command == "summary":
            print_config_summary()
        
        elif args.command == "export":
            export_config_to_file(args.output, args.format)
        
        elif args.command == "init":
            create_user_config_from_template()
        
        elif args.command == "env":
            if args.get:
                print(f"Current environment: {get_environment()}")
            elif args.set:
                set_environment(args.set)
                print(f"Environment set to: {args.set}")
            else:
                print(f"Current environment: {get_environment()}")
        
        elif args.command == "update":
            # Parse value type
            value = args.value
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
            
            # Create nested dictionary for the update
            keys = args.key.split(".")
            update_dict = {}
            current = update_dict
            for key in keys[:-1]:
                current[key] = {}
                current = current[key]
            current[keys[-1]] = value
            
            update_user_config(update_dict)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()