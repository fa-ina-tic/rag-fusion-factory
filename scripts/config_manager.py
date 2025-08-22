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
    get_environment,
    set_environment,
    show_config_edit_instructions,
    suggest_config_edit,
    get_user_config_path,
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
    
    # Edit command
    edit_parser = subparsers.add_parser("edit", help="Show how to edit configuration")
    edit_parser.add_argument("--key", help="Configuration key to get edit instructions for")
    edit_parser.add_argument("--value", help="Desired value (used with --key)")
    
    # Instructions command
    subparsers.add_parser("instructions", help="Show configuration editing instructions")
    
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
        
        elif args.command == "edit":
            if args.key and args.value:
                # Parse value type
                value = args.value
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "").isdigit():
                    value = float(value)
                
                suggest_config_edit(args.key, value)
            elif args.key:
                current_value = get_config_value(args.key)
                print(f"Current value of {args.key}: {current_value}")
                print("")
                suggest_config_edit(args.key, "<new_value>")
            else:
                user_config_path = get_user_config_path()
                print(f"Edit configuration files:")
                print(f"  User config: {user_config_path}")
                print(f"  Environment configs: config/development.yaml, config/production.yaml")
                print("")
                print("Use 'config_manager.py edit --key <key> --value <value>' for specific guidance")
        
        elif args.command == "instructions":
            show_config_edit_instructions()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()