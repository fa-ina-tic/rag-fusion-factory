#!/usr/bin/env python3
"""Test runner for comprehensive integration tests.

This script runs all comprehensive integration tests and generates a detailed report.
It can be used to validate the complete RAG Fusion Factory system.

Usage:
    python tests/run_integration_tests.py [--include-real-engines] [--include-performance] [--report-file output.json]
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_pytest_with_capture(test_file: str, extra_args: List[str] = None) -> Dict[str, Any]:
    """Run pytest on a test file and capture results."""
    cmd = [
        sys.executable, "-m", "pytest",
        test_file,
        "-v",
        "--tb=short"
    ]
    
    # Add JSON reporting if available
    try:
        import pytest_json_report
        cmd.extend(["--json-report", "--json-report-file=temp_report.json"])
    except ImportError:
        pass
    
    if extra_args:
        cmd.extend(extra_args)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        end_time = time.time()
        
        # Load JSON report if available
        report_data = {}
        if Path("temp_report.json").exists():
            try:
                with open("temp_report.json", "r") as f:
                    report_data = json.load(f)
                Path("temp_report.json").unlink()  # Clean up
            except Exception:
                pass
        
        return {
            "test_file": test_file,
            "return_code": result.returncode,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "report_data": report_data,
            "success": result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            "test_file": test_file,
            "return_code": -1,
            "duration": time.time() - start_time,
            "stdout": "",
            "stderr": "Test timed out after 5 minutes",
            "report_data": {},
            "success": False
        }
    except Exception as e:
        return {
            "test_file": test_file,
            "return_code": -1,
            "duration": time.time() - start_time,
            "stdout": "",
            "stderr": f"Error running test: {str(e)}",
            "report_data": {},
            "success": False
        }


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {}
    
    # Check Python packages
    required_packages = [
        "pytest", "pytest-asyncio", "numpy", "psutil", "docker", "requests", "aiohttp"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            dependencies[f"python_{package}"] = True
        except ImportError:
            dependencies[f"python_{package}"] = False
    
    # Check Docker services (if requested)
    try:
        import socket
        
        services = [
            ("elasticsearch", 9200),
            ("solr", 8983),
            ("opensearch", 9200)
        ]
        
        for service, port in services:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                dependencies[f"docker_{service}"] = result == 0
            except Exception:
                dependencies[f"docker_{service}"] = False
                
    except ImportError:
        pass
    
    return dependencies


def generate_report(test_results: List[Dict[str, Any]], dependencies: Dict[str, bool], 
                   output_file: str = None) -> Dict[str, Any]:
    """Generate comprehensive test report."""
    
    # Calculate summary statistics
    total_tests = len(test_results)
    successful_tests = sum(1 for r in test_results if r["success"])
    failed_tests = total_tests - successful_tests
    total_duration = sum(r["duration"] for r in test_results)
    
    # Extract test details from pytest reports
    test_details = []
    for result in test_results:
        if result["report_data"] and "tests" in result["report_data"]:
            for test in result["report_data"]["tests"]:
                test_details.append({
                    "test_file": result["test_file"],
                    "test_name": test.get("nodeid", "unknown"),
                    "outcome": test.get("outcome", "unknown"),
                    "duration": test.get("duration", 0.0),
                    "error": test.get("call", {}).get("longrepr", "") if test.get("outcome") == "failed" else ""
                })
    
    # Create comprehensive report
    report = {
        "summary": {
            "total_test_files": total_tests,
            "successful_test_files": successful_tests,
            "failed_test_files": failed_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
            "total_duration_seconds": total_duration,
            "average_duration_per_file": total_duration / total_tests if total_tests > 0 else 0.0
        },
        "dependencies": dependencies,
        "test_file_results": test_results,
        "individual_test_results": test_details,
        "recommendations": []
    }
    
    # Add recommendations based on results
    if failed_tests > 0:
        report["recommendations"].append("Some tests failed. Check individual test results for details.")
    
    if not dependencies.get("docker_elasticsearch", False):
        report["recommendations"].append("Elasticsearch not available. Real engine integration tests were skipped.")
    
    if not dependencies.get("docker_solr", False):
        report["recommendations"].append("Solr not available. Real engine integration tests were skipped.")
    
    if successful_tests == total_tests:
        report["recommendations"].append("All tests passed! The RAG Fusion Factory system is working correctly.")
    
    # Save report if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed report saved to: {output_file}")
    
    return report


def print_summary(report: Dict[str, Any]):
    """Print a summary of test results."""
    summary = report["summary"]
    
    print("\n" + "="*60)
    print("RAG FUSION FACTORY - COMPREHENSIVE INTEGRATION TEST RESULTS")
    print("="*60)
    
    print(f"\nTest Files Summary:")
    print(f"  Total test files: {summary['total_test_files']}")
    print(f"  Successful: {summary['successful_test_files']}")
    print(f"  Failed: {summary['failed_test_files']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total duration: {summary['total_duration_seconds']:.2f} seconds")
    
    print(f"\nDependency Status:")
    for dep, available in report["dependencies"].items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    print(f"\nTest File Results:")
    for result in report["test_file_results"]:
        status = "✓" if result["success"] else "✗"
        duration = result["duration"]
        print(f"  {status} {result['test_file']} ({duration:.2f}s)")
        
        if not result["success"] and result["stderr"]:
            # Show first few lines of error
            error_lines = result["stderr"].split("\n")[:3]
            for line in error_lines:
                if line.strip():
                    print(f"      {line}")
    
    if report["recommendations"]:
        print(f"\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
    
    print("\n" + "="*60)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run comprehensive integration tests for RAG Fusion Factory")
    parser.add_argument("--include-real-engines", action="store_true", 
                       help="Include tests that require real search engines (Docker)")
    parser.add_argument("--include-performance", action="store_true",
                       help="Include performance benchmarking tests")
    parser.add_argument("--report-file", type=str,
                       help="Save detailed JSON report to specified file")
    parser.add_argument("--fast", action="store_true",
                       help="Run only fast tests (skip slow performance tests)")
    
    args = parser.parse_args()
    
    print("RAG Fusion Factory - Comprehensive Integration Test Runner")
    print("="*60)
    
    # Check dependencies
    print("Checking dependencies...")
    dependencies = check_dependencies()
    
    # Determine which tests to run
    test_files = [
        "tests/test_comprehensive_integration.py"
    ]
    
    if args.include_real_engines:
        test_files.append("tests/test_real_engine_integration.py")
        # Set environment variable to enable real engine tests
        os.environ['SKIP_REAL_ENGINE_TESTS'] = 'false'
    
    if args.include_performance:
        test_files.append("tests/test_performance_benchmarks.py")
    
    # Add pytest arguments based on options
    pytest_args = []
    if args.fast:
        pytest_args.extend(["-m", "not slow"])
    
    print(f"Running {len(test_files)} test files...")
    
    # Run tests
    test_results = []
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        result = run_pytest_with_capture(test_file, pytest_args)
        test_results.append(result)
        
        if result["success"]:
            print(f"  ✓ Completed successfully ({result['duration']:.2f}s)")
        else:
            print(f"  ✗ Failed ({result['duration']:.2f}s)")
    
    # Generate and display report
    report = generate_report(test_results, dependencies, args.report_file)
    print_summary(report)
    
    # Exit with appropriate code
    if report["summary"]["failed_test_files"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()