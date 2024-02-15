import os
import ast
import sys
from os.path import isfile, isdir
from subprocess import DEVNULL, check_output, CalledProcessError, TimeoutExpired 
import subprocess
import argparse
import tempfile
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'

MAX_DELTA = 0.01

run_tests = 0
success_tests = 0
failed_tests = 0
error_tests = 0
ignored_tests = 0
errors = []
mlir_assigner = "build/bin/mlir-assigner"
zkml_compiler = "build/bin/zkml-onnx-compiler"
fixed_sizes = ["16.16", "16.32", "32.16", "32.32"]
file_filter = []


def build_error_object(file, reason):
    return dict({
        'file': file,
        'reason': reason
    })

def test_onnx(file, subfolder_path, timeout, verbose, keep_mlir):
    global run_tests
    global success_tests
    global failed_tests
    global error_tests
    global errors
    global zkml_compiler
    global fixed_sizes
    mlir_file = os.path.join(subfolder_path, file.replace(".onnx", ".mlir"))
    args = [zkml_compiler, os.path.join(subfolder_path, file), "--output", mlir_file, "-zk", "ALL_PUBLIC"]
    if verbose:
        print("running: '" + " ".join(args) + "'...", flush=True)
    #todo remove check output
    subprocess.run(args, stderr=DEVNULL, timeout=timeout).check_returncode()
    for fixed_size in fixed_sizes:
        print(f"Testing {file} {fixed_size}...", end="",flush=True)
        # Construct the JSON file name by replacing the ".mlir" extension with ".json"
        json_file = file.replace(".onnx", ".json")
        json_output_file = file.replace(".onnx", f"{fixed_size}.output.json")
        json_file_path = os.path.join(subfolder_path, json_file)
        json_output_file_path = os.path.join(subfolder_path, json_output_file)
        # Call the assigner binary with the input files
        run_tests += 1
        args_full = [mlir_assigner, "-b" , mlir_file, "-i", json_file_path, "-o", "something", "-c", "circuit", "-t", "table", "-e", "pallas", "-f", "dec", "--check", "-x", fixed_size]
        args_only_circuit = [mlir_assigner, "-b" , mlir_file, "-i", json_file_path, "-o", "somethindifferent", "-c", "ocircuit", "-e", "pallas", "-f", "dec", "--check", "-x", fixed_size, "--generate-type", "circuit"]
        args_diff = ["diff", "circuit" , "ocircuit"]
        if verbose:
            print("running: '" + " ".join(args_full) + "'...",  flush=True)
        try:
            subprocess.run(args_full, stdout=DEVNULL, stderr=DEVNULL, timeout=timeout).check_returncode()
            if verbose:
                print("running: '" + " ".join(args_only_circuit) + "'...",  flush=True)
            subprocess.run(args_only_circuit, stdout=DEVNULL, stderr=DEVNULL, timeout=timeout).check_returncode()
            if verbose:
                print("running: '" + " ".join(args_diff) + "'...",  flush=True)
            diff_result = subprocess.run(args_diff, stderr=DEVNULL, timeout=timeout)
            if diff_result.returncode != 0:
                error_tests += 1
                print(f"{bcolors.FAIL} failed{bcolors.ENDC}")
                errors.append(build_error_object(file, "diff in circuits"))
            else:
                print(f"{bcolors.OKGREEN} success{bcolors.ENDC}")
                success_tests += 1
        except CalledProcessError:
                error_tests += 1
                print(f"{bcolors.FAIL} error{bcolors.ENDC}")
                errors.append(build_error_object(file, f"unexpteced error from subprocess"))
        except TimeoutExpired:
                error_tests += 1
                print(f"{bcolors.FAIL} error{bcolors.ENDC}")
                errors.append(build_error_object(file, f"ran into timeout ({timeout}s)"))
        finally:
            if isfile("circuit"):
                os.remove("circuit")
            if isfile("ocircuit"):
                os.remove("ocircuit")
            if isfile("something"):
                os.remove("something")
            if isfile("somethindifferent"):
                os.remove("somethindifferent")
    if not keep_mlir and isfile(mlir_file):
        if verbose:
            print("removing {}".format(mlir_file))
        os.remove(mlir_file)

def test_mlir(file, subfolder_path,  timeout, verbose):
    global run_tests
    global success_tests
    global failed_tests
    global error_tests
    global errors
    global mlir_assigner

    for fixed_size in fixed_sizes:
        print(f"Testing {file} {fixed_size}...", end="",flush=True)
        # Construct the JSON file name by replacing the ".mlir" extension with ".json"
        json_file = file.replace(".mlir", ".json")
        json_output_file = file.replace(".mlir", f"{fixed_size}.output.json")
        json_file_path = os.path.join(subfolder_path, json_file)
        json_output_file_path = os.path.join(subfolder_path, json_output_file)
        # Call the assigner binary with the input files
        run_tests += 1
        args = [mlir_assigner, "-b" , os.path.join(subfolder_path, file), "-i", json_file_path, "-o", json_output_file_path, "-c", "circuit", "-t", "table", "-e", "pallas", "-f", "dec", "--check", "-x", fixed_size]
        if verbose:
            print("running: '" + " ".join(args) + "'...", flush=True)
        try:
            result = subprocess.run(args, stderr=DEVNULL, timeout=timeout)
            if not result.returncode == 0:
                failed_tests += 1
                print(f"{bcolors.FAIL} failed{bcolors.ENDC}")
                errors.append(build_error_object(file, error_string))
        except CalledProcessError:
                error_tests += 1
                print(f"{bcolors.FAIL} error{bcolors.ENDC}")
                errors.append(build_error_object(file, f"unexpteced error from subprocess"))
        except TimeoutExpired:
                error_tests += 1
                print(f"{bcolors.FAIL} error{bcolors.ENDC}")
                errors.append(build_error_object(file, f"ran into timeout ({timeout}s)"))


def test_folder(test_suite, folder, mlir_tests, timeout, verbose, keep_mlir):
    global ignored_tests
    # Get a list of all files and folders within the "ops" folder
    items = os.listdir(folder)

    # Iterate over the list and check if each item is a folder
    subfolders = []
    for item in items:
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
    subfolders.sort()
    print(f"################### Testing {test_suite} ###################")
    # Iterate over each subfolder
    for subfolder in subfolders:
        if len(file_filter) > 0:
            if not any(filter in subfolder.upper() for filter in file_filter):
                continue
        subfolder_path = os.path.join(folder, subfolder)
        # Get a list of all files within the subfolder
        files = os.listdir(subfolder_path)

        files.sort()
        ignore_tests = []
        if ".ignore" in files:
            with open(os.path.join(subfolder_path, ".ignore")) as ignore_file:
                ignore_tests = ignore_file.read()
                if ignore_tests == "":
                    ignore_tests = files
                else:
                    ignore_tests = list(ignore_tests.splitlines())

        for file in files:
            if file.endswith(".onnx") and not mlir_tests:
                if file in ignore_tests:
                    ignored_tests += 1
                    print(f"Testing {file}...", end="",flush=True)
                    print(f"{bcolors.OKCYAN} ignored {bcolors.ENDC}")
                else:
                    test_onnx(file, subfolder_path, timeout, verbose, keep_mlir)
            if file.endswith(".mlir") and mlir_tests:
                if file in ignore_tests:
                    ignored_tests += 1
                    print(f"Testing {file}...", end="",flush=True)
                    print(f"{bcolors.OKCYAN} ignored {bcolors.ENDC}")
                else:
                    test_mlir(file, subfolder_path, timeout, verbose)



parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true', help='Run fast tests only')
parser.add_argument('--verbose', action='store_true', help='Print detailed output')
parser.add_argument('--keep-mlir', action='store_true', help='Keep generated mlir files')
parser.add_argument('--models', action='store_true', help='Test only the models')
parser.add_argument('--current', action='store_true', help='do only the current folder')
parser.add_argument('--filter', nargs='?', help='only tests that have filter in name')

args = parser.parse_args()
if args.filter:
    file_filter = list(map(lambda x: x.upper(), list(args.filter.split(','))))

start = time.time()
if args.current:
    if args.fast:
        fixed_sizes = ["16.16"]
    test_folder("SingleOps E2E", "mlir-assigner/tests/Ops/Current", False, 30, args.verbose, args.keep_mlir)
else:
    test_models = file_filter != [] or not args.fast or args.models
    if args.fast:
        fixed_sizes = ["16.16"]
    if test_models:
        test_folder("Models", "mlir-assigner/tests/Models/", False, 500, args.verbose, args.keep_mlir)
    if not args.models:
        test_folder("SingleOps E2E", "mlir-assigner/tests/Ops/Onnx", False, 30, args.verbose, args.keep_mlir)
        # test_folder("SingleOps special MLIR", "mlir-assigner/tests/Ops/Mlir", True, 30, args.verbose, args.keep_mlir)
end = time.time()
_, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)

# cleanup
if isfile("circuit"):
    os.remove("circuit")
if isfile("table"):
    os.remove("table")
print("\n")
print(f"Test Report - run {run_tests} tests, {success_tests} success, {failed_tests} failed, {error_tests} errors, {ignored_tests} ignored")
print(f"Test Report - took {minutes} min, {seconds:.3} s")
for error in errors:
    print("\t" + error['file'] + ": \"" + error['reason'] + "\"")
    print("")

if len(errors) == 0:
    print("Test Suite complete!")
else:
    exit(-1)