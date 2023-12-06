import os
import sys
from os.path import isfile, isdir
from subprocess import STDOUT, check_output, CalledProcessError, TimeoutExpired
import argparse
import tempfile

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'

run_tests = 0
success_tests = 0
failed_tests = 0
error_tests = 0
errors = []
mlir_assigner = "build/bin/mlir-assigner"
zkml_compiler = "build/bin/zkml-onnx-compiler"

def build_error_object(file, reason):
    return dict({
        'file': file,
        'reason': reason
    })

def test_onnx(file, subfolder_path, timeout, verbose):
    global run_tests 
    global success_tests
    global failed_tests
    global error_tests
    global errors 
    global zkml_compiler
    with tempfile.NamedTemporaryFile() as fp:
        args = [zkml_compiler, os.path.join(subfolder_path, file), "-i", fp.name]
    if verbose:
        print("running: '" + " ".join(args) + "'...", end="", flush=True)

        print(args)
        raise Exception("sadge")

def test_mlir(file, subfolder_path, timeout, verbose):
    global run_tests 
    global success_tests
    global failed_tests
    global error_tests
    global errors 
    global mlir_assigner
    # read output file
    output_file = os.path.join(subfolder_path, file.replace(".mlir", ".res"))
    if not isfile(output_file):
        print(f"{bcolors.FAIL} error{bcolors.ENDC}")
        errors.append(build_error_object(file, f"cannot find output file"))
        error_tests += 1
        return
    with open(output_file,mode='r') as f:
        should_output = f.read().strip()
    # Construct the JSON file name by replacing the ".mlir" extension with ".json"
    json_file = file.replace(".mlir", ".json")
    json_file_path = os.path.join(subfolder_path, json_file)
    # Call the assigner binary with the input files
    run_tests += 1
    args = [mlir_assigner, "-b" , os.path.join(subfolder_path, file), "-i", json_file_path, "-c", "circuit", "-t", "table", "-e", "pallas", "--print_circuit_output"]
    if verbose:
        print("running: '" + " ".join(args) + "'...", end="", flush=True)
    try:
        is_output = check_output(args, stderr=STDOUT, timeout=timeout).decode().strip()
        if is_output == should_output:
            print(f"{bcolors.OKGREEN} success{bcolors.ENDC}")
            success_tests += 1
        else: 
            failed_tests += 1
            print(f"{bcolors.FAIL} failed{bcolors.ENDC}")
            errors.append(build_error_object(file, f"output mismatch"))
    except CalledProcessError:
            error_tests += 1
            print(f"{bcolors.FAIL} error{bcolors.ENDC}")
            errors.append(build_error_object(file, f"unexpteced error from subprocess"))
    except TimeoutExpired:
            error_tests += 1
            print(f"{bcolors.FAIL} error{bcolors.ENDC}")
            errors.append(build_error_object(file, f"ran into timeout ({timeout}s)"))


def test_folder(test_suite, folder, timeout, verbose):
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
        subfolder_path = os.path.join(folder, subfolder)
        # Get a list of all files within the subfolder
        files = os.listdir(subfolder_path)
        files.sort()
        for file in files:
            if file.endswith(".onnx"):
                print(f"Testing {file}...", end="", flush=True) 
                test_onnx(file, subfolder_path, timeout, verbose)
            if file.endswith(".mlir"):
                print(f"Testing {file}...", end="", flush=True) 
                test_mlir(file, subfolder_path, timeout, verbose)



parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true', help='Run fast tests only')
parser.add_argument('--verbose', action='store_true', help='Print detailed output')

args = parser.parse_args()

if args.fast:
    slow_test = False
else:
    slow_test = True

test_folder("SingleOps", "mlir-assigner/tests/Test/", 30, args.verbose)
# Rest of your code...
# test_folder("SingleOps", "mlir-assigner/tests/Ops/", 30, args.verbose)
# if slow_test:
#     test_folder("Models", "mlir-assigner/tests/Models/", 500, args.verbose)

# cleanup
os.remove("circuit")
os.remove("table")
print("\n")
print(f"Test Report - run {run_tests} tests, {success_tests} success, {failed_tests} failed, {error_tests} errors")
for error in errors:
    print("\t" + error['file'] + ": \"" + error['reason'] + "\"")
    print("")

if len(errors) == 0:
    print("Test Suite complete!")
else:
    exit(-1)
