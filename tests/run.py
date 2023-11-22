import os
import sys
from os.path import isfile, isdir
from subprocess import STDOUT, check_output, CalledProcessError, TimeoutExpired

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

def build_error_object(file, reason):
    return dict({
        'file': file,
        'reason': reason
    })


def test_folder(test_suite, folder, timeout):
    
    global run_tests 
    global success_tests
    global failed_tests
    global error_tests
    global errors 
    assigner_binary = "build/src/mlir-assigner"

    # Get a list of all files and folders within the "ops" folder
    items = os.listdir(folder)

    # Iterate over the list and check if each item is a folder
    subfolders = []
    for item in items:
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            subfolders.append(item)

    print(f"################### Testing {test_suite} ###################")
    # Iterate over each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder, subfolder)
        # Get a list of all files within the subfolder
        files = os.listdir(subfolder_path)
        # Iterate over the files and grab those ending in ".mlir"
        for file in files:
            if file.endswith(".mlir"):
                # read output file
                print(f"Testing {file}...", end="", flush=True) 
                output_file = os.path.join(subfolder_path, file.replace(".mlir", ".res"))
                if not isfile(output_file):
                    print(f"{bcolors.FAIL} error{bcolors.ENDC}")
                    errors.append(build_error_object(file, f"cannot find output file"))
                    error_tests += 1;
                    continue
                with open(output_file,mode='r') as f:
                    should_output = f.read().strip()
                # Construct the JSON file name by replacing the ".mlir" extension with ".json"
                json_file = file.replace(".mlir", ".json")
                json_file_path = os.path.join(subfolder_path, json_file)
                # Call the assigner binary with the input files
                run_tests += 1
                args = [assigner_binary, "-b" , os.path.join(subfolder_path, file), "-i", json_file_path, "-c", "circuit", "-t", "table", "-e", "pallas", "--print_circuit_output"]
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


slow_test = len(sys.argv) != 2

test_folder("SingleOps", "tests/Ops/", 15)
if slow_test:
    test_folder("Models", "tests/Models/", 500)


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
