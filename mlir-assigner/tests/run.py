import os
import ast
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

MAX_DELTA = 0.001

run_tests = 0
success_tests = 0
failed_tests = 0
error_tests = 0
ignored_tests = 0
errors = []
mlir_assigner = "build/bin/mlir-assigner"
zkml_compiler = "build/bin/zkml-onnx-compiler"

def assert_output(should_output, is_output):
    global MAX_DELTA 
    is_lines = is_output.splitlines()
    if len(is_lines) < 3:
        return False, "Output incomplete got less than 3 lines"
    ziped = zip(should_output.splitlines(), is_lines)
    #1 First Line Result
    s,i = next(ziped)
    if s != i:
        return False, "Cannot get Result (First Line does not match) "
    s,i = next(ziped)
    memref_index = s.find('[')
    if s[0: memref_index] != i[0: memref_index]:
        return False, "Type mismatch (should={}; is={})".format(s[0: memref_index], i[0: memref_index])
    s_values = ast.literal_eval(s[memref_index:])
    i_values = ast.literal_eval(i[memref_index:])
    if any(filter(lambda a: a >= MAX_DELTA, map(lambda a: abs(a[0]-a[1]), zip(s_values, i_values)))):
        return False, "Values diverge more than > {}".format(MAX_DELTA)
    s,i = next(ziped)
    s = s.replace("rows", "").strip()
    i = i.replace("rows", "").strip()
    if s != i:
        return False, "Amount of rows mismatch (should={}; is={})".format(s, i)
    return True, ""

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
    mlir_file = os.path.join(subfolder_path, file.replace(".onnx", ".mlir"))
    args = [zkml_compiler, os.path.join(subfolder_path, file), "-i", mlir_file]
    if verbose:
        print("running: '" + " ".join(args) + "'...", flush=True)
    #todo remove check output
    check_output(args, stderr=STDOUT, timeout=timeout).decode().strip()
    # read output file
    output_file = os.path.join(subfolder_path, file.replace(".onnx", ".res"))
    if not isfile(output_file):
        print(f"{bcolors.FAIL} error{bcolors.ENDC}")
        errors.append(build_error_object(file, f"cannot find output file"))
        error_tests += 1
        return
    with open(output_file,mode='r') as f:
        should_output = f.read().strip()
    # Construct the JSON file name by replacing the ".mlir" extension with ".json"
    json_file = file.replace(".onnx", ".json")
    json_file_path = os.path.join(subfolder_path, json_file)
    # Call the assigner binary with the input files
    run_tests += 1
    args = [mlir_assigner, "-b" , mlir_file, "-i", json_file_path, "-c", "circuit", "-t", "table", "-e", "pallas", "--print_circuit_output"]
    if verbose:
        print("running: '" + " ".join(args) + "'...",  flush=True)
    try:
        valid, error_string = assert_output(should_output, check_output(args, stderr=STDOUT, timeout=timeout).decode().strip())
        if valid:
            print(f"{bcolors.OKGREEN} success{bcolors.ENDC}")
            success_tests += 1
        else: 
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
    finally:
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
        print("running: '" + " ".join(args) + "'...", flush=True)
    try:
        valid, error_string = assert_output(should_output, check_output(args, stderr=STDOUT, timeout=timeout).decode().strip())
        if valid:
            print(f"{bcolors.OKGREEN} success{bcolors.ENDC}")
            success_tests += 1
        else: 
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
                print(f"Testing {file}...", end="",flush=True) 
                if file in ignore_tests:
                    ignored_tests += 1
                    print(f"{bcolors.OKCYAN} ignored {bcolors.ENDC}")
                else:
                    test_onnx(file, subfolder_path, timeout, verbose, keep_mlir)
            if file.endswith(".mlir") and mlir_tests:
                print(f"Testing {file}...", end="", flush=True) 
                if file in ignore_tests:
                    ignored_tests += 1
                    print(f"{bcolors.OKCYAN} ignored {bcolors.ENDC}")
                else:
                    test_mlir(file, subfolder_path, timeout, verbose)



parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true', help='Run fast tests only')
parser.add_argument('--verbose', action='store_true', help='Print detailed output')
parser.add_argument('--keep-mlir', action='store_true', help='Keep generated mlir files')
parser.add_argument('--current', action='store_true', help='do only the current folder')

args = parser.parse_args()

if args.fast:
    slow_test = False
else:
    slow_test = True

if args.current:
    test_folder("SingleOps E2E", "mlir-assigner/tests/Ops/Current", False, 30, args.verbose, args.keep_mlir)
else:
    test_folder("SingleOps E2E", "mlir-assigner/tests/Ops/Onnx", False, 30, args.verbose, args.keep_mlir)
    test_folder("SingleOps special MLIR", "mlir-assigner/tests/Ops/Mlir", True, 30, args.verbose, args.keep_mlir)
    if slow_test:
        test_folder("Models", "mlir-assigner/tests/Models/", False, 500, args.verbose, args.keep_mlir)

# cleanup
if isfile("circuit"):
    os.remove("circuit")
if isfile("table"):
    os.remove("table")
print("\n")
print(f"Test Report - run {run_tests} tests, {success_tests} success, {failed_tests} failed, {error_tests} errors, {ignored_tests} ignored")
for error in errors:
    print("\t" + error['file'] + ": \"" + error['reason'] + "\"")
    print("")

if len(errors) == 0:
    print("Test Suite complete!")
else:
    exit(-1)
