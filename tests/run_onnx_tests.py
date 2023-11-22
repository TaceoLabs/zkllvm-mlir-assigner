from os import listdir
from os.path import isfile, isdir, join
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


def build_error_object(file, reason):
    return dict({
        'file': file,
        'reason': reason
    })


path_to_onnx_tests = "tests/onnx/"
path_to_inputs = "tests/inputs/onnx/"

assigner_cmd = "build/bin/assigner/assigner -b {} -i {} -t assignment.tbl -c circuit.crct -e pallas --check --print_circuit_output"

if not isdir(path_to_onnx_tests) or not isdir(path_to_inputs):
    print("cannot find test definitions. Did you call it not from root or justfile?")
    exit(-1)

test_definitions = [f for f in listdir(path_to_onnx_tests) if isdir(join(path_to_onnx_tests, f))]
errors = []
tests = 0
sucess_tests = 0
failed_tests = 0
error_tests = 0
for onnx_test in test_definitions:
    # get input
    current_test = join(path_to_onnx_tests, onnx_test)
    should_output_path = join(current_test, "expected_output.txt")
    input_file = join(path_to_inputs, onnx_test, "in.json")
    print(f"################### Testing {onnx_test} ###################")
    if not isfile(input_file):
        error_tests += 1
        print(f"\t{bcolors.FAIL}ERROR{bcolors.ENDC}: cannot find input")
        errors.append(build_error_object(onnx_test, f'cannot find input file {input_file}'))
        continue

    if not isfile(should_output_path):
        error_tests += 1
        print(f"\t{bcolors.FAIL}ERROR{bcolors.ENDC}: cannot find expected output")
        errors.append(build_error_object(onnx_test, f'cannot find expected output file {should_output_path}'))
        continue

    # load exptected output
    should_output_file = open(should_output_path,mode='r')
    should_output = should_output_file.read().strip()
    should_output_file.close()

    test_ll_files = [f for f in listdir(current_test) if isfile(join(current_test, f)) and f.endswith(".ll")]
    for integration_test in test_ll_files:
        tests += 1
        print(f"\t {integration_test}: ", end ="")
        args = assigner_cmd.format(join(current_test, integration_test), input_file).split(" ")
        try:
            is_output = check_output(args, stderr=STDOUT, timeout=60).decode().strip()
            # maybe make parse the output and check for smaller then ETA or something...
            if is_output == should_output:
                print(f"{bcolors.OKGREEN} success{bcolors.ENDC}")
                sucess_tests += 1
            else:
                failed_tests += 1
                print(f"{bcolors.FAIL} FAILED{bcolors.ENDC}")
                errors.append(build_error_object(join(current_test, integration_test), f"""
\t\texptected : {should_output},
\t\tgot:        {is_output}
"""))
        except CalledProcessError:
                failed_tests += 1
                print(f"{bcolors.FAIL} ERROR{bcolors.ENDC}")
                errors.append(build_error_object(join(current_test, integration_test), f"unexpteced error from subprocess"))
        except TimeoutExpired:
                failed_tests += 1
                print(f"{bcolors.FAIL} ERROR{bcolors.ENDC}")
                errors.append(build_error_object(join(current_test, integration_test), f"ran into timeout (15s)"))

# print two lines for test report
print("\n")
print(f"Test Report - run {tests} tests, {sucess_tests} success, {failed_tests} failed, {error_tests} errors")
for error in errors:
    print("\t" + error['file'].replace('tests/onnx/', '') + ":")
    print("\t\t" + error['reason'])
    print("")

if len(errors) == 0:
    print("Test Suite complete!")
else:
    exit(-1)
