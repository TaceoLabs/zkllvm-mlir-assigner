import sys

print('start filter.py')
all_input = sys.stdin.read()
std_in_len = len(all_input)
idx = all_input.find('error:')
error_counter = 0
while idx != -1:
    error_counter += 1
    warning_index = min(all_input.find('warning:', idx), std_in_len) 
    error_string = all_input[idx:warning_index]
    if error_string.startswith('error: Recipe `build`'):
        break
    print("=== Error {} ===".format(error_counter))
    print(error_string)
    idx = all_input.find('error:', warning_index)
    print("\n")

if error_counter == 0:
    print("No errors found!")
