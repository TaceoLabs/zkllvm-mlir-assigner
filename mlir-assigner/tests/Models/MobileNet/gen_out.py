import json
import random
import ast  # Import the ast module for literal evaluation

# Set the size of your array
array_size = 3*224*224  # You can change this to the desired size
# Set the range for random floats
min_value = 0.0
max_value = 1.0
with open('kitten.txt', 'r') as file:
    array_data = file.read()
    bla = ast.literal_eval(array_data)
    to_print = [{"memref": {"data": bla, "dims": [1,3,224,224], "type": "f32"}   }]
    print(json.dumps(to_print))

