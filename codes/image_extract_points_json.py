import numpy as np
import json

# Function to extract points from a JSON file
def extract_points_from_json(filename):
    # Open the JSON file
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Extracting points from the JSON data
    points = []
    for shape in data["shapes"]:
        points.append(shape["points"][0])  # Extracting the first set of coordinates for each shape

    # Converting the inner list of single coordinate point to python tuple()
    points = inner_list_to_tuple(points)
    
    return points

def inner_list_to_tuple(points):
    points_new = []
    for point in points:
        #print(f"Point: {point}")
        
        point = float_to_integer_dtype(point) 
        my_tuple = tuple(point)
        points_new.append(my_tuple)

    return points_new
    
def print_points(points):
    for point in points:
      print(f"Point: {point}")
      
      
def float_to_integer(point_list):
    #int_list = [int(x) for x in point_list]
    rounded_list = [round(num) for num in point_list]
    
    return rounded_list

def float_to_integer_dtype(point_list):
    rounded_list = [np.int32(int(num) + 1 if num % 1 >= 0.5 else int(num)) for num in point_list]

    return rounded_list


'''
# Specify your file name here
filename = "dataset/DJI_0075_1.json"

# Extract points and print them
points = extract_points_from_json(filename)
#points = inner_list_to_tuple(points)

#print("All points: ", points)
print_points(points)
'''

