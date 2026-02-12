"""
This file is just to generate the stings that get copyied and pasted into COMSOL
See the comments below to see the params used to make each grid type
"""


def print_numbers_separated_by_distance(x, n):
    # Generate the numbers centered at zero
    numbers = [i * x for i in range(-n // 2, n // 2 + 1)]

    # round each number to the nearest 8 decimal places
    numbers = [round(i, 8) for i in numbers]

    # Convert numbers to strings and join them with commas
    result = ", ".join(map(str, numbers))

    # Print the result
    print(result)


# Example usage
# print_numbers_separated_by_distance(0.000125, 640)

# Grid 1
# X --> print_numbers_separated_by_distance(0.01, 80)
# Y --> print_numbers_separated_by_distance(0.0005, 160)
# Z --> print_numbers_separated_by_distance(0.0005, 160)

# Grid 2
# X --> print_numbers_separated_by_distance(0.005, 160)
# Y --> print_numbers_separated_by_distance(0.000125, 640)
# Z --> print_numbers_separated_by_distance(0.000125, 640)


# 2D Grid1 in meters... gotta check with rest of code
#print_numbers_separated_by_distance(2 * 10 ** (-6), 150)  # x
#print_numbers_separated_by_distance(2 * 10 ** (-6), 120)  # y
#print_numbers_separated_by_distance(2 * 10 ** (-6), 120)  # z

