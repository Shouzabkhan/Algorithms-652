import numpy as np
import math

# Add your name and blazer id here
name = "SHOUZAB KHAN"
blazer_id = "SKHAN6"

# For all questions, please do not hard-code your algorithms for testing inputs.
# You code should work with any input with arbitrary sizes. If you hard-code your algorithms, 10 points
# will be deduced per question.

# Question 1 (20 points): Implement a dyanamic programming algorithm to solve the Rod Cutting Problem.
# Input:
# price_table: market prices of rod of different lengths
# rod_length: length of given rod
# Output:
# A series of rod lengths after cut that provide the maximum profit
# The sum of all lengths should equal to rod_length.
# Lengths in vector can be in any order.
# Example:
# Given price_table = [1, 5, 8, 9] and rod_length = 4.
# The algorithm should return a set of length inculding [2, 2]

def rod_cutting_problem(price_table, rod_length):
    total_length = rod_length
    max_profit = [0] * (total_length + 1) # Initialize the max profit array
    cuts = [-1] * (total_length + 1)  # Initialize an array to store the cuts made


    # Loop over all lengths of rod from 1 to the total rod length
    for length in range(1, total_length + 1):
        current_max = -math.inf # Start with a very negative value for maximum profit
        for cut in range(length):
            profit = price_table[cut] + max_profit[length - cut - 1]
            if current_max < profit:
                current_max = profit  # Update maximum profit
                cuts[length] = cut + 1  # Store the cut made
            max_profit[length] = current_max  # Store the maximum profit for the current length

    result = []
    while total_length > 0:
        result.append(cuts[total_length])  # Add the cut length to the result
        total_length -= cuts[total_length]  # Reduce the total length
    return result

# Question 2 (20 points): Implement a dyanamic programming algorithm to solve the Maximum Subarray Problem.
# Input:
# A: input array containing a series of integers
# Output:
# Array of intergers containing two intergers representing the start and end index of the maximum subarray
# Index starts from 0.
# Example:
# Given array A = [-23, 18, 20, -7, 12]
# The algorithm should return two indices [1, 4]

def maximum_subarray(A):
    value = 0
    m= 0  # Starting index of the maximum subarray
    n = 0  # Ending index of the maximum subarray
    i = 0
    ans = 0
    while(i<len(A)):
        value=value+A[i]  # Add current element to the value
        if(value>ans):
            ans = value  # Update maximum value
            n = i  # Update end index of the subarray
        elif(value<0):
            value = 0
            m = i + 1
        i=i+1
    return [m,n]  # Return the start and end indices of the maximum subarray



# Question 3 (20 points): Implement a dyanamic programming algorithm to solve the Matrix Chain Product Problem.
# Input:
# A: input array containing a series of matrix dimensions
# For N matrices, the total number of dimensions is N+1 and the total number of multiplications is N-1
# Output:
# Array of intergers containing N-1 intergers representing the order of the multiplications (order starts from 1)
# e.g., 1 presents the first multiplication and N-1, represents the last multiplication.
# Example:
# Given 3 matrics, the dimension array = [100, 2, 50, 6] representing matrix A1(100*2), A2(2*50), A3(50*6).
# The algorithm should return two intergers for the 2 matrix multiplications [2, 1] i.e., A1*(A2*A3)

def matrix_chain_product(matrix_dimensions):
    n = len(matrix_dimensions) - 1  # Calculate the number of matrices
    min_cost = [[0] * n for _ in range(n)]
    split_points = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1  # Calculate the end index of the chain
            min_cost[i][j] = math.inf  # Start with a large value for minimum cost
            for k in range(i, j):
                cost = (matrix_dimensions[i] * matrix_dimensions[k + 1] * matrix_dimensions[j + 1] +
                        min_cost[i][k] + min_cost[k + 1][j])
                if cost < min_cost[i][j]:
                    min_cost[i][j] = cost  # Update the minimum cost
                    split_points[i][j] = k  # Store the split point for the optimal solution

    order = []
    def construct_order(i, j):
        if i != j:
            construct_order(i, split_points[i][j])
            construct_order(split_points[i][j] + 1, j)
            order.append(split_points[i][j] + 1)
    construct_order(0, n - 1)  # Call the function to get the multiplication order
    return order

# Question 4 (20 points): Implement a dyanamic programming algorithm to solve the 0/1 Knapsack Problem.
# Input:
# Two input arrays containing a series of weights and profits
# An integer K representing the knapsack capacity.
# Output:
# 0/1 Array of indicating which items are selected (0: not selected; 1: selected)
# Example:
# Given 4 items with weight = [5, 4, 6, 3] and profit = [10, 40, 30, 50]
# The algorithm should return array = [0, 1, 0, 1] represeting the second and the last item are selected.

def zero_one_knapsack(weight, profit, K):
    n = len(weight)
    A = []  # DP table for storing optimal values
    p = 0
    while(p<(n + 1)):
        d = []
        q = 0
        while(q<(K + 1)):
            d.append(0)  # Fill DP table with zeros initially
            q += 1
        A.append(d)
        p=p+1
    p = 1
    while(p<=n):
      q = 1
      while(q<=K):
        if(weight[p - 1]<=q):
            A[p][q] = max(A[p - 1][q], profit[p - 1] + A[p - 1][q - weight[p - 1]])
        elif(weight[p-1]>q):
            A[p][q] = A[p - 1][q]  # Carry forward the previous optimal value
        q=q+1
      p=p+1
    profit_objects = [0] * len(weight)
    v1 = A[len(weight)][K]
    q = K
    p = len(weight)
    while(p > 0):
     if(v1 != A[p - 1][q]):  # Item is selected
        profit_objects[p - 1] = 1
        v1 = v1- profit[p - 1]
        q = q- weight[p - 1]
     p=p-1
    return profit_objects


# Question 5 (20 points): Implement the Radix sort algorithm. For sorting each digit you should use counting sort.
# Input:
# An input arrays containing a series of integers. Each interger has maximum K digits. The range of each digit is from 0 to 9.
# Note that some integers may have less than K digits.
# Output:
# sorted array
# Example:
# Given array = [15, 24, 26, 3] and K = 2
# The algorithm should return sorted array [3, 15, 24, 26].

def radix_sort(A, k):
    for exp in range(k):  # Loop over the digit positions starting from the least significant digit
        counting_sort(A, 10 ** exp)
    return A

def counting_sort(A, exp):
    output = [0] * len(A)
    count = [0] * 10

    for num in A:
        index = (num // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for num in reversed(A):
        index = (num // exp) % 10
        output[count[index] - 1] = num
        count[index] -= 1

    for i in range(len(A)):
        A[i] = output[i]

# Question 6 (20 points): Implement the activity selection algorithm using greedy appraoch.
# Input:
# Two input arrays containing a series of start time and finish time
# Output:
# 0/1 indicating which activities are selected (0: not selected; 1: selected)
# Example:
# Given N=4 items with start time = [1, 3, 0, 5, 8, 5] and finish time = [2, 4, 6, 7, 9, 9]
# The algorithm should return array =  [1, 1, 0, 1, 1, 0] represeting the 4 activites: (1, 2), (3, 4), (5, 7), (8, 9) are selected.

def activity_selection(start_time, finish_time):
    n = len(start_time)
    selected_activities = [0] * n
    activities = [(start, finish, i) for i in range(len(start_time)) for start, finish in [(start_time[i], finish_time[i])]]
    activities.sort(key=lambda x: x[1])

    selected_activities[activities[0][2]] = 1
    prev_finish_time = activities[0][1]
    i = 1
    while i < n:
        start, finish, index = activities[i]
        if start >= prev_finish_time:
            selected_activities[index] = 1 # Select activity if its start time is >= last finish time
            prev_finish_time = finish # Update the finish time of the last selected activity
        i += 1

    return selected_activities



########### DO NOT MODIFY CODE BELOW THIS LINE #############

def verify_rod_cutting(price_table, result, true_result):
    total_price = sum(price_table[i-1] for i in result)
    return true_result == total_price

def verify_max_subarray(A, result, true_result):
    total_sum = sum(A[i] for i in range(result[0], result[1] + 1))
    return true_result == total_sum

def verify_matrix_chain(result, true_result):
    for i in range(len(true_result)):
        if result[i] != true_result[i]:
            return False
    return True

def verify_zero_one_knapsack(weight, profit, result, true_result, K):
    total_weight = 0
    total_profit = 0

    for i in range(len(result)):
        if result[i]:
            total_weight += weight[i]
            total_profit += profit[i]

    return total_weight <= K and total_profit == true_result

def verify_radix_sort(result, true_result):
    for i in range(len(true_result)):
        if result[i] != true_result[i]:
            return False
    return True

def verify_activity_selection(result, true_result):
    for i in range(len(true_result)):
        if result[i] != true_result[i]:
            return False
    return True

print("Name: {} blazer_id: {}".format(name, blazer_id))

print("---------- Rod Cutting Problem ---------")
price_table_1 = [1, 5, 8, 9]
result1 = rod_cutting_problem(price_table_1, len(price_table_1))
pass1 = verify_rod_cutting(price_table_1, result1, 10)
print("Case 1: Correct:", "Yes" if pass1 else "No")

price_table_2 = [2, 3, 6, 9, 10, 17, 17]
result2 = rod_cutting_problem(price_table_2, len(price_table_2))
pass2 = verify_rod_cutting(price_table_2, result2, 19)
print("Case 2: Correct:", "Yes" if pass2 else "No")

print("------ Maximum Subarray Problem ------")
array_1 = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
result1 = maximum_subarray(array_1)
pass1 = verify_max_subarray(array_1, result1, 43)
print("Case 1: Correct:", "Yes" if pass1 else "No")

array_2 = [-2, -3, 4, -1, -2, 1, 5, -3]
result2 = maximum_subarray(array_2)
pass2 = verify_max_subarray(array_2, result2, 7)
print("Case 2: Correct:", "Yes" if pass2 else "No")


print("------ Matrix Chain Product ------")
matrix_dimensions_1 = [100, 2, 50, 6]
result1 = matrix_chain_product(matrix_dimensions_1)
pass1 = verify_matrix_chain(result1, [2, 1])
print("Case 1:", "Correct: " + ("Yes" if pass1 else "No"))

matrix_dimensions_2 = [100, 60, 70, 30, 100]
result2 = matrix_chain_product(matrix_dimensions_2)
pass2 = verify_matrix_chain(result2, [2, 1, 3])
print("Case 2:", "Correct: " + ("Yes" if pass2 else "No"))

print("------ 0/1 Knapsack Problem ------")
weight_1 = [10, 20, 30]
profit_1 = [60, 100, 120]
result1 = zero_one_knapsack(weight_1, profit_1, 50)
pass1 = verify_zero_one_knapsack(weight_1, profit_1, result1, 220, 50)
print(f"Case 1: Correct: {'Yes' if pass1 else 'No'}")

weight_2 = [5, 4, 6, 3]
profit_2 = [10, 40, 30, 50]
result2 = zero_one_knapsack(weight_2, profit_2, 10)
pass2 = verify_zero_one_knapsack(weight_2, profit_2, result2, 90, 10)
print(f"Case 2: Correct: {'Yes' if pass2 else 'No'}")

print("------ Radix Sort ------")
array_1 = [23, 12, 65, 89, 62, 38, 48]
result1 = radix_sort(array_1, 2)
pass1 = verify_radix_sort(result1, [12, 23, 38, 48, 62, 65, 89])
print(f"Case 1: Correct: {'Yes' if pass1 else 'No'}")

array_2 = [1047, 2992, 9473, 7362, 1938, 4845, 3838, 5693, 9128]
result2 = radix_sort(array_2, 4)
pass2 = verify_radix_sort(result2, [1047, 1938, 2992, 3838, 4845, 5693, 7362, 9128, 9473])
print(f"Case 2: Correct: {'Yes' if pass2 else 'No'}")

print("------ Activity Selection Problem ------")
start_1 = [1, 3, 0, 5, 8, 5]
finish_1 = [2, 4, 6, 7, 9, 9]
result1 = activity_selection(start_1, finish_1)
pass1 = verify_activity_selection(result1, [1, 1, 0, 1, 1, 0])
print(f"Case 1: Correct: {'Yes' if pass1 else 'No'}")

start_2 = [4, 3, 2, 5, 1, 3, 6, 1, 7, 4]
finish_2 = [6, 6, 4, 7, 2, 5, 7, 3, 9, 7]
result2 = activity_selection(start_2, finish_2)
pass2 = verify_activity_selection(result2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
print(f"Case 2: Correct: {'Yes' if pass2 else 'No'}")
