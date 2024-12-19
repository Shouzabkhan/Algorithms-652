import numpy as np
import random
import time
import sys

# **************************IMPORTANT**************************
# The following code includes solutions for the maximum subarray
# problem and various sorting algorithms. Each function completes
# the task without altering the function signature.
#
# For Questions 1.1 - 1.2:
# Implement two different algorithms to solve the maximum subarray
# problem. The result should be the sum of the maximum subarray,
# not the subarray itself.
# Example: For input array {13, -3, -25, 20}, return 20.
#
# For Questions 2.1 - 2.4:
# Implement four sorting algorithms that sort an array in ascending
# order.
# Example: For input array {8, 5, -10, 23}, return {-10, 5, 8, 23}.
#
# Do not modify the function interfaces. You can create helper
# functions or auxiliary structures like numpy arrays if needed.
# All inputs are 32-bit integers, and array sizes can vary widely.

# Add your name and Blazer ID here
name = "Shouzab Khan"
blazer_id = "Skhan6"

# Question 1.1 (10 points): Implement an improved brute-force approach
# for solving the maximum subarray problem using prefix sums.
def maximum_subarray_brute_force(A):
    max_sum = -np.inf
    curr_sum = 0
    index = 0
    while index < len(A):
        curr_sum += A[index]
        max_sum = max(curr_sum, max_sum)
        if curr_sum < 0:
            curr_sum = 0
        index += 1
    return max_sum

# Question 1.2 (20 points): Implement the divide-and-conquer approach
# for solving the maximum subarray problem.
def maximum_subarray_divide_and_conquer(A):
    def find_crossing_max(A, lo, mid, up):
        left_sum = -np.inf
        total = 0
        for i in range(mid, lo - 1, -1):
            total += A[i]
            if total > left_sum:
                left_sum = total

        right_sum = -np.inf
        total = 0
        for j in range(mid + 1, up + 1):
            total += A[j]
            if total > right_sum:
                right_sum = total

        return left_sum + right_sum

    def recursive_max_subarray(A, lo, up):
        if lo == up:
            return A[lo]
        else:
            mid = (lo + up) // 2
            left_max = recursive_max_subarray(A, lo, mid)
            right_max = recursive_max_subarray(A, mid + 1, up)
            cross_max = find_crossing_max(A, lo, mid, up)
            return max(left_max, right_max, cross_max)

    return recursive_max_subarray(A, 0, len(A) - 1)

# Question 2.1 (10 points): Implement the insertion sort algorithm
# to sort the array in ascending order.
def insertion_sort(A):
    for i in range(1, len(A)):
        key = A[i]
        j = i - 1
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key
    return A

# Question 2.2 (20 points): Implement the merge sort algorithm to
# sort the array in ascending order.
def merge_sort(A):
    def merge(A, lo, mid, up):
        left_half = A[lo:mid + 1]
        right_half = A[mid + 1:up + 1]

        i = j = 0
        k = lo

        while i < len(left_half) and j < len(right_half):
            if left_half[i] <= right_half[j]:
                A[k] = left_half[i]
                i += 1
            else:
                A[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            A[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            A[k] = right_half[j]
            j += 1
            k += 1

    def recursive_sort(A, lo, up):
        if lo < up:
            mid = (lo + up) // 2
            recursive_sort(A, lo, mid)
            recursive_sort(A, mid + 1, up)
            merge(A, lo, mid, up)

    recursive_sort(A, 0, len(A) - 1)
    return A

# Question 2.3 (20 points): Implement quick sort where the pivot is
# always the last element.
import random

def quick_sort(A):
    def partition(A, lo, up):
        pivot_index = random.randint(lo, up)  # Randomly select pivot
        A[pivot_index], A[up] = A[up], A[pivot_index]  # Swap with the last element
        pivot = A[up]
        i = lo - 1
        for j in range(lo, up):
            if A[j] <= pivot:
                i += 1
                A[i], A[j] = A[j], A[i]
        A[i + 1], A[up] = A[up], A[i + 1]
        return i + 1

    def recursive_sort(A, lo, up):
        while lo < up:  # Convert recursive calls to a loop to limit recursion depth
            pivot_index = partition(A, lo, up)
            if pivot_index - lo < up - pivot_index:
                recursive_sort(A, lo, pivot_index - 1)
                lo = pivot_index + 1  # Tail recursion optimization
            else:
                recursive_sort(A, pivot_index + 1, up)
                up = pivot_index - 1

    recursive_sort(A, 0, len(A) - 1)
    return A

# Question 2.4 (10 points): Implement randomized quick sort where the
# pivot is selected randomly.
def randomized_quick_sort(A):
    def partition(A, lo, up):
        pivot_index = random.randint(lo, up)
        A[pivot_index], A[up] = A[up], A[pivot_index]
        return quick_sort(A)

    return quick_sort(A)

# Question 3.1 (2 points): Why is divide and conquer faster than brute
# force for the maximum subarray problem?
# Answer: Divide and conquer splits the problem into smaller subproblems, solving them in O(n log n). Brute force, on the other hand, checks every possible subarray, leading to O(n^2) time complexity.

# Question 3.2 (2 points): What inputs lead to best/worst cases for
# insertion sort?
# Answer: Best case is when the array is already sorted, leading to O(n) time. Worst case occurs when the array is sorted in reverse order, with O(n^2). The average case is also O(n^2).

# Question 3.3 (2 points): Does merge sort's performance change with
# different inputs?
# Answer: No, merge sort maintains O(n log n) complexity regardless of input because it splits and merges the array consistently.

# Question 3.4 (2 points): When does quick sort perform poorly, and how
# does randomized quick sort help?
# Answer: Quick sort performs worst when consistently choosing the smallest or largest element as the pivot. Randomized quick sort reduces the likelihood of consistently poor pivots, improving performance.

# Question 3.5 (2 points): Does randomized quick sort do extra work for
# an array of size 1?
# Answer: No, both deterministic and randomized quick sort handle arrays of size 1 similarly, performing minimal work.


########### DO NOT MODIFY CODE BELOW THIS LINE #############

def gen_random(n):
  data = np.empty([n], dtype=np.int32)
  for i in range(n):
    data[i] = random.randrange(0, 100)
  return data

def gen_ascending(n):
  data = np.empty([n], dtype=np.int32)
  for i in range(n):
    data[i] = i
  return data

def gen_descending(n):
  data = np.empty([n], dtype=np.int32)
  for i in range(n):
    data[i] = n - i - 1
  return data

def verify_max_subarray(A, result):
  true_result = -sys.maxsize - 1
  for start in range(A.shape[0]):
    for end in range(A.shape[0]):
      sum = 0
      for i in range(start, end + 1):
        sum += A[i]
      true_result = max(sum, true_result)
  if (true_result == result):
    return 'Yes'
  else:
    return 'No'

def verify_sort(A):
  correct = 'Yes'
  for i in range(A.shape[0] - 1):
    if (A[i] > A[i + 1]):
      correct = 'No'
      break
  return correct



print("Name: {} blazer_id: {}".format(name, blazer_id))
print("---------- Maximum Subarray Problem - Brute Force ---------")
for n in [int(1e1), int(1e2), int(1e3)]:
  A = gen_random(int(n))
  start = time.time()
  result = maximum_subarray_brute_force(A)
  stop = time.time()
  duration = stop - start
  correct = verify_max_subarray(A, result)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("------ Maximum Subarray Problem - Divide and Conquer ------")
for n in [int(1e1), int(1e2), int(1e3)]:
  A = gen_random(n)
  start = time.time()
  result = maximum_subarray_divide_and_conquer(A)
  stop = time.time()
  duration = stop - start
  correct = verify_max_subarray(A, result)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("---------------------- Insertion Sort ---------------------")

print("Input: already sorted in ascending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_ascending(n)
  start = time.time()
  A_sorted = insertion_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: already sorted in descending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_descending(n)
  start = time.time()
  A_sorted = insertion_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: random")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_random(n)
  start = time.time()
  A_sorted = insertion_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("------------------------- Merge Sort ----------------------")
print("Input: already sorted in ascending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_ascending(n)
  start = time.time()
  A_sorted = merge_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: already sorted in descending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_descending(n)
  start = time.time()
  A_sorted = merge_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: random")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_random(n)
  start = time.time()
  A_sorted = merge_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("------------------------ Quick Sort -----------------------")
print("Input: already sorted in ascending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_ascending(n)
  start = time.time()
  A_sorted = quick_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: already sorted in descending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_descending(n)
  start = time.time()
  A_sorted = quick_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: random")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_random(n)
  start = time.time()
  A_sorted = quick_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("------------------ Randomized Quick Sort ------------------")
print("Input: already sorted in ascending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_ascending(n)
  start = time.time()
  A_sorted = randomized_quick_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: already sorted in descending order")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_descending(n)
  start = time.time()
  A_sorted = randomized_quick_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))


print("Input: random")
for n in [int(1e2), int(1e3), int(1e4)]:
  A = gen_random(n)
  start = time.time()
  A_sorted = randomized_quick_sort(A)
  stop = time.time()
  duration = stop - start
  correct = verify_sort(A_sorted)
  print("n = {:6} Correct: {}\tRunning Time: {:6f}\tseconds".format(n, correct, duration))
