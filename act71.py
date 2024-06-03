import multiprocessing
from multiprocessing import Process, Pool, Array

import pandas as pd

import heapq

from functools import partial
from itertools import product

from ctypes import c_int64
from typing import Callable
from typing import Optional
from typing import Tuple


def binary_modular_exponentiation(b: int, x: int, m: int):
    """
    Performs binary modular exponentiation.

    Args:
        b (int): The base.
        x (int): The exponent.
        m (int): The modulus.

    Returns:
        int: The result of (b^x) % m.
    """
    # Convert the exponent to binary representation
    binary_exponent = bin(x)[2:]

    # Initialize the result to 1
    result = 1

    # Iterate through each bit of the binary exponent
    for bit in binary_exponent:
        # Square the result and take the modulus
        result = (result * result) % m

        # If the current bit is 1, multiply the result by the base and take the modulus
        if bit == '1':
            result = (result * b) % m

    # Return the final result
    return result


def modexp(b: int, x: int, m: int, ncores: int) -> int:
    """
    Calculates the modular exponentiation of a number.

    Args:
        b (int): The base number.
        x (int): The exponent.
        m (int): The modulus.
        ncores (int): The number of cores to use for parallel processing.

    Returns:
        int: The result of the modular exponentiation.

    """
    # Divide the exponent by the number of cores to determine the workload for each core
    exp = x // ncores
    remainder = x % ncores

    # Distribute the remaining workload evenly among the cores
    workloads = [exp + 1 if i < remainder else exp for i in range(ncores)]

    # Remove any zero-length workloads
    workloads = [workload for workload in workloads if workload != 0]

    # Get the number of workloads
    work_len = len(workloads)

    # Use multiprocessing to perform parallel processing
    with Pool(work_len) as p:
        # Use starmap to apply the binary_modular_exponentiation function to each workload
        results = p.starmap(binary_modular_exponentiation, [
                            (b, workload, m) for workload in workloads])

    # Calculate the final result by multiplying all the intermediate results and taking the modulus
    result = 1
    for r in results:
        result = (result * r) % m

    return result


def discrete_log(b: int, m: int, y: int, work, results, index: int) -> int:
    """
    Calculates the discrete logarithm of a number 'y' with base 'b' modulo 'm'.

    Args:
        b (int): The base of the logarithm.
        m (int): The modulus.
        y (int): The number for which the logarithm is calculated.
        work (list): A list containing the range of values to check.
        results (list): A list to store the results.
        index (int): The index at which to store the result in the 'results' list.

    Returns:
        int: The calculated discrete logarithm. If no solution is found, -1 is returned.
    """
    # Iterate over the range of values to check
    for x in range(work[0], work[1]):
        # Check if b^x mod m is equal to y
        if pow(b, x, m) == y:
            # Store the result in the 'results' list at the specified index
            results[index] = x
    # If no solution is found, set the result to -1
    results[index] = -1


def modlog(b: int, m: int, y: int, ncores: int) -> int:
    """
    Calculate the discrete logarithm of a number 'b' modulo 'm' using multiple processes.

    Args:
        b (int): The base of the logarithm.
        m (int): The modulus.
        y (int): The value for which the logarithm is calculated.
        ncores (int): The number of cores to use for parallel processing.

    Returns:
        int: The discrete logarithm of 'y' modulo 'm', or -1 if no solution is found.
    """

    # Calculate the workload for each process
    n = m // ncores

    # Divide the workload into ranges for each process
    workloads = [(i*n, (i+1)*n) for i in range(ncores)]

    # Remove any empty workloads
    workloads = [workload for workload in workloads if workload[0]
                 != 0 or workload[1] != 0]

    # Adjust the last workload if the modulus is not divisible by the number of cores
    if n % ncores != 0:
        workloads[-1] = (workloads[-1][0], m)

    # Create an array to store the results from each process
    results = Array(c_int64, len(workloads))

    # Create a process for each workload
    processes = [Process(target=discrete_log, args=(
        b, m, y, work, results, index)) for index, work in enumerate(workloads)]

    # Start all the processes
    [process.start() for process in processes]

    # Wait for all the processes to finish
    [process.join() for process in processes]

    # Check the results for a valid solution
    for result in results:
        if result != -1:
            return result

    # Return -1 if no solution is found
    return -1


def delta_read(tm: pd.DataFrame) -> Callable[[str, str], Tuple[str, str, str]]:
    # Transitions
    def delta(q: str, w: str) -> Tuple[str, str, str]:
        # Verify if the element (q, w) in QXS, otherwise, throw an error
        try:
            values = tm.loc[q, w]  # get transition as 'state,write,move'
            q, write, move = values.split(',')  # separte values by comma

            return q, write, move

        except Exception:
            raise ValueError('Transition ({}, {}) not defined'.format(q,
                                                                      w)) from None

    return delta


def process_string(M: str,
                   w: str,
                   iters: Optional[int] = None,
                   debug: Optional[bool] = False) -> Tuple[bool, str]:
    # Read transition table
    read_tm = pd.read_csv(M, index_col=0)

    # Convert index to string
    read_tm.index = read_tm.index.astype(str)

    # Essential elements of automaton M from transition table
    Q = read_tm.index.to_list()
    q0 = Q[0]
    F = [Q[-1]]
    B = read_tm.columns[0]
    delta = delta_read(read_tm)

    current_state = q0  # Initialize state
    n = 0  # Initial read position
    while current_state not in F:
        # Print state and current word if debug enabled
        if debug:
            print(f'q{current_state} :  {w[:n]}\033[4m{
                  w[n]}\033[0m{w[n + 1:]}')

        word_len = len(w)  # Current length
        symbol = w[n]  # Read symbol

        # Transition from current state and input symbol read
        try:
            current_state, write, move = delta(current_state, symbol)
        except Exception:
            # if no transition defined: halt
            # print(Exception)
            break

        # Update the tape (word)
        w = w[:n] + write + w[n + 1:]

        # Move head Right or Left of TM according to delta
        if move == 'R':
            n += 1
            if n == word_len:  # Add blank if it reaches end of world *word
                w += B
        elif move == 'L':
            n -= 1
            if n < 0:  # Add blank if it tries to go beyond the start
                w = B + w
                n = 0
        # Like a Two-way Infinite Tape

        # Check iterations left to read more symbols,
        # if not defined read forever, else check if not 0
        if iters is not None and not (iters := iters - 1):
            # if max iterations reached: halt
            # print("Halted after reaching maximum computations")
            break

    # At completion, determine if a final state has been reached or not beacuse of errors
    is_final = current_state in F

    return is_final, w


def check_tm_single(M: str, combination: tuple) -> None:
    """
    Checks if a given combination is accepted by a Turing machine.

    Args:
        M (str): The Turing machine.
        combination (tuple): The combination to be checked.

    Returns:
        str: The accepted combination if it is accepted by the Turing machine, None otherwise.
    """
    # Convert the combination tuple to a string
    word = ''.join(combination)

    # Process the string using the Turing machine
    if process_string(M, word)[0]:
        return word
    return None


def parallel_enumerate(M: str, n: Optional[int] = 50) -> None:
    """
    Enumerate all possible combinations of binary strings up to length n and
    check each combination using the `check_tm_single` function in parallel.

    Args:
        M (str): The input string.
        n (int, optional): The maximum length of binary strings to consider. Defaults to 50.

    Returns:
        None
    """
    # Set the default value of n to 50 if not provided
    n = n or 50

    # Get the number of cores in the machine
    ncores = multiprocessing.cpu_count()

    # Create a pool of worker processes
    with Pool(ncores) as p:
        # Iterate over different lengths of binary strings
        for currN in range(1, n + 1):
            # Generate all possible combinations of binary strings of length currN
            listCombinations = product(("0", "1"), repeat=currN)

            # Check each combination using the check_tm_single function in parallel
            results = p.starmap(
                check_tm_single, [(M, combination) for combination in listCombinations])

            # Print the valid words
            for word in results:
                if word is not None:
                    print(word)


def insertion_sort(A: list) -> list:
    """
    Sorts a list of integers using the insertion sort algorithm.

    Parameters:
    A (list): The list of integers to be sorted.

    Returns:
    list: The sorted list of integers.
    """
    # Iterate over the list starting from the second element
    for i in range(1, len(A)):
        key = A[i]  # Current element to be inserted at the right position
        j = i - 1  # Index of the previous element

        # Move elements of A[0..i-1], that are greater than key, to one position ahead of their current position
        while j >= 0 and A[j] > key:
            A[j + 1] = A[j]  # Shift the element to the right
            j -= 1

        A[j + 1] = key  # Insert the key at its correct position in the sorted subarray

    return A


def merge(left: list, right: list) -> list:
    """
    Merge two sorted lists into a single sorted list.

    Args:
        left (list): The first sorted list.
        right (list): The second sorted list.

    Returns:
        list: A new list containing all elements from both input lists, sorted in ascending order.
    """
    # Create an empty list to store the merged result
    result = []

    # Initialize two pointers for the left and right lists
    i = j = 0

    # Compare elements from both lists and append the smaller element to the result list
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append any remaining elements from the left list
    while i < len(left):
        result.append(left[i])
        i += 1

    # Append any remaining elements from the right list
    while j < len(right):
        result.append(right[j])
        j += 1

    # Return the merged and sorted list
    return result


def merge_sort(A: list) -> list:
    """
    Sorts a list of elements using the merge sort algorithm.

    Parameters:
    A (list): The list of elements to be sorted.

    Returns:
    list: The sorted list.

    """
    # Base case: if the list has 0 or 1 element, it is already sorted
    if len(A) <= 1:
        return A

    # If the list has less than or equal to 100 elements, use insertion sort for efficiency
    if len(A) <= 100:
        return insertion_sort(A)

    # Divide the list into two halves
    mid = len(A) // 2
    left = merge_sort(A[:mid])
    right = merge_sort(A[mid:])

    # Merge the sorted halves
    return merge(left, right)


def merge_all(sorted_sublists):
    """
    Merges multiple sorted sublists into a single sorted list.

    Args:
        sorted_sublists (list): A list of sorted sublists.

    Returns:
        list: A single sorted list containing all elements from the sublists.
    """
    return list(heapq.merge(*sorted_sublists))


def hybrid_sort(A: list):
    """
    Sorts a list using hybrid sort algorithm.

    If the length of the list is less than or equal to 100, it uses insertion sort.
    Otherwise, it divides the list into sublists based on the number of CPU cores available,
    and sorts each sublist using merge sort in parallel using multiprocessing.Pool.
    Finally, it merges all the sorted sublists into a single sorted list.

    Args:
        A (list): The list to be sorted.

    Returns:
        list: The sorted list.
    """
    # If the length of the list is less than or equal to 100, use insertion sort
    if len(A) <= 100:
        return insertion_sort(A)
    else:
        # Get the number of CPU cores available
        ncores = multiprocessing.cpu_count()

        # Divide the list into sublists based on the number of CPU cores
        sublists = [A[i::ncores] for i in range(ncores)]

        # Sort each sublist using merge sort in parallel using multiprocessing.Pool
        with Pool(ncores) as p:
            sorted_sublists = p.map(merge_sort, sublists)

        # Merge all the sorted sublists into a single sorted list
        return merge_all(sorted_sublists)
