import multiprocessing
from multiprocessing import Process, Array, Pool, Queue

import threading

import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import math
from math import ceil, sqrt, gcd
from itertools import chain, product

import pandas as pd

import queue
import heapq

from ctypes import c_int64

from functools import partial

from typing import Callable
from typing import Optional
from typing import Tuple


# ------------------------- Parallel modexp by GPT -------------------------
def modexp_gpt(base, exponent, modulus):
    """Compute modular exponentiation."""
    result = 1
    base = base % modulus  # Reduce base modulo modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        exponent //= 2
    return result


def parallel_modexp_gpt(base, exponent, modulus):
    """Compute modular exponentiation using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunks = math.ceil(exponent / executor._max_workers)
        futures = []
        for i in range(executor._max_workers):
            start = i * chunks
            end = min((i + 1) * chunks, exponent)
            futures.append(executor.submit(
                modexp_gpt, base, end - start, modulus))
        result = 1
        for future in concurrent.futures.as_completed(futures):
            result = (result * future.result()) % modulus
        return result


# ------------------------- Auxiliar modexp -------------------------
def binary_modular_exponentiation(b, x, m):
    binary_exponent = bin(x)[2:]
    result = 1
    for bit in binary_exponent:
        result = (result * result) % m
        if bit == '1':
            result = (result * b) % m
    return result


# ------------------------- Auxiliar modexp -------------------------
def binary_modular_exponentiation_arr(b, x, m, results, index):
    binary_exponent = bin(x)[2:]
    result = 1
    for bit in binary_exponent:
        result = (result * result) % m
        if bit == '1':
            result = (result * b) % m

    results[index] = result


# ------------------------- modexp v1 -------------------------
def modexp_Multiprocess(b: int, x: int, m: int, ncores: int) -> int:
    exp = x // ncores
    remainder = x % ncores
    workloads = [exp + 1 if i < remainder else exp for i in range(ncores)]
    workloads = [workload for workload in workloads if workload != 0]
    work_len = len(workloads)
    results = Array(c_int64, work_len)
    proc = [Process(target=binary_modular_exponentiation_arr, args=(
        b, workloads[i], m, results, i)) for i in range(work_len)]
    [process.start() for process in proc]
    [process.join() for process in proc]
    result = 1
    for r in results:
        result = (result * r) % m
    return result

# ------------------------- modexp v2 -------------------------


def modexp_ThreadPool(b: int, x: int, m: int, ncores: int) -> int:
    exp = x // ncores
    remainder = x % ncores
    workloads = [exp + 1 if i < remainder else exp for i in range(ncores)]
    workloads = [workload for workload in workloads if workload != 0]
    work_len = len(workloads)
    with ThreadPoolExecutor(max_workers=ncores) as executor:
        results = list(executor.map(binary_modular_exponentiation, [
                       b]*work_len, workloads, [m]*work_len))
    result = 1
    for r in results:
        result = (result * r) % m
    return result


# ------------------------- Auxiliar modexp -------------------------
def binary_modular_exponentiation_thread(b, x, m, result, index):
    result[index] = binary_modular_exponentiation(b, x, m)


# ------------------------- modexp v3 -------------------------
def modexp_Threading(b: int, x: int, m: int, ncores: int) -> int:
    exp = x // ncores
    remainder = x % ncores
    workloads = [exp + 1 if i < remainder else exp for i in range(ncores)]
    workloads = [workload for workload in workloads if workload != 0]
    work_len = len(workloads)

    threads = []
    results = [None] * work_len
    for i in range(work_len):
        thread = threading.Thread(target=binary_modular_exponentiation_thread, args=(
            b, workloads[i], m, results, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    result = 1
    for r in results:
        result = (result * r) % m
    return result

# ------------------------- Single && Simple modlog -------------------------


def modlog_Simple(b: int, m: int, y: int) -> int:
    for x in range(m):
        if pow(b, x, m) == y:
            return x
    return -1  # if no such x is found


# Turing machine functions: delta_read
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

# Turing machine functions: process_string


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

# ------------- Auxiliar Check_Turing Machine for enumerate -----------------


def check_tm(M: str, combinations: str, q: queue.Queue):
    for combination in combinations:
        word = ''.join(combination)
        if process_string(M, word)[0]:
            q.put(word)

# ------------- Auxiliar Check_Turing Machine for enumerate -----------------


def check_tm_single(M: str, combination: tuple):
    word = ''.join(combination)
    if process_string(M, word)[0]:
        return word
    return None


# ------------------------- Parallel enumerate v1 -------------------------
def parallel_enumerate_Threading(M: str, n: Optional[int] = 50):
    try:
        n = n or 50  # Assigns default value if n is None
        ncores = multiprocessing.cpu_count()  # Number of cores in the machine
        q = queue.Queue()  # Queue to store the words to be processed
        threads = []  # List to store the threads

        # currN is the current length of the string to be generated, starts at 1
        for currN in range(1, n + 1):
            # ListCombinations is a list of tuples of all possible
            # strings with the given length
            listCombinations = list(product(("0", "1"), repeat=currN))

            # Split combinations into chunks for each thread
            workloads = [listCombinations[i::ncores] for i in range(ncores)]
            workloads = [workload for workload in workloads if workload != []]

            for work in workloads:
                thread = threading.Thread(target=check_tm, args=(M, work, q))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            while not q.empty():
                print(q.get())
    except KeyboardInterrupt:
        for thread in threads:
            thread.join()
        print("")


# ------------------------- Parallel enumerate v2 -------------------------
def parallel_enumerate_Multiproccess(M: str, n: Optional[int] = 50):
    try:
        n = n or 50  # Assigns default value if n is None
        ncores = multiprocessing.cpu_count()  # Number of cores in the machine
        q = Queue()  # Queue to store the words to be processed
        processes = []  # List to store the processes

        # currN is the current length of the string to be generated, starts at 1
        for currN in range(1, n + 1):
            # ListCombinations is a list of tuples of all possible
            # strings with the given length
            listCombinations = list(product(("0", "1"), repeat=currN))

            # Split combinations into chunks for each process
            workloads = [listCombinations[i::ncores] for i in range(ncores)]
            workloads = [workload for workload in workloads if workload != []]

            for work in workloads:
                process = Process(target=check_tm, args=(M, work, q))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            while not q.empty():
                print(q.get())
    except KeyboardInterrupt:
        for process in processes:
            process.join()
        print("")


# ------------------------- Parallel enumerate v3 -------------------------
def parallel_enumerate_ThreadPool(M: str, n: Optional[int] = 50):
    n = n or 50  # Assigns default value if n is None
    ncores = multiprocessing.cpu_count()  # Number of cores in the machine

    with ThreadPoolExecutor(max_workers=ncores) as executor:
        for currN in range(1, n + 1):
            listCombinations = list(product(("0", "1"), repeat=currN))
            # Create a new function that has M as a fixed argument
            partial_worker = partial(check_tm_single, M)
            # Use this new function with executor.map
            results = executor.map(partial_worker, listCombinations)
            for word in results:
                if word is not None:
                    print(word)


# ------------------------- Parallel enumerate v4 -------------------------
def parallel_enumerate_ProcessPool(M: str, n: Optional[int] = 50):
    n = n or 50  # Assigns default value if n is None
    ncores = multiprocessing.cpu_count()  # Number of cores in the machine

    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for currN in range(1, n + 1):
            listCombinations = list(product(("0", "1"), repeat=currN))
            # Create a new function that has M as a fixed argument
            partial_worker = partial(check_tm_single, M)
            # Use this new function with executor.map
            results = executor.map(partial_worker, listCombinations)
            for word in results:
                if word is not None:
                    print(word)


# ------------------------- Single && Simplw enumerate ----------------------
def enumerate(M: str, n: Optional[int] = 50):
    n = n or 50  # Assigns default value if n is None

    # currN is the current length of the string to be generated, starts at 1
    for currN in range(1, n + 1):
        # ListCombinations is a list of tuples of all possible
        # strings with the given length
        listCombinations = list(product(("0", "1"), repeat=currN))

        # processed in the turing machine, if it is valid it is printed and
        # the counter is incremented, if the counter reaches 50, the function ends
        for combination in listCombinations:
            word = ''.join(
                combination)  # Each tuple in list is converted to a string
            if process_string(M, word)[0]:
                print(word)
