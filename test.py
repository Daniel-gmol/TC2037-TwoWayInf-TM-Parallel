from act71 import modexp, modlog, parallel_enumerate, insertion_sort, merge_sort, hybrid_sort
from compare_functions import *

import multiprocessing

import time
import random

if __name__ == '__main__':
    """
    This script demonstrates various functions related to modular arithmetic, parallel processing, and sorting algorithms.
    """

    num_cores = multiprocessing.cpu_count()

    base = 454545454
    mod = 5654445
    exp = 44444548

    # Binary Modular Exponentiation
    start = time.time()
    print(binary_modular_exponentiation(base, exp, mod))
    end = time.time()
    print(f"Simple Binary Modular Exponentiation took {end - start:.10f} seconds")

    # Built-in Pow
    start = time.time()
    print(pow(base, exp, mod))
    end = time.time()
    print(f"Simple built-in Pow took {end - start:.10f} seconds")

    # Main modexp (Pool of Multiprocesses)
    start = time.time()
    print(modexp(base, exp, mod, num_cores))
    end = time.time()
    print(f"Main modexp (Pool of Multiprocesses) took {end - start:.10f} seconds")

    # modexp from GPT
    start = time.time()
    print(parallel_modexp_gpt(base, exp, mod))
    end = time.time()
    print(f"modexp from GPT took {end - start:.10f} seconds")

    # modexp using Multiprocess
    start = time.time()
    print(modexp_Multiprocess(base, exp, mod, num_cores))
    end = time.time()
    print(f"modexp using Multiprocess took {end - start:.10f} seconds")

    # modexp using Threading
    start = time.time()
    print(modexp_Threading(base, exp, mod, num_cores))
    end = time.time()
    print(f"modexp using Threading took {end - start:.10f} seconds")

    # modexp using Pool of Threads
    start = time.time()
    print(modexp_ThreadPool(base, exp, mod, num_cores))
    end = time.time()
    print(f"modexp using Pool of Threads took {end - start:.10f} seconds")

    # modlog
    base = 454545454
    mod = 5654445
    y = 44444548

    # Main modlog using Multiprocesses
    start = time.time()
    print(modlog(base, mod, y, num_cores))
    end = time.time()
    print(f"Main modlog using Multiprocesses took {end - start:.10f} seconds")

    # Simple modlog
    start = time.time()
    print(modlog_Simple(base, mod, y))
    end = time.time()
    print(f"Simple modlog took {end - start:.10f} seconds")

    # Main parallel_enumerate
    start = time.time()
    parallel_enumerate('010.csv', 12)
    end = time.time()
    print(f"Main parallel_enumerate (Pool of Multiprocesses) took {end - start:.10f} seconds")

    # parallel_enumerate using Multiprocesses
    start = time.time()
    parallel_enumerate_Multiproccess('010.csv', 12)
    end = time.time()
    print(f"parallel_enumerate using Multiprocesses took {end - start:.10f} seconds")

    # parallel_enumerate using Process Pool
    start = time.time()
    parallel_enumerate_ProcessPool('010.csv', 12)
    end = time.time()
    print(f"parallel_enumerate using Process Pool from concurrent.future took {end - start:.10f} seconds")

    # parallel_enumerate using Threads
    start = time.time()
    parallel_enumerate_Threading('010.csv', 12)
    end = time.time()
    print(f"parallel_enumerate using Threads took {end - start:.10f} seconds")

    # parallel_enumerate using Pool of threads
    start = time.time()
    parallel_enumerate_ThreadPool('010.csv', 12)
    end = time.time()
    print(f"parallel_enumerate using Pool of threads took {end - start:.10f} seconds")

    # Simple enumerate
    start = time.time()
    enumerate('010.csv', 12)
    end = time.time()
    print(f"Simple enumerate took {end - start:.10f} seconds")

    # Hybrid sort
    A = [random.randint(1, 9999) for _ in range(10000)]

    # insertion_sort
    start = time.time()
    x = insertion_sort(A)
    end = time.time()
    print(f"insertion_sort took {end - start:.10f} seconds")

    # merge_sort
    start = time.time()
    y = merge_sort(A)
    end = time.time()
    print(f"merge_sort took {end - start:.10f} seconds")

    # hybrid_sort
    start = time.time()
    z = hybrid_sort(A)
    end = time.time()
    print(f"hybrid_sort took {end - start:.10f} seconds")
