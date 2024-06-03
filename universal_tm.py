'''
Delivery Activity: Developing a Turing Machine

Daniel Molina A01285254
Kaled Enríquez A01198666
Fidel Morales A01198630
Isaac Hernández Pérez A01411947

'''


import pandas as pd
from itertools import product
from typing import Callable
from typing import Optional
from typing import Tuple
'''
  delta_read: creates transition functions (delta) from a state q given an input w 
              according the TM defined by its transition table in the csv file
  
  params:
    - tm: pandas dataframe with transition table of the turing machine
    
  returns: 
    - function delta for the transitions with ->
      
      params: 
        - q: current state in the TM
        - w: read symbol of the string to be processed

      returns:
        - q: next state from given w transition
        - write: symbol to write in the tape
        - move: direction to move the head of TM (R or L)   
'''


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


'''
  process_string: function that processes a string under the definition of the
        automaton M from the turing machine specified by its transition table in 
        a csv file
        
  params:
    - M: string of csv file path defining a turing machine by its transition table
    - w: string to be processed
    - iters: max numbers of reads to perform
    - debug: if true, prints all transitions in the form, q_1: Bw_0w_1 ... w_i ... w_nB
    
  returns: 
      Tuple (bool, string)
      - boolean indicating if it reached a final state ot not
      - string of the resulting tape after halting
'''


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
      print(f'q{current_state} :  {w[:n]}\033[4m{w[n]}\033[0m{w[n + 1:]}')

    word_len = len(w)  # Current length
    symbol = w[n]  # Read symbol

    # Transition from current state and input symbol read
    try:
      current_state, write, move = delta(current_state, symbol)
    except Exception:
      # if no transition defined: halt
      #print(Exception)
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
      #print("Halted after reaching maximum computations")
      break

  # At completion, determine if a final state has been reached or not beacuse of errors
  is_final = current_state in F

  return is_final, w


'''
  enumerate: function that generates strings and validates if they are valid
  with a turing machine, it prints the first 50 valid strings or stops printing when 
  all possible strings have been generated with a given length

  params:
    - M: route of csv file containing the turing machine
    - n: optional paremeter, default is set to 50, it represents the max length of the strings to be generated

  returns:
    - It prints the first valid strings and stops printing when all possible strings have been generated with the given length
'''


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


enumerate("010.csv", 50)