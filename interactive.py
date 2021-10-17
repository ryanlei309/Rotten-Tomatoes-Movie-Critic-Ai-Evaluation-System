"""
File: interactive.py
Name: Ryan Lei
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from util import *
from submission import *
FILENAME = 'weights'


def main():
	weight = {}  # Empty dictionary to store the weight dictionary
	with open(FILENAME, 'r') as f:  # Open the weight file.
		for line in f:
			line_lst = line.split()  # Split with space
			weight[str(line_lst[0])] = float(line_lst[1])  # Add the word and weight into weight dictionary.

	interactivePrompt(extractWordFeatures, weight)  # Do the interactive thing.


if __name__ == '__main__':
	main()