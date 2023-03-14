import argparse
import os
# Define the command-line arguments
parser = argparse.ArgumentParser(description='M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring')
parser.add_argument('-w', '--week', type=int, choices=[1, 2, 3, 4, 5], help='week to execute. Options are [1,2,3,4,5]')
parser.add_argument('-t', '--task', type=str, help='task to execute. Options depend on each week.')

# Parse the command-line arguments
args = parser.parse_args()

# Print the help message if no arguments are provided
if not any(vars(args).values()) or args.help:
    parser.print_help()
    exit()


# Define the path to the main.py file in the specified subdirectory
main_file = os.path.join("w"+str(args.week), 'main.py')

# Run the main.py file with the argument
os.system(f'python {main_file} {args.task}')