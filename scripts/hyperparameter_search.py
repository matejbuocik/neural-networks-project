import subprocess
from multiprocessing import Process
import os

# Specify the range of values for the parameters
s_values = [2 ** i for i in range(4, 9)]
r_values = [round(0.001 * 1.15 ** i, 5) for i in range(1, 101) if round(0.001 * 1.15 ** i, 5) < 1]

# Create a directory to store output files
output_directory = 'output_files'
os.makedirs(output_directory, exist_ok=True)


# Function to run the program with given parameters and save output to a file
def run_program(s, r):
    output_file = os.path.join(output_directory, f'output_s{s}_r{r}.txt')
    command = f"./mlp -s {s} -r {r} > {output_file} 2>&1"
    subprocess.run(command, shell=True, check=True)


# Loop through all parameter combinations and run the program
processes = []
for s in s_values:
    for r in r_values:
        p = Process(target=run_program, args=(s, r))
        p.start()
        processes.append(p)

        if len(processes) >= 16:
            for proc in processes:
                proc.join()
            
            processes = []

print("All jobs completed.")
