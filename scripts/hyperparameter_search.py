import subprocess
from multiprocessing import Process
import os


# Specify the range of values for the parameters
s_values = [2 ** i for i in range(4, 9)]
r_values = [round(0.001 * 1.05 ** i, 5) for i in range(1, 101) if round(0.001 * 1.05 ** i, 5) < 0.5]

print("Number of combinations: ", len(s_values)*len(r_values))

# Create a directory to store output files
output_directory = 'output_files_16-11'
weights_dir = 'weights_files_16-11'

os.makedirs(output_directory, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)


# Function to run the program with given parameters and save output to a file
def run_program(s, r, max_attempts=5):
    lr_coef = 1

    for attempt in range(1, max_attempts + 1):
        output_file = os.path.join(output_directory, f'output_s{s}_r{r}.txt')
        weight_out_file = os.path.join(weights_dir, f'output_s{s}_r{r}_attempt{attempt}.txt')

        command = f"./mlp -s {s} -r {r * lr_coef} -n {(2 * 60000) // s + 1} -o {weight_out_file}"
        if attempt > 1:
            command += f' -i {weight_in_file}'
        command += f" > {output_file} 2>&1"

        print("Now working on: [", command, "]")

        # Run the command
        subprocess.run(command, shell=True, check=True)

        lr_coef *= 1/2
        weight_in_file = weight_out_file
        
        # Check if 'nan' is present in the output file
        if is_nan_present(output_file):
            break  # Exit the loop if 'nan' is present

def is_nan_present(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return 'nan' in content

# Loop through all parameter combinations and run the program
max_processes = 16

process_queue = []

for s in s_values:
    for r in r_values:
        # Wait for a process to finish if the maximum number of processes is reached
        while len(process_queue) >= max_processes:
            for i, process in enumerate(process_queue):
                if not process.is_alive():
                    process_queue.pop(i)
                    break

        p = Process(target=run_program, args=(s, r))
        p.start()
        process_queue.append(p)

print("All jobs completed.")
