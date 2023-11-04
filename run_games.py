import subprocess
import re

# Define the different agents and layouts
ghosts = ['dumby', 'greedy', 'smarty']
layouts = ['small_adv', 'medium_adv', 'large_adv']

# Open a file to write the results
with open('game_results.txt', 'w') as results_file:
    # Iterate over all combinations of ghosts and layouts
    for ghost in ghosts:
        for layout in layouts:
            # Construct the command (CHANGER NOM DE FICHIER)
            command = f'python run.py --agent gilles --ghost {ghost} --layout {layout}'
            results_file.write(f'Running command: {command}\n')

            # Run the command
            try:
                # Set a timeout for the subprocess (e.g., 120 seconds)
                output = subprocess.check_output(command, shell=True, timeout=120, stderr=subprocess.STDOUT)
                output = output.decode('utf-8')  # Decode output to string

                # Check for the score in the output
                score_match = re.search(r'Score: (-?\d+)', output)
                if score_match:
                    score = int(score_match.group(1))
                    results_file.write(f'Score: {score}\n')

                    # Check if the score is below -50
                    if score < -50:
                        results_file.write('Score dropped below -50, terminating this run.\n')
                else:
                    results_file.write('No score found in the output.\n')

            except subprocess.TimeoutExpired:
                results_file.write('Game timed out.\n')
            except subprocess.CalledProcessError as e:
                results_file.write(f'An error occurred: {e.output}\n')

            results_file.write('\n')  # Add a newline for readability between runs

# Print out the path to the results file
print("Results have been written to game_results.txt")
