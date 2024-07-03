import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def find_json_files(root_dir):
    json_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname.startswith('r_'):
                full_dir_path = os.path.join(dirpath, dirname)
                for folder in os.listdir(full_dir_path):
                    json_files.append(os.path.join(full_dir_path, folder, 'trainer_state.json'))
        break
    return json_files

def parse_json_files(json_files):
    parsed_data = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                parsed_name = json_file.split('/')[-3].split('_')[-3:]
                parsed_name = tuple(float(x) for x in parsed_name)
                parsed_data[parsed_name] = data['log_history'][-1]['eval_r2']
        except json.JSONDecodeError as e:
            print(f"Error parsing {json_file}: {e}")
        except Exception as e:
            print(f"Unexpected error with {json_file}: {e}")
    return parsed_data

if __name__ == "__main__":
    root_directory = "/gscratch/ml4ml/sidlak/structure_to_signal/generation"
    json_files = find_json_files(root_directory)
    data = parse_json_files(json_files)
    grouped_data = defaultdict(list)
    for (group, percentage, trial), value in data.items():
        grouped_data[(group, percentage)].append(value)

    # Calculate means and standard deviations
    means = defaultdict(float)
    std_devs = defaultdict(float)
    for key, values in grouped_data.items():
        means[key] = np.mean(values)
        std_devs[key] = np.std(values)

    # Prepare data for plotting
    groups = sorted(set(k[0] for k in means.keys()))
    percentages = sorted(set(k[1] for k in means.keys()))

    # Bar width and positions
    bar_width = 0.2
    bar_positions = np.arange(len(percentages))

    # Plotting
    fig, ax = plt.subplots()

    for i, group in enumerate(groups):
        mean_values = [means[(group, percentage)] for percentage in percentages]
        std_dev_values = [std_devs[(group, percentage)] for percentage in percentages]
        positions = bar_positions + i * bar_width
        ax.bar(positions, mean_values, bar_width, label=f'Group {group}')
    
        # Plot individual trial values as scatter points
        for j, percentage in enumerate(percentages):
            trial_values = grouped_data[(group, percentage)]
            scatter_positions = np.full_like(trial_values, positions[j])
            ax.scatter(scatter_positions, trial_values, marker='x', color='black')
    # Labels and title
    ax.set_xlabel('Percentage of Data Included')
    ax.set_ylabel('R2')
    ax.set_title('Grouped Bar Graph with Error Bars for Each Trial')
    ax.set_xticks(bar_positions + bar_width * (len(groups) - 1) / 2)
    ax.set_xticklabels(percentages)
    ax.legend()

    plt.savefig('./grouped_bar_graph.png')
