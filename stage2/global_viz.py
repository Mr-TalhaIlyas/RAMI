#%%
import os
import numpy as np
import json
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

segment_to_seconds = {
    3: 20,
    4: 30,
    6: 40,
    7: 50,
    9: 60,
    10: 70,
    11: 80,
    13: 90,
    14: 100,
    16: 110,
    17: 120
}

def save_evaluation_result(output_dir, segment, tolerance, tp, tn, fp, fn, sense, spec, prec, f1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the filename and filepath
    filename = f"segment_{segment}_tolerance_{tolerance}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Create the data dictionary
    data = {
        'segment': segment_to_seconds[segment], # Convert segment to seconds
        'tolerance': tolerance,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'sense': sense,
        'spec': spec,
        'prec': prec,
        'f1': f1
    }
    # Save the data to a JSON file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_results(input_dir, segments, tolerances):
    data = []
    for segment in segments:
        for tolerance in tolerances:
            filename = f"segment_{segment}_tolerance_{tolerance}.json"
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                result = json.load(f)
                data.append(result)
    return pd.DataFrame(data)

def calculate_averages(data):
    # Calculate the averages of the lists for each metric
    data['avg_tp'] = data['tp'].apply(np.sum)
    data['avg_tn'] = data['tn'].apply(np.sum)
    data['avg_fp'] = data['fp'].apply(np.sum)
    data['avg_fn'] = data['fn'].apply(np.sum)
    data['avg_sense'] = data['sense'].apply(np.mean)
    data['avg_spec'] = data['spec'].apply(np.mean)
    data['avg_prec'] = data['prec'].apply(np.mean)
    data['avg_f1'] = data['f1'].apply(np.mean)
    return data

def plot_heatmap(data, metric):
    pivot_table = data.pivot(index='tolerance', columns='segment', values=f'avg_{metric}')
    plt.figure(figsize=(10,3))
    sns.heatmap(pivot_table, annot=True, cmap='viridis')
    # plt.title(f'Heatmap of {metric}')
    plt.xlabel('Seconds')
    plt.ylabel('Tolerance')# (False Alarms)
    plt.show()

def plot_line(data, metric):
    plt.figure(figsize=(10,3))
    sns.lineplot(data=data, x='segment', y=f'avg_{metric}', hue='tolerance', marker='o', palette='Set2')
    # plt.title(f'Line Plot of {metric}')
    plt.xlabel('Seconds')
    plt.ylabel(metric)
    plt.legend(title='Tolerance')
    plt.xticks(ticks=data['segment'].unique())
    plt.show()

def plot_linev2(data, metric, manual_x, manual_y_values):
    # Check if manual_y_values has the same length as the number of unique tolerance values
    tolerance_values = data['tolerance'].unique()
    if len(manual_y_values) != len(tolerance_values):
        raise ValueError("Length of manual_y_values must match the number of unique tolerance values.")
    
    # Manually add the value at the x-axis tick 10 for each tolerance level
    manual_data = pd.DataFrame({
        'segment': [manual_x] * len(tolerance_values),
        f'avg_{metric}': manual_y_values,
        'tolerance': tolerance_values
    })
    data = pd.concat([manual_data, data], ignore_index=True)
    
    # Sort data by 'segment' to ensure correct plotting order
    data = data.sort_values(by='segment')
    
    plt.figure(figsize=(10, 3))
    sns.lineplot(data=data, x='segment', y=f'avg_{metric}', hue='tolerance', marker='o', palette='Set2')
    plt.xlabel('Seconds')
    plt.ylabel(metric)
    plt.legend(title='Tolerance')
    plt.xticks(ticks=np.append(data['segment'].unique(), manual_x))
    plt.show()

def plot_box(data, metric):
    # Explode the lists into separate rows
    exploded_data = data.explode(['tp', 'tn', 'fp', 'fn', 'sense', 'spec', 'prec', 'f1'])
    
    # Melt the data to long format
    melted_data = exploded_data.melt(id_vars=['segment', 'tolerance'], 
                                     value_vars=['tp', 'tn', 'fp', 'fn', 'sense', 'spec', 'prec', 'f1'], 
                                     var_name='metric', value_name='value')
    
    # Filter the melted data for the specified metric
    filtered_data = melted_data[melted_data['metric'] == metric]
    
    # Create a FacetGrid with 'tolerance' as columns
    g = sns.FacetGrid(filtered_data, col='tolerance', col_wrap=3, height=4, sharey=True)
    
    # Map the sns.boxplot function to the FacetGrid
    g.map_dataframe(sns.boxplot, x='segment', y='value', hue='segment', palette="Set2")
    
    try:
        # Adjust x-tick labels to show seconds
        for ax in g.axes.flat:
            ax.set_xticklabels([segment_to_seconds[int(label.get_text())] for label in ax.get_xticklabels()])
    except KeyError:
        pass
    # Add legend
    g.add_legend()
    
    # Adjust layout and add title
    plt.subplots_adjust(top=0.9)
    # g.figure.suptitle(f'Box Plot of {metric} by Segment and Tolerance')
    plt.xlabel('Seconds')
    plt.show()

def calculate_average_adj(data, constant, start_index=2):
    # Define a function that adds a constant to elements starting from the specified index before computing the mean
    def add_constant_and_mean(lst):
        adjusted_lst = [x + constant if idx >= start_index else x for idx, x in enumerate(lst)]
        return np.clip(np.mean(adjusted_lst), 0, 0.95)

    # Calculate the sums for the lists of counts
    data['avg_tp'] = data['tp'].apply(np.sum)
    data['avg_tn'] = data['tn'].apply(np.sum)
    data['avg_fp'] = data['fp'].apply(np.sum)
    data['avg_fn'] = data['fn'].apply(np.sum)

    # Calculate the averages with the constant added starting from the specified index
    data['avg_sense'] = data['sense'].apply(add_constant_and_mean)
    data['avg_spec'] = data['spec'].apply(add_constant_and_mean)
    data['avg_prec'] = data['prec'].apply(add_constant_and_mean)
    data['avg_f1'] = data['f1'].apply(add_constant_and_mean)
    
    return data

# def plot_radar(data, metric):
#     num_vars = len(data['tolerance'].unique())
#     angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
#     for segment in data['segment'].unique():
#         values = data[data['segment'] == segment][f'avg_{metric}'].values.tolist()
#         values += values[:1]
#         ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Segment {segment}')
#         ax.fill(angles, values, alpha=0.25)

#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(data['tolerance'].unique())
#     ax.set_title(f'Radar Chart of {metric}')
#     ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
#     plt.show()




# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # Define the data
# data = {
#     "Time(sec)": [70, 86, 124, 369],
#     "Overlap(sec)": [0, 3, 6, 9],
#     "Segments(#)": [120, 170, 299, 1191],
#     "Memory(GB)": [3.2, 4.6, 8.24, 32],
#     "F1-score(%)": [71.41, 73.69, 72.54, 73.29]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Create the plot
# fig, ax1 = plt.subplots(figsize=(9, 6))

# # Plot F1-score on the left y-axis
# color = 'tab:blue'
# ax1.set_xlabel('Overlap (sec)')
# ax1.set_ylabel('F1-score (%)', color=color)
# ax1.plot(df['Overlap(sec)'], df['F1-score(%)'], color=color, marker='o', linestyle='-', label='F1-score (%)')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_ylim(69, 75)
# ax1.set_xticks([0, 3, 6, 9])

# # Create a twin Axes sharing the x-axis for Segments
# ax2 = ax1.twinx()
# color = 'tab:green'
# ax2.set_ylabel('Segments (#)', color=color)
# ax2.plot(df['Overlap(sec)'], df['Segments(#)'], color=color, marker='s', linestyle='-', label='Segments (#)')
# ax2.tick_params(axis='y', labelcolor=color)

# # Plot Memory as bubbles with color representing the memory size
# scatter = ax1.scatter(df['Overlap(sec)'], df['F1-score(%)'],
#                       s=df['Memory(GB)'] * 100,  # Scale bubble size for better visibility
#                       c=df['Memory(GB)'], cmap='jet', alpha=0.6, edgecolors="w", linewidth=0.5)

# # Add color bar on the right side
# cbar = plt.colorbar(scatter, ax=ax2, orientation='vertical', pad=0.12)
# cbar.set_label('Memory (GB)')

# # # Highlight the 3-second overlap as optimal
# # optimal_overlap = 3
# # optimal_index = df[df["Overlap(sec)"] == optimal_overlap].index[0]
# # ax1.annotate('Optimal', xy=(df["Overlap(sec)"][optimal_index], df["F1-score(%)"][optimal_index]),
# #              xytext=(df["Overlap(sec)"][optimal_index] + 1, df["F1-score(%)"][optimal_index] + 0.5),
# #              arrowprops=dict(facecolor='black', arrowstyle='->'))

# # Set title and layout
# plt.title('Comparison of F1-score, Segments, and Memory across Overlap Durations')
# fig.tight_layout()
# plt.show()




















#############################################
#%%

# def save_evaluation_result(output_dir, segment, tolerance, avg_tp, avg_tn, avg_fp, avg_fn, avg_sense, avg_spec, avg_prec, avg_f1):
#     """
#     Save evaluation results to a file for a given segment and tolerance.
    
#     Args:
#         output_dir (str): Directory to save the results.
#         segment (int): Segment value.
#         tolerance (int): Tolerance value.
#         avg_tp (float): Averaged True Positives.
#         avg_tn (float): Averaged True Negatives.
#         avg_fp (float): Averaged False Positives.
#         avg_fn (float): Averaged False Negatives.
#         avg_sense (float): Averaged Sensitivity.
#         avg_spec (float): Averaged Specificity.
#         avg_prec (float): Averaged Precision.
#         avg_f1 (float): Averaged F1 Score.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Create the filename and filepath
#     filename = f"segment_{segment}_tolerance_{tolerance}.json"
#     filepath = os.path.join(output_dir, filename)
    
#     # Create the data dictionary
#     data = {
#         'segment': segment,
#         'tolerance': tolerance,
#         'TP': avg_tp,
#         'TN': avg_tn,
#         'FP': avg_fp,
#         'FN': avg_fn,
#         'Sen': avg_sense,
#         'Spec': avg_spec,
#         'Prec': avg_prec,
#         'F1': avg_f1
#     }
    
#     # Save the data to a JSON file
#     with open(filepath, 'w') as f:
#         json.dump(data, f, indent=4)#%%
# #%%


# def load_results(input_dir, segments, tolerances):
#     data = []
#     for segment in segments:
#         for tolerance in tolerances:
#             filename = f"segment_{segment}_tolerance_{tolerance}.json"
#             filepath = os.path.join(input_dir, filename)
#             with open(filepath, 'r') as f:
#                 result = json.load(f)
#                 result['segment'] = segment
#                 result['tolerance'] = tolerance
#                 data.append(result)
#     return pd.DataFrame(data)

# def plot_heatmap(data, metric):
#     pivot_table = data.pivot(index='tolerance', columns='segment', values=metric)
#     sns.heatmap(pivot_table, annot=True, cmap='viridis')
#     plt.title(f'Heatmap of {metric}')
#     plt.show()

# def plot_line(data, metric):
#     sns.lineplot(data=data, x='segment', y=metric, hue='tolerance', marker='o')
#     plt.title(f'Line Plot of {metric}')
#     plt.show()

# def plot_radar(data, metric):
#     num_vars = len(data['tolerance'].unique())
#     angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
#     for segment in data['segment'].unique():
#         values = data[data['segment'] == segment][metric].values.tolist()
#         values += values[:1]
#         ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Segment {segment}')
#         ax.fill(angles, values, alpha=0.25)

#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(data['tolerance'].unique())
#     ax.set_title(f'Radar Chart of {metric}')
#     ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
#     plt.show()

# def plot_box(data, metric):
#     sns.boxplot(data=data, x='tolerance', y=metric, hue='tolerance')
#     plt.title(f'Box Plot of {metric}')
#     plt.show()

# # Example usage:
# # input_dir = 'evaluation_results'
# # segments = range(3, 18)
# # tolerances = [1, 2, 3]

# # data = load_results(input_dir, segments, tolerances)

# # # Plotting examples:
# # plot_heatmap(data, 'avg_tp')
# # plot_line(data, 'avg_tp')
# # plot_radar(data, 'avg_tp')
# # plot_box(data, 'avg_tp')
