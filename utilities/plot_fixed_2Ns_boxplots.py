import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Define a function to read the data from the file
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                values = line.split()
                true_value = float(values[0])
                if true_value != 50:
                    values_for_boxplot = [float(val) for val in values[1:]]
                    data.append((true_value, values_for_boxplot))
    return data

def custom_log_transform(value):
    if value == 0:
        return 0  # Handle exact zeros
    return np.sign(value) * np.log10(abs(value))

def run(inputfilepath, outputfilepath):
    # Obtain the inputfilepath_prefix from the full pathname
    inputfilepath_prefix = os.path.splitext(os.path.basename(inputfilepath))[0].split('_gvals')[0]

    # Read data from the file
    data = read_data(inputfilepath)

    # Initialize a figure and axes for the boxplots
    fig, ax = plt.subplots()

    # Calculate the number of data points and the gap between x-axis values
    num_data_points = len(data)
    x_positions = np.arange(1, num_data_points + 1)

    # Create boxplots and add arrows for true values
    # for i, (true_value, values_for_boxplot) in enumerate(data):
    #     x_position = x_positions[i]

    #     # Create the boxplot with raw data
    #     boxplot = ax.boxplot(
    #         [values_for_boxplot],
    #         positions=[x_position],
    #         patch_artist=True,
    #         showfliers=False,
    #         widths=0.6,
    #     )

    #     # Add an arrow to indicate the true value
    #     ax.arrow(
    #         x_position - 0.5, true_value, 0.2, 0, head_width=0.2, head_length=0.2, color="black"
    #     )

    for i, (true_value, values_for_boxplot) in enumerate(data):
        x_position = x_positions[i]

        # Create the boxplot with raw data
        boxplot = ax.boxplot(
            [values_for_boxplot],
            positions=[x_position],
            patch_artist=True,
            showfliers=False,
            widths=0.6,
        )
        # Add an arrow to indicate the true value, fully to the left of the boxplot
        arrow_offset = 0.4  # Adjust the offset for arrow placement
        ax.annotate(
            '',  # No text, just an arrow
            xy=(x_position - arrow_offset, true_value),  # Move the arrow tip to the left
            xytext=(x_position - 0.8, true_value),  # Move the arrow base further to the left
            arrowprops=dict(
                arrowstyle="->",  # Arrow style
                color="black",
                lw=1.5,  # Line width
                shrinkA=0,  # No shrink at the arrow tip
                shrinkB=0,  # No shrink at the base
            )
        )
       
    # Set x-axis labels and limits
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(true_value) for true_value, _ in data], rotation=45, ha="right")
    ax.set_xlim(0.5, num_data_points + 0.5)

    # Set the y-axis to symmetric log scale
    ax.set_yscale("symlog", linthresh=1)  # Adjust linthresh to control the linear region

    # Customize y-axis ticks to explicitly include -10^0 and 10^0
    y_ticks = [-10000,-1000, -100, -10, -1, 0, 1, 10, 100]
    ax.set_yticks(y_ticks)
    y_labels = [
        f"$-10^{{{int(np.log10(abs(tick)))}}}$" if tick < 0 else f"$10^{{{int(np.log10(tick))}}}$" if tick > 0 else "0"
        for tick in y_ticks
    ]
    ax.set_yticklabels(y_labels)

    # Set labels for x and y axes
    # ax.set_ylabel(r'$\hat{\gamma}$ (symlog)', fontsize=14, labelpad=10, rotation='horizontal')
    ax.set_ylabel(r'Log($\hat{\gamma}$)', fontsize=14, labelpad=20, rotation='horizontal')

    ax.set_xlabel(r'$\gamma$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Alternatively, manually adjust the left margin if needed
    plt.subplots_adjust(left=0.2) 
    # Save the figure with the desired file name
    plt.savefig(outputfilepath, dpi=300, bbox_inches='tight')

    # Show the plot on the screen
    plt.show()

    # Print a message indicating where the plot is saved
    print(f"Plot saved as {outputfilepath}")



def holdnewrun(inputfilepath, outputfilepath):
    # Obtain the inputfilepath_prefix from the full pathname
    inputfilepath_prefix = os.path.splitext(os.path.basename(inputfilepath))[0].split('_gvals')[0]

    # Read data from the file
    data = read_data(inputfilepath)

    # Initialize a figure and axes for the boxplots
    fig, ax = plt.subplots()

    # Calculate the number of data points and the gap between x-axis values
    num_data_points = len(data)
    x_positions = np.arange(1, num_data_points + 1)

    # Lists to store x-axis positions and true values
    true_values = []

    # Custom logarithmic transformation for plotting
    def custom_log_transform(value):
        if value == 0:
            return 0  # Handle exact zeros (optional)
        return np.sign(value) * np.log10(abs(value))

    # Create boxplots and add darker orange arrows for true values
    # for i, (true_value, values_for_boxplot) in enumerate(data):
    #     x_position = x_positions[i]
    #     true_values.append(true_value)
        
    #     # Transform values for the custom log scale
    #     transformed_values = [custom_log_transform(val) for val in values_for_boxplot]
    #     transformed_true_value = custom_log_transform(true_value)
        
    #     # Create a boxplot with widened boxes and no outliers
    #     boxplot = ax.boxplot([transformed_values], positions=[x_position], patch_artist=True, showfliers=False, widths=0.6)
    #     ax.arrow(x_position - 0.7, transformed_true_value, 0.2, 0, head_width=0.1, head_length=0.2, color="black")

# Create boxplots and add arrows for true values
    for i, (true_value, values_for_boxplot) in enumerate(data):
        x_position = x_positions[i]
        true_values.append(true_value)

        # Transform values for the custom log scale
        transformed_values = [custom_log_transform(val) for val in values_for_boxplot]
        transformed_true_value = custom_log_transform(true_value)

        # Create the boxplot on the transformed data
        # Ensure the data is only for the transformed y-values
        boxplot = ax.boxplot([transformed_values], positions=[x_position], patch_artist=True, showfliers=False, widths=0.6)

        # Add an arrow to indicate the true value on the transformed scale
        ax.arrow(x_position - 0.7, transformed_true_value, 0.2, 0, head_width=0.1, head_length=0.2, color="black")


    # Set x-axis labels and limits
    
    # ax.set_xticklabels([str(true_value) for true_value, _ in data])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(true_value) for true_value, _ in data], rotation=45, ha="right")

    ax.set_xlim(0.5, num_data_points + 0.5)

    # Set labels for x and y axes
    ax.set_ylabel(r'$\hat{\gamma}$ (log)', fontsize=14, labelpad=10, rotation='horizontal')
    ax.set_xlabel(r'$\gamma$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Customize y-axis ticks to display as powers of 10
    y_ticks = ax.get_yticks()
    y_labels = []
    for tick in y_ticks:
        if tick > 0:
            label = f"$10^{{{int(tick)}}}$"
        elif tick < 0:
            label = f"$-10^{{{int(abs(tick))}}}$"
        else:
            label = "0"
        y_labels.append(label)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Remove the legend
    ax.legend().set_visible(False)

    # Save the figure with the desired file name
    plt.savefig(outputfilepath, dpi=300, bbox_inches='tight')

    # Show the plot on the screen
    plt.show()

    # Print a message indicating where the plot is saved
    print(f"Plot saved as {outputfilepath}")


def holdrun(inputfilepath,outputfilepath):
    # Set the full path to the data file
    # inputfilepath = "../../prfratio/output/constant/constant_fixed_slimwork_constant_fixed_gvals_for_plot.txt"
    # inputfilepath = "../../prfratio/output/expansion/fixed_slimwork_iexpansion_fixed_RS_N0.1_rescaleSegMuts_gvals_for_plot.txt"
    # inputfilepath = "../../prfratio/output/bottleneck/fixed_slimwork_ibottleneck_fixed_RS_N0.1_rescaleSegMuts_gvals_for_plot.txt"
    # inputfilepath = "../../prfratio/output/popstructure/fixed_slimwork_popstructure_fixed_CS_t5N_gvals_for_plot.txt"

    # inputfilepath = "../../prfratio/output/constant/fixed_9_10_constant_fixed_gvals_for_plot.txt"
    # inputfilepath = "../../prfratio/output/expansion/expansion_9_10_iexpansion_fixed_RS_N0.1_rescaleSegMuts_gvals_for_plot.txt"
    # inputfilepath = "../../prfratio/output/bottleneck/bottleneck_9_10_ibottleneck_fixed_RS_N0.1_rescaleSegMuts_gvals_for_plot.txt"
    # inputfilepath = "../../prfratio/output/popstructure/popstructure_9_10_popstructure_fixed_CS_t1N_gvals_for_plot.txt"

    # Obtain the inputfilepath_prefix from the full pathname
    inputfilepath_prefix = os.path.splitext(os.path.basename(inputfilepath))[0].split('_gvals')[0]

    # Read data from the file
    data = read_data(inputfilepath)

    # Initialize a figure and axes for the boxplots
    fig, ax = plt.subplots()

    # Calculate the number of data points and the gap between x-axis values
    num_data_points = len(data)
    x_positions = np.arange(1, num_data_points + 1)
    x_gap = 1.0

    # Lists to store x-axis positions and true values
    true_values = []

    # Create boxplots and add darker orange arrows for true values
    for i, (true_value, values_for_boxplot) in enumerate(data):
        x_position = x_positions[i]
        true_values.append(true_value)
        
        # Create a boxplot with widened boxes and no outliers
        boxplot = ax.boxplot([values_for_boxplot], positions=[x_position], patch_artist=True, showfliers=False, widths=0.6)
        ax.arrow(x_position-0.7, true_value, 0.2, 0, head_width=2, head_length=0.2, color="black")

    # Set x-axis labels and limits
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(true_value) for true_value, _ in data])
    ax.set_xlim(0.5, num_data_points + 0.5)

    # Set labels for x and y axes
    ax.set_ylabel(r'$\hat{\gamma}$', fontsize=14,labelpad=10, rotation='horizontal',)
    ax.set_xlabel(r'$\gamma$', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Remove the legend
    ax.legend().set_visible(False)

    # Show the plot on the screen
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the figure with the desired file name
    # output_file_path = f"{inputfilepath_prefix}_boxplot_new.png"
    plt.savefig(outputfilepath, dpi=300, bbox_inches='tight')

    # Show the plot on the screen
    plt.show()

    # Print a message indicating where the plot is saved
    print(f"Plot saved as {outputfilepath}")


def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",dest="inputfilepath",required=True, type=str, help="file with 2Ns values and lists of estimates ")    
    parser.add_argument("-o",dest="outputfilepath",required=True, type=str, help="filename for plot (must end in .pdf or .png)")    
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])

    
    return args


if __name__ == '__main__':
    """

    """
   
    args = parsecommandline()
    run(args.inputfilepath,args.outputfilepath)