#Adding new items

def apply_percentage(numbers, percentages):
    """
    Computes the percentage of each number in the list.

    :param numbers: List of integers.
    :param percentages: List of percentages corresponding to each number.
                        For example, if percentages[i] is 66, it represents 66%.
    :return: A list where each element is (percentage/100 * corresponding number).
    """
    if len(numbers) != len(percentages):
        raise ValueError("The lists 'numbers' and 'percentages' must have the same length.")

    return [(p / 100.0) * num for num, p in zip(numbers, percentages)]

# Example usage:
numbers = [100, 200, 300]
percentages = [50, 75, 66]  # Representing 50%, 75%, 66%
result = apply_percentage(numbers, percentages)
print(result)  # Output: [50.0, 150.0, 198.0]

def barPlot_3(heuristic_cut, neural_cut1, neural_cut2, labels, nn_std_percent, nn_std_percent_ran, title = 'Comparison of 3-way Maximum Cut Values by Algorithm', y_lim = None,
              nn_std_percent_GCN = [], nn_std_percent_cplex = []):
    """
    Plots three bars (Integer Solver, Randomizer, Neural Network) side by side
    for each item on the x-axis. Only the Neural Network bars will have error bars
    based on a standard deviation expressed in percentage.

    :param heuristic_cut: List of values for "Integer Solver"
    :param neural_cut1:   List of values for "Randomizer"
    :param neural_cut2:   List of values for "Neural Network" (the bars that get error bars)
    :param labels:        List of x-axis labels
    :param nn_std_percent:
        List of standard deviations in PERCENTAGE (same length as the other lists).
        For example, if nn_std_percent[i] == 5.0, that means 5% of neural_cut2[i].
    """

    if not (len(heuristic_cut) == len(neural_cut1) == len(neural_cut2) == len(labels) == len(nn_std_percent) == len(nn_std_percent_ran) ):
        raise ValueError("All input lists must have the same length (including nn_std_percent).")

    # Number of groups
    n_groups = len(heuristic_cut)
    index = np.arange(n_groups)
    bar_width = 0.25  # Adjusted width for three bars per group

    plt.figure(figsize=(14, 6))

    # Bar 1: Integer Solver
    bar1 = plt.bar(
        index,
        heuristic_cut,
        bar_width,
        label='Integer Solver',
        color='skyblue'
    )

    # Bar 2: Randomizer
    # bar2 = plt.bar(
    #     index + bar_width,
    #     neural_cut1,
    #     bar_width,
    #     label='Randmizer',
    #     color='orange'
    # )
    nn_std_abs_ran = [
        (nn_std_percent_ran[i] / 100.0) * neural_cut1[i]
        for i in range(n_groups)
    ]
    bar3 = plt.bar(
        index + 1 * bar_width,
        neural_cut1,
        bar_width,
        label='Randmized Algorithm',
        color='orange',
        yerr=nn_std_abs_ran,   # <--- Attach error bars here
        capsize=5,
        ecolor='black'
    )
    # -------------------------------
    # Convert percentage std dev to absolute std dev for the NN bars
    # If nn_std_percent[i] = 5.0, that means "5% of neural_cut2[i]"
    nn_std_abs = [
        (nn_std_percent[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    # -------------------------------

    # Bar 3: Neural Network (with error bars)
    bar3 = plt.bar(
        index + 2 * bar_width,
        neural_cut2,
        bar_width,
        label='Neural Network',
        color='green',
        yerr=nn_std_abs,   # <--- Attach error bars here
        capsize=5,
        ecolor='black'
    )



    # Add labels, title, legend
    plt.xlabel('Graphs (nodes)')
    plt.ylabel('Maximum Cut Value')
    plt.title(title)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    if (y_lim != None):
        plt.ylim(top=y_lim)

    # Optionally, annotate the "Randomizer" & "Neural Network" bars with percentages
    for i in range(n_groups):
        cuts = [neural_cut1[i], neural_cut2[i]]
        for j, cut in enumerate(cuts):
            # Calculate the percentage relative to the Heuristic bar
            percentage = (cut / heuristic_cut[i]) * 100 if heuristic_cut[i] != 0 else 0

            # Position
            x_pos = index[i] + (j + 1) * bar_width
            y_pos = cut

            # Choose text color based on bar height for readability
            text_color = 'white' if y_pos > max(cuts) * 0.1 else 'black'

            plt.text(
                x_pos, y_pos / 2,
                f'{percentage:.0f}%',
                ha='center', va='center',
                color=text_color,
                fontsize=10, fontweight='bold'
            )

    plt.tight_layout()
    plt.show()



def barPlot_3_speedup(
        heuristic_cut, neural_cut1, neural_cut2, labels, nn_std_percent,
        nn_std_percent_ran, title='Comparison of 3-way Maximum Cut Values by Algorithm',
        y_lim=None
):
    """
    Plots three bars (Integer Solver, Randomizer, Neural Network) side by side
    for each item on the x-axis. Only the Neural Network bars will have error bars
    based on a standard deviation expressed in percentage.

    :param heuristic_cut:  List of values for "CPLEX"
    :param neural_cut1:    List of values for "Randomizer"
    :param neural_cut2:    List of values for "Neural Network"
    :param labels:         List of x-axis labels
    :param nn_std_percent: List of std devs in PERCENTAGE for neural_cut2
    :param nn_std_percent_ran: List of std devs in PERCENTAGE for neural_cut1
    :param title:          Plot title
    :param y_lim:          Optional Y-axis upper limit
    """
    # Basic input validation
    if not (
            len(heuristic_cut) == len(neural_cut1) == len(neural_cut2) ==
            len(labels) == len(nn_std_percent) == len(nn_std_percent_ran)
    ):
        raise ValueError("All input lists must have the same length.")

    n_groups = len(heuristic_cut)
    index = np.arange(n_groups)
    bar_width = 0.25

    plt.figure(figsize=(14, 6))

    # Bar 1: CPLEX (Heuristic)
    bar1 = plt.bar(
        index,
        heuristic_cut,
        bar_width,
        label='CPLEX',
        color='skyblue'
    )

    # Convert percentage std dev for Randomizer to absolute error
    nn_std_abs_ran = [
        (nn_std_percent_ran[i] / 100.0) * neural_cut1[i]
        for i in range(n_groups)
    ]
    # Bar 2: Randomizer
    bar2 = plt.bar(
        index + 1 * bar_width,
        neural_cut1,
        bar_width,
        label='Randomized Algorithm',
        color='orange',
        yerr=nn_std_abs_ran,
        capsize=5,
        ecolor='black'
    )

    # Convert percentage std dev for Neural Network to absolute error
    nn_std_abs = [
        (nn_std_percent[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    # Bar 3: Neural Network (with error bars)
    bar3 = plt.bar(
        index + 2 * bar_width,
        neural_cut2,
        bar_width,
        label='Neural Network',
        color='green',
        yerr=nn_std_abs,
        capsize=5,
        ecolor='black'
    )

    # Labeling, legend, etc.
    plt.xlabel('Graphs (nodes)')
    plt.ylabel('Maximum Cut Value')
    plt.title(title)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    if y_lim is not None:
        plt.ylim(top=y_lim)

    # Annotate the Randomizer & Neural Network bars with values on top
    for i in range(n_groups):
        cuts = [neural_cut1[i], neural_cut2[i]]
        for j, cut in enumerate(cuts):
            # Position the text above the bar
            x_pos = index[i] + (j + 1) * bar_width
            y_pos = cut

            # Small offset so text doesn't sit exactly on top of the bar
            offset = 0.02 * cut  # 2% of the bar's height, adjust as needed
            text_y = y_pos + offset+1

            # Choose text color based on bar height for readability
            # (Optional logic)
            text_color = 'black' #if cut > 0.3 * max(cuts) else 'black'

            # Place the text
            plt.text(
                x_pos,
                text_y,
                f'{cut:.0f}s',
                ha='center',
                va='bottom',     # Anchor the text from its bottom
                color=text_color,
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.show()

def barPlot_2_speedUp(heuristic_cut, neural_cut, labels, std_percent,  title = 'Balanced 3-way max-cut', y_lim=None):
    # Input validation
    if not (len(heuristic_cut) == len(neural_cut) == len(labels)):
        raise ValueError("All input lists must have the same length.")

    # Number of groups
    n_groups = len(heuristic_cut)
    index = np.arange(n_groups)
    bar_width = 0.35

    # Create the plot
    plt.figure(figsize=(12, 6))  # Adjusted the figure size for better visibility

    # Plot the bars
    # bar1 = plt.bar(index, heuristic_cut, bar_width, label='Cplex', color='skyblue')
    bar2 = plt.bar(index, neural_cut, bar_width, label='GCN', color='orange')

    # Add labels, title, and legend
    # Add labels, title, and legend
    plt.xlabel('Graph size (nodes)')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()

    nn_std_abs = [
        (std_percent[i] / 100.0) * neural_cut[i]
        for i in range(n_groups)
    ]

    bar3 = plt.bar(
        index ,
        neural_cut,
        bar_width,
        label='Neural Network',
        color='orange',
        yerr=nn_std_abs,   # <--- Attach error bars here
        capsize=5,
        ecolor='black'
    )

    if y_lim is not None:
        plt.ylim(top=y_lim)
    # Calculate percentages and add them inside the 'Neural Network' bars
    for i in range(n_groups):
        # Calculate the percentage
        percentage = (neural_cut[i] / heuristic_cut[i]) * 100 if heuristic_cut[i] != 0 else 0

        # Get the position and height of the 'Neural Network' bar
        x_pos = index[i]
        y_pos = neural_cut[i]


        # Position the text above the bar
        y_pos = neural_cut[i]

        # Small offset so text doesn't sit exactly on top of the bar
        offset = 0.02 * neural_cut[i]  # 2% of the bar's height, adjust as needed
        text_y = y_pos + offset+1

        # Choose text color based on bar height for readability
        # (Optional logic)
        text_color = 'black' #if cut > 0.3 * max(cuts) else 'black'

        # Place the percentage text inside the bar
        plt.text(
            x_pos,                      # X position
            text_y,                  # Y position (middle of the bar)
            f'{neural_cut[i]:.0f}s',       # Text to display
            ha='center',                # Horizontal alignment
            va='center',                # Vertical alignment
            color=text_color,           # Text color
            fontsize=10,                # Font size
            fontweight='bold'           # Font weight
        )

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def barPlot_3_dot(
        heuristic_cut,
        neural_cut1,
        neural_cut2,
        neural_cut2_dots,
        labels,
        nn_std_percent,
        nn_std_percent_ran,
        title='Comparison of 3-way Maximum Cut Values by Algorithm (With Dot Pattern)',
        y_lim=None,
        nn_std_percent_GCN = [],
        nn_std_percent_cplex = []
):
    """
    Plots three bars (Integer Solver, Randomizer, Neural Network) side by side
    for each item on the x-axis. Only the Randomizer and Neural Network bars
    will have error bars based on a standard deviation expressed in percentage.

    Additionally, a "transparent dot-pattern" overlay bar is added on top of
    the third bar (Neural Network) using the separate data list 'neural_cut2_dots'.

    :param heuristic_cut:      List of values for "Integer Solver"
    :param neural_cut1:        List of values for "Randomizer"
    :param neural_cut2:        List of values for "Neural Network" (bars + error bars)
    :param neural_cut2_dots:   List of values for a dot-pattern overlay on the 3rd bar
                               (must be same length as neural_cut2).
    :param labels:             List of x-axis labels
    :param nn_std_percent:     Std dev in PERCENT for neural_cut2
    :param nn_std_percent_ran: Std dev in PERCENT for neural_cut1
    :param title:              Plot title
    :param y_lim:              Optional Y-axis upper limit
    """
    print(len(heuristic_cut), len(neural_cut1), len(neural_cut2), len(neural_cut2_dots), len(labels), len(nn_std_percent), len(nn_std_percent_ran))
    print(nn_std_percent_ran)
    # Basic input check
    if not (
            len(heuristic_cut) == len(neural_cut1) == len(neural_cut2) ==
            len(neural_cut2_dots) == len(labels) ==
            len(nn_std_percent) == len(nn_std_percent_ran)
    ):
        raise ValueError(
            "All input lists (heuristic_cut, neural_cut1, neural_cut2, "
            "neural_cut2_dots, labels, nn_std_percent, nn_std_percent_ran) "
            "must have the same length."
        )

    n_groups = len(heuristic_cut)
    index = np.arange(n_groups)
    bar_width = 0.22

    plt.figure(figsize=(14, 6))

    # -------------------------------
    # Bar 1: Integer Solver
    # -------------------------------
    nn_std_percent_cplex_abs = [
        (nn_std_percent_cplex[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]

    bar1 = plt.bar(
        index,
        heuristic_cut,
        bar_width,
        label='CPLEX',
        color='skyblue',
        yerr=nn_std_percent_cplex_abs,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Bar 2: Randomizer (with error bars)
    # -------------------------------
    # Convert percentage std dev for Randomizer to absolute
    nn_std_abs_ran = [
        (nn_std_percent_ran[i] / 100.0) * neural_cut1[i]
        for i in range(n_groups)
    ]
    bar2 = plt.bar(
        index + bar_width,
        neural_cut1,
        bar_width,
        label='Randomized Algorithm',
        color='orange',
        yerr=nn_std_abs_ran,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Bar 3: Neural Network (with error bars)
    # -------------------------------
    # Convert percentage std dev for Neural Network to absolute
    nn_std_abs = [
        (nn_std_percent[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    bar3 = plt.bar(
        index + 3 * bar_width,
        neural_cut2,
        bar_width,
        label='GCN With Post-processing',
        color='green',
        yerr=nn_std_abs,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Dot-Pattern Overlay on Bar 3
    # -------------------------------
    # This uses 'neural_cut2_dots' for heights, with a hatch pattern for a dotted look
    # facecolor='none' makes the bar transparent, so only the hatch is visible
    # bar3_dots = plt.bar(
    #     index + 2 * bar_width,
    #     neural_cut2_dots,
    #     bar_width,
    #     facecolor='green',   # transparent fill
    #     edgecolor='black',  # or 'green' if you prefer
    #     alpha=0.3,          # how transparent the hatch lines are
    #     hatch='xx',          # 'o' hatch pattern for dotted style
    #     label='Neural Net'
    # )

    nn_std_abs = [
        (nn_std_percent_GCN[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    bar3 = plt.bar(
        index + 2 * bar_width,
        neural_cut2_dots,
        bar_width,
        label='GCN',
        color='red',
        yerr=nn_std_abs,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Labeling, legend, etc.
    # -------------------------------
    plt.xlabel('Graph size (nodes)')
    plt.ylabel('Maximum Cut Value')
    plt.title(title)
    plt.xticks(index + bar_width, labels)

    if y_lim is not None:
        plt.ylim(top=y_lim)

    plt.legend()

    # Example: Annotate the Randomizer & Neural Network bars with a relative percentage
    for i in range(n_groups):
        cuts = [neural_cut1[i],  neural_cut2_dots[i], neural_cut2[i]]
        for j, cut in enumerate(cuts):
            # Calculate the percentage relative to the Heuristic bar
            base = heuristic_cut[i] if heuristic_cut[i] else 1
            percentage = (cut / base) * 100

            x_pos = index[i] + (j + 1) * bar_width
            y_pos = cut

            text_color = 'white' if y_pos > max(cuts) * 0.3 else 'black'
            plt.text(
                x_pos, y_pos * 0.5,
                f'{percentage:.0f}%',
                ha='center', va='center',
                color=text_color,
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.show()

def barPlot_generic_dot(
        heuristic_cut,
        neural_cut1,
        neural_cut2,
        neural_cut2_dots,
        labels,
        nn_std_percent,
        nn_std_percent_ran,
        title='Comparison of 3-way Maximum Cut Values by Algorithm (With Dot Pattern)',
        y_lim=None,
        nn_std_percent_GCN = [],
        nn_std_percent_cplex = [],
        barLabels = ['CPLEX', 'Randomized Algorithm', 'GCN With Post-processing', 'GCN']
):
    """
    Plots three bars (Integer Solver, Randomizer, Neural Network) side by side
    for each item on the x-axis. Only the Randomizer and Neural Network bars
    will have error bars based on a standard deviation expressed in percentage.

    Additionally, a "transparent dot-pattern" overlay bar is added on top of
    the third bar (Neural Network) using the separate data list 'neural_cut2_dots'.

    :param heuristic_cut:      List of values for "Integer Solver"
    :param neural_cut1:        List of values for "Randomizer"
    :param neural_cut2:        List of values for "Neural Network" (bars + error bars)
    :param neural_cut2_dots:   List of values for a dot-pattern overlay on the 3rd bar
                               (must be same length as neural_cut2).
    :param labels:             List of x-axis labels
    :param nn_std_percent:     Std dev in PERCENT for neural_cut2
    :param nn_std_percent_ran: Std dev in PERCENT for neural_cut1
    :param title:              Plot title
    :param y_lim:              Optional Y-axis upper limit
    """
    print(len(heuristic_cut), len(neural_cut1), len(neural_cut2), len(neural_cut2_dots), len(labels), len(nn_std_percent), len(nn_std_percent_ran))
    print(nn_std_percent_ran)
    # Basic input check
    if not (
            len(heuristic_cut) == len(neural_cut1) == len(neural_cut2) ==
            len(neural_cut2_dots) == len(labels) ==
            len(nn_std_percent) == len(nn_std_percent_ran)
    ):
        raise ValueError(
            "All input lists (heuristic_cut, neural_cut1, neural_cut2, "
            "neural_cut2_dots, labels, nn_std_percent, nn_std_percent_ran) "
            "must have the same length."
        )

    n_groups = len(heuristic_cut)
    index = np.arange(n_groups)
    bar_width = 0.22

    plt.figure(figsize=(14, 6))

    # -------------------------------
    # Bar 1: Integer Solver
    # -------------------------------
    nn_std_percent_cplex_abs = [
        (nn_std_percent_cplex[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]

    bar1 = plt.bar(
        index,
        heuristic_cut,
        bar_width,
        label=barLabels[0],
        color='skyblue',
        yerr=nn_std_percent_cplex_abs,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Bar 2: Randomizer (with error bars)
    # -------------------------------
    # Convert percentage std dev for Randomizer to absolute
    nn_std_abs_ran = [
        (nn_std_percent_ran[i] / 100.0) * neural_cut1[i]
        for i in range(n_groups)
    ]
    bar2 = plt.bar(
        index + bar_width,
        neural_cut1,
        bar_width,
        label= barLabels[1],
        color='orange',
        yerr=nn_std_abs_ran,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Bar 3: Neural Network (with error bars)
    # -------------------------------
    # Convert percentage std dev for Neural Network to absolute
    nn_std_abs = [
        (nn_std_percent[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    bar3 = plt.bar(
        index + 3 * bar_width,
        neural_cut2,
        bar_width,
        label=barLabels[2],
        color='green',
        yerr=nn_std_abs,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Dot-Pattern Overlay on Bar 3
    # -------------------------------
    # This uses 'neural_cut2_dots' for heights, with a hatch pattern for a dotted look
    # facecolor='none' makes the bar transparent, so only the hatch is visible
    # bar3_dots = plt.bar(
    #     index + 2 * bar_width,
    #     neural_cut2_dots,
    #     bar_width,
    #     facecolor='green',   # transparent fill
    #     edgecolor='black',  # or 'green' if you prefer
    #     alpha=0.3,          # how transparent the hatch lines are
    #     hatch='xx',          # 'o' hatch pattern for dotted style
    #     label='Neural Net'
    # )

    nn_std_abs = [
        (nn_std_percent_GCN[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    bar3 = plt.bar(
        index + 2 * bar_width,
        neural_cut2_dots,
        bar_width,
        label=barLabels[3],
        color='red',
        yerr=nn_std_abs,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Labeling, legend, etc.
    # -------------------------------
    plt.xlabel('Graph size (nodes)')
    plt.ylabel('Maximum Cut Value')
    plt.title(title)
    plt.xticks(index + bar_width, labels)

    if y_lim is not None:
        plt.ylim(top=y_lim)

    plt.legend()

    # Example: Annotate the Randomizer & Neural Network bars with a relative percentage
    for i in range(n_groups):
        cuts = [neural_cut1[i],  neural_cut2_dots[i], neural_cut2[i]]
        for j, cut in enumerate(cuts):
            # Calculate the percentage relative to the Heuristic bar
            base = heuristic_cut[i] if heuristic_cut[i] else 1
            percentage = (cut / base) * 100

            x_pos = index[i] + (j + 1) * bar_width
            y_pos = cut

            text_color = 'white' if y_pos > max(cuts) * 0.3 else 'black'
            plt.text(
                x_pos, y_pos * 0.5,
                f'{percentage:.0f}%',
                ha='center', va='center',
                color=text_color,
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.show()


def barPlot_3_speedup_dot(
        heuristic_cut, neural_cut1, neural_cut2, neural_cut2_dots, labels, nn_std_percent,
        nn_std_percent_ran, title='Comparison of 3-way Maximum Cut Values by Algorithm',
        y_lim=None, nn_std_percent_GCN = []
):
    """
    Plots three bars (Integer Solver, Randomizer, Neural Network) side by side
    for each item on the x-axis. Only the Neural Network bars will have error bars
    based on a standard deviation expressed in percentage.

    :param heuristic_cut:  List of values for "CPLEX"
    :param neural_cut1:    List of values for "Randomizer"
    :param neural_cut2:    List of values for "Neural Network"
    :param labels:         List of x-axis labels
    :param nn_std_percent: List of std devs in PERCENTAGE for neural_cut2
    :param nn_std_percent_ran: List of std devs in PERCENTAGE for neural_cut1
    :param title:          Plot title
    :param y_lim:          Optional Y-axis upper limit
    """
    # Basic input validation
    if not (
            len(heuristic_cut) == len(neural_cut1) == len(neural_cut2) ==
            len(labels) == len(nn_std_percent) == len(nn_std_percent_ran)
    ):
        raise ValueError("All input lists must have the same length.")

    n_groups = len(heuristic_cut)
    index = np.arange(n_groups)
    bar_width = 0.22

    plt.figure(figsize=(14, 6))

    # Bar 1: CPLEX (Heuristic)
    # bar1 = plt.bar(
    #     index,
    #     heuristic_cut,
    #     bar_width,
    #     label='CPLEX',
    #     color='skyblue'
    # )

    # Convert percentage std dev for Randomizer to absolute error
    nn_std_abs_ran = [
        (nn_std_percent_ran[i] / 100.0) * neural_cut1[i]
        for i in range(n_groups)
    ]
    # Bar 2: Randomizer
    bar2 = plt.bar(
        index + 0 * bar_width,
        neural_cut1,
        bar_width,
        label='Randomized Algorithm',
        color='orange',
        yerr=nn_std_abs_ran,
        capsize=5,
        ecolor='black'
    )

    # # Convert percentage std dev for Neural Network to absolute error
    # nn_std_abs = [
    #     (nn_std_percent[i] / 100.0) * neural_cut2[i]
    #     for i in range(n_groups)
    # ]
    # # Bar 3: Neural Network (with error bars)
    # bar3 = plt.bar(
    #     index + 2 * bar_width,
    #     neural_cut2,
    #     bar_width,
    #     label='Neural Network',
    #     color='green',
    #     yerr=nn_std_abs,
    #     capsize=5,
    #     ecolor='black'
    # )

    # -------------------------------
    # Bar 3: Neural Network (with error bars)
    # -------------------------------
    # Convert percentage std dev for Neural Network to absolute
    nn_std_abs = [
        (nn_std_percent[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    bar3 = plt.bar(
        index + 2 * bar_width,
        neural_cut2,
        bar_width,
        label='GCN With Post-processing',
        color='green',
        yerr=nn_std_abs,
        capsize=5,
        ecolor='black'
    )

    # -------------------------------
    # Dot-Pattern Overlay on Bar 3
    # -------------------------------
    # This uses 'neural_cut2_dots' for heights, with a hatch pattern for a dotted look
    # facecolor='none' makes the bar transparent, so only the hatch is visible
    # bar3_dots = plt.bar(
    #     index + 2 * bar_width,
    #     neural_cut2_dots,
    #     bar_width,
    #     facecolor='green',   # transparent fill
    #     edgecolor='black',  # or 'green' if you prefer
    #     alpha=0.3,          # how transparent the hatch lines are
    #     hatch='xx',          # 'o' hatch pattern for dotted style
    #     label='Neural Net'
    # )

    nn_std_abs = [
        (nn_std_percent_GCN[i] / 100.0) * neural_cut2[i]
        for i in range(n_groups)
    ]
    bar3 = plt.bar(
        index + 1 * bar_width,
        neural_cut2_dots,
        bar_width,
        label='GCN',
        color='red',
        yerr=nn_std_abs,
        capsize=5,
        ecolor='black'
    )


    # Labeling, legend, etc.
    plt.xlabel('Graph size (nodes)')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    if y_lim is not None:
        plt.ylim(top=y_lim)

    # Annotate the Randomizer & Neural Network bars with values on top
    for i in range(n_groups):
        cuts = [neural_cut1[i],  neural_cut2_dots[i], neural_cut2[i]]
        for j, cut in enumerate(cuts):
            # Position the text above the bar
            x_pos = index[i] + (j) * bar_width
            y_pos = cut

            # Small offset so text doesn't sit exactly on top of the bar
            offset = 0.02 * cut  # 2% of the bar's height, adjust as needed
            text_y = y_pos + offset+1

            # Choose text color based on bar height for readability
            # (Optional logic)
            text_color = 'black' #if cut > 0.3 * max(cuts) else 'black'

            # Place the text
            plt.text(
                x_pos,
                text_y,
                f'{cut:.0f}s',
                ha='center',
                va='bottom',     # Anchor the text from its bottom
                color=text_color,
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.show()

def barPlot_2(heuristic_cut, neural_cut, labels, std_percent,  title = 'Balanced 3-way max-cut', nn_std_percent_cplex_abs = []):
    # Input validation
    if not (len(heuristic_cut) == len(neural_cut) == len(labels)):
        raise ValueError("All input lists must have the same length.")

    # Number of groups
    n_groups = len(heuristic_cut)
    index = np.arange(n_groups)
    bar_width = 0.35

    # Create the plot
    plt.figure(figsize=(12, 6))  # Adjusted the figure size for better visibility

    # Plot the bars
    bar1 = plt.bar(index, heuristic_cut, bar_width, label='Cplex', color='skyblue')
    bar2 = plt.bar(index + bar_width, neural_cut, bar_width, label='GCN', color='orange')

    # Add labels, title, and legend
    # Add labels, title, and legend
    plt.xlabel('Graph size (nodes)')
    plt.ylabel('Maximum Cut Value')
    plt.title(title)
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()

    nn_std_percent_cplex_abs = [
        (nn_std_percent_cplex[i] / 100.0) * heuristic_cut[i]
        for i in range(n_groups)
    ]

    bar1 = plt.bar(
        index,
        heuristic_cut,
        bar_width,
        label='CPLEX',
        color='skyblue',
        yerr=nn_std_percent_cplex_abs,
        capsize=5,
        ecolor='black'
    )

    nn_std_abs = [
        (std_percent[i] / 100.0) * neural_cut[i]
        for i in range(n_groups)
    ]

    bar3 = plt.bar(
        index + 1 * bar_width,
        neural_cut,
        bar_width,
        label='Neural Network',
        color='orange',
        yerr=nn_std_abs,   # <--- Attach error bars here
        capsize=5,
        ecolor='black'
    )
    # Calculate percentages and add them inside the 'Neural Network' bars
    for i in range(n_groups):
        # Calculate the percentage
        percentage = (neural_cut[i] / heuristic_cut[i]) * 100 if heuristic_cut[i] != 0 else 0

        # Get the position and height of the 'Neural Network' bar
        x_pos = index[i] + bar_width
        y_pos = neural_cut[i]

        # Choose text color based on bar height for readability
        text_color = 'white' if y_pos > max(neural_cut) * 0.1 else 'black'

        # Place the percentage text inside the bar
        plt.text(
            x_pos,                      # X position
            y_pos / 2,                  # Y position (middle of the bar)
            f'{percentage:.0f}%',       # Text to display
            ha='center',                # Horizontal alignment
            va='center',                # Vertical alignment
            color=text_color,           # Text color
            fontsize=10,                # Font size
            fontweight='bold'           # Font weight
        )

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def create_line_graph(data_lists, x_labels, line_labels, y_lim = 60, index = 3):
    """
    Plots multiple lists on the same line graph, with one line per list.

    :param data_lists: A list (or tuple) of lists, each containing y-values to plot.
    :param x_labels:   A list of x-axis labels (must match the length of each data list).
    :param line_labels: A list of labels for each line (must match the number of data_lists).
    """
    # 1. Validate that the number of data lists matches the number of line labels
    if len(data_lists) != len(line_labels):
        raise ValueError("The number of data lists must match the number of line labels.")

    # 2. Validate that each data list has the same length as x_labels
    if not all(len(lst) == len(x_labels) for lst in data_lists):
        raise ValueError("All data lists must have the same length as x_labels.")

    # Create the line plot
    plt.figure(figsize=(10, 6))

    # Plot each list with the corresponding line label
    for i, data_list in enumerate(data_lists):
        # plt.plot(x_labels, data_list, label=line_labels[i], marker='o', alpha=0.8 if (i == 8 or i== 9) else 1)
        if i == index:
            plt.plot(
                x_labels,
                data_list,
                label=line_labels[i],
                marker='o',
                zorder=10,  # higher zorder => drawn on top
                alpha=0.9
            )
        else:
            plt.plot(
                x_labels,
                data_list,
                label=line_labels[i],
                marker='o',
                zorder=2,   # lower zorder => drawn behind
                alpha=1.0
            )

    # Set Y-axis limits and ticks (customize as needed)
    plt.ylim(y_lim, 100)
    plt.yticks(range(y_lim, 101, 10))

    # Add labels and title
    plt.xlabel('Graph Size (Nodes)')
    plt.ylabel('Accuracy')
    plt.title('Effect of the number of large graphs in the training set on scalability')
    plt.legend()

    plt.tight_layout()
    plt.show()