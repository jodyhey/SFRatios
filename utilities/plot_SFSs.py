"""
    make SFS plot(s), cumulative and regular,  of SFSs built by downsample_from_vcf_consequence_pickle.py
    or the parallel version of that  program 
usage: plot_SFSs.py [-h] [-a] [-b] [-e NEUTRALLABEL] [-f] [-g] [-k KSTESTS] [-l] [-L LABELS [LABELS ...]] [-m] -o PLOTFILEPATH [-r] -s
                    SFSFILEPATH [-u XAXISUPPERLIMIT] [-w] [-x XAXISLOWERLIMIT] [-y YAXISLIMIT]

options:
  -h, --help            show this help message and exit
  -a                    Use alternate plotting: pairs share colors, first gets pattern, second gets solid line
  -b                    Use the text in the file as the plot legend text
  -e NEUTRALLABEL       If using -k, the neutral SFS label
  -f                    Fold the SFSs
  -g                    Add gridlines
  -k KSTESTS            Do Kolmogorov-Smirnov test, with intergenic as neutral, -k 1 one sided -k 2 two sided, default = 0
  -l                    Plot log of SFS, 0's are skipped, does not work with -r
  -L LABELS [LABELS ...]
                        A series of labels, typically the same number as the number SFSs in the sfs file
  -m                    Plot the cumulative SFS, default is regular
  -o PLOTFILEPATH       Path and filename for plot figure
  -r                    Plot the SFS, whether reg or cumulative, proportional to the lowest bin, default is regular
  -s SFSFILEPATH        Path and filename for SFSs
  -u XAXISUPPERLIMIT    Highest x axis bin to include, default = None
  -w                    Show the plot on the screen
  -x XAXISLOWERLIMIT    Lowest x axis bin to include (can be 0 to include invariant sites), default = 1
  -y YAXISLIMIT         If '-m ' y axis lower limit, else upper limit

"""
"""
    make SFS plot(s), cumulative and regular,  of SFSs built by downsample_from_vcf_consequence_pickle.py
    or the parallel version of that  program 
    
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse 
import os.path as op
import itertools
from  scipy.stats import ks_2samp
import math


def readSFS(fn, foldit):
    """
    Reads a file containing headers and SFS data in alternating lines.
    There may be additional nondata lines,  e.g. line 0
    If a nondata line begins with a digit, there will be a problem

    Headers are any non-numeric lines.
    SFS data are space-separated numbers (integer or float) on the next line after each header.
    All SFSs must be the same length
    
    Parameters:
        fn (str): Filename to read
        foldit (bool): Whether to fold the SFS
        
    Returns:
        tuple: (headers, SFSs) where:
            headers (list): List of header strings
            SFSs (list): List of SFS lists (numeric data)
    """
    with open(fn, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines and whitespace
    
    headers = []
    SFSs = []
    sums = []
    ncall = None  # Initialize ncall
    i = 0
    
    while i < len(lines):
        # Check if line starts with a number
        if not lines[i][0].isdigit():
            i += 1
            continue
            
        # Found data
        sfs = list(map(float, lines[i].split()))
        if ncall is None:  # First SFS sets the expected length
            ncall = len(sfs)
        else:
            assert len(sfs) == ncall, f"SFS length mismatch: {len(sfs)} != {ncall}"
        
        if foldit:
            nc = len(sfs)
            if nc % 2 == 0:  # even length
                sfs = [sfs[0]] + [sfs[k] + sfs[nc-k] for k in range(1, nc//2)] + [sfs[nc//2]]
            else:  # odd length
                sfs = [sfs[0]] + [sfs[k] + sfs[nc-k] for k in range(1, 1 + nc//2)]
        sums.append(sum(sfs))        
        SFSs.append(sfs)        
        headers.append(lines[i-1].strip())
        i += 1
    assert len(headers) == len(SFSs), "Mismatch between number of headers and SFS counts"
    i = len(SFSs) -1
    while i >= 0:
        if sum(SFSs[i]) <= 0:
            print("No SNPs in SFS : ",headers[i])
            SFSs.pop(i)
            headers.pop(i)
        i -= 1
    
    return headers, SFSs, sums


def calculate_proportional_cumulative_sum(numbers):
    cumsum = np.cumsum(numbers)
    return cumsum / cumsum[-1]

def calculate_custom_sum(datai,numbers, xaxislowerlimit, xaxisupperlimit,cumulative, proportional):
    if xaxislowerlimit > 1:
        numbers = numbers[xaxislowerlimit-1:]    
    if xaxisupperlimit is not None:
        numbers = numbers[:xaxisupperlimit]
    if cumulative:
        result = np.cumsum(numbers)
        if proportional:
            if result[-1] <= 0.0:
                print("problem plotting data set ",datai, "result[-1] ", result[-1])
                print(numbers)
                return 0 
            else:
                return result/result[-1]
        else:
            return result
    else:
        # numbers = numbers[1:]
        numbers = numbers[xaxislowerlimit:]
        if proportional:
            return np.array(numbers) / numbers[0]
        else:
            return np.array(numbers)

def plot_data(data, counts, labels, args,ksresults=None):

    if len(labels) < len(data):
        print("problem,  len(labels) != len(data)")
        print(" was -b invoked ?")
        print("headers :",labels)
        exit()
        # for j in range(len(labels),len(data)):
        #     labels.append("dataset_{}".format(j))
    if ksresults:
        labels = ["{} ({}){}".format(l,round(counts[i]),ksresults[i]) for i,l in enumerate(labels)]
    plt.rcParams.update({'font.size': 15})  # Set default font size to 15
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors and line styles
    colors = [
        "#000000",  # Black
        "#FF1493",  # Deep Pink
        "#0000CC",  # Strong Dark Blue
        "#FF7F00",  # Orange
        "#4CAF50",  # Green
        "#9C27B0",  # Purple
        "#FF5722",  # Deep Orange
        "#00BCD4"  # Cyan
    ]

    if args.alternate_plotline:
        # For alternate plotting: each pair gets same color, different styles
        pair_line_styles = [
            '--',           # dashed
            (0, (8,2)),     # Even longer dashes
            '-.',           # dash-dot
            (0, (3, 1)),    # more densely dashed
            ':',            # dotted
            (0, (7, 3)),    # more sparsely dashed
            (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
            (0, (1, 2))     # densely dotted
        ]
    else:
        # Original line styles
        line_styles = [
            '-',            # solid
            '--',           # dashed
            (0, (8,2)),     # Even longer dashes
            '-.',           # dash-dot
            (0, (3, 1)),    # more densely dashed
            ':',            # dotted
            (0, (7, 3)),    # more sparsely dashed
            (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
            (0, (1, 2))     # densely dotted
        ]

    # Keep original order from input file
    sorted_labels = labels
    sorted_data = data

    for i, (numbers, label) in enumerate(zip(sorted_data, sorted_labels)):
        plotvals = calculate_custom_sum(i,numbers, args.xaxislowerlimit,args.xaxisupperlimit, args.plotcumulative, args.plotproportional)
        x = range(args.xaxislowerlimit, len(plotvals) + args.xaxislowerlimit)
        
        if args.alternate_plotline:
            # Use the current index for pairing logic (since we're not sorting anymore)
            pair_index = i // 2
            is_second_in_pair = i % 2 == 1
            
            color = colors[pair_index % len(colors)]
            if is_second_in_pair:
                line_style = '-'  # solid line for second in pair
            else:
                line_style = pair_line_styles[pair_index % len(pair_line_styles)]
        else:
            # Original behavior: cycle through colors and line styles in original order
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]

        if args.plotlogsfs:
            # Filter out values where plotvals[i] <= 0
            x_log = [x[i] for i in range(len(plotvals)) if plotvals[i] > 0]
            y_log = [math.log(plotvals[i]) for i in range(len(plotvals)) if plotvals[i] > 0]
            ax.plot(x_log, y_log, label=label, color=color, linestyle=line_style, linewidth=3)
        else:
            ax.plot(x, plotvals, label=label, color=color, linestyle=line_style, linewidth=3)

    # Set the labels and title
    ax.set_xlabel('Index', fontsize=15)
    if args.plotcumulative:
        if args.plotproportional:
            if args.yaxislimit is not None:
                ax.set_ylim(args.yaxislimit, 1.001)
            ax.set_ylabel('Proportional Cumulative Sum', fontsize=15)
        else:
            if args.plotlogsfs:
                ax.set_ylabel('Log Cumulative Sum', fontsize=15)
            else:
                ax.set_ylabel('Cumulative Sum', fontsize=15)
    else:
        if args.plotproportional:
            if args.yaxislimit is not None:
                ax.set_ylim(0.0,args.yaxislimit)
            ax.set_ylabel("Proportional to lowest frequency bin", fontsize=15)
        else:
            if args.plotlogsfs:
                ax.set_ylabel('Log Count', fontsize=15)
            else:
                ax.set_ylabel('Count', fontsize=15)
            if args.yaxislimit is not None:
                ax.set_ylim(0.0,args.yaxislimit)
            ax.set_title('SNP Count', fontsize=15)    

    # Add legend with smaller font size and longer line samples
    if sorted_labels:
        ax.legend(loc='best', fontsize=12, frameon=True, handlelength=3)

    # Set tick marks
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Set grid lines
    if args.gridlines:
        ax.grid(True, which='major', linestyle='-', alpha=0.2)

    # Save the figure
    plt.savefig(args.plotfilepath, dpi=300, bbox_inches='tight')
    
    # Show the plot if necessary
    if args.plot_to_screen:
        plt.show()

def kstest(counts1,counts2,alternative='greater'):
    d1 = []
    for ci,count in enumerate(counts1):
        val = ci + 1
        for i in range(round(count)):
            d1.append(val)
    d2 = []
    for ci,count in enumerate(counts2):
        val = ci + 1
        for i in range(round(count)):
            d2.append(val)            
    res = ks_2samp(d1,d2,alternative='greater' if args.KStests==1 else "two-sided")
    
    # Determine direction by comparing cumulative distributions at midpoint
    cum1 = np.cumsum(counts1)
    cum2 = np.cumsum(counts2)
    # Normalize to proportions
    cum1_norm = cum1 / cum1[-1] if cum1[-1] > 0 else cum1
    cum2_norm = cum2 / cum2[-1] if cum2[-1] > 0 else cum2
    
    # Compare at midpoint
    midpoint = len(cum1_norm) // 2
    if midpoint < len(cum1_norm) and midpoint < len(cum2_norm):
        if cum1_norm[midpoint] > cum2_norm[midpoint]:
            direction = "sel>neut@low"  # Selected has more low-frequency variants
        elif cum1_norm[midpoint] < cum2_norm[midpoint]:
            direction = "sel<neut@low"  # Selected has fewer low-frequency variants  
        else:
            direction = "sel≈neut@low"  # Similar at midpoint
    else:
        direction = "unclear"
    
    return res.pvalue, res.statistic, direction

def parsecommandline():
    parser = argparse.ArgumentParser()
    
    # Alphabetized argparse options
    parser.add_argument("-a", dest="alternate_plotline", action="store_true", default=False,
                       help="Use alternate plotting: pairs share colors, first gets pattern, second gets solid line")
    parser.add_argument("-b", dest="useheaderlabels", action="store_true", default=False,
                       help="Use the text in the file as the plot legend text")
    parser.add_argument("-e", dest="neutrallabel", default=None,
                       help="If using -k,and a specific SFS is the neutral control, give the neutral SFS label, if None, assume alternating selected, neutral")
    parser.add_argument("-f", dest="foldit", action="store_true", default=False,
                       help="Fold the SFSs")
    parser.add_argument("-g", dest="gridlines", action="store_true", default=False,
                       help="Add gridlines")
    parser.add_argument("-k", dest="KStests", type=int, default=0,
                       help="Do Kolmogorov-Smirnov test, with intergenic as neutral, -k 1 one sided -k 2 two sided, default = 0, does not work with -r")
    parser.add_argument("-l", dest="plotlogsfs", action="store_true", default=False,
                       help="Plot log of SFS, 0's are skipped, does not work with -r")
    parser.add_argument("-L", dest="labels", nargs="+", default=[],
                       help="A series of labels, typically the same number as the number of SFSs in the sfs file")
    parser.add_argument("-m", dest="plotcumulative", action="store_true", default=False,
                       help="Plot the cumulative SFS, default is regular")
    parser.add_argument("-o", dest="plotfilepath", type=str, required=True,
                       help="Path and filename for plot figure")
    parser.add_argument("-r", dest="plotproportional", action="store_true", default=False,
                       help="Plot the SFS, whether reg or cumulative, proportional to the lowest bin, default is regular")
    parser.add_argument("-s", dest="sfsfilepath", type=str, required=True,
                       help="Path and filename for SFSs")
    parser.add_argument("-u", dest="xaxisupperlimit", type=int, default=None,
                       help="Highest x axis bin to include, default = None")
    parser.add_argument("-w", dest="plot_to_screen", action="store_true", default=False,
                       help="Show the plot on the screen")
    parser.add_argument("-x", dest="xaxislowerlimit", type=int, default=1,
                       help="Lowest x axis bin to include (can be 0 to include invariant sites), default = 1")
    parser.add_argument("-y", dest="yaxislimit", type=float, default=None,
                       help="If '-m ' y axis lower limit, else upper limit")

    args = parser.parse_args(sys.argv[1:])   
    args.commandstring = " ".join(sys.argv[1:])
    return args

if __name__ == '__main__':
    """

    """
    args = parsecommandline()
    headers, data, counts = readSFS(args.sfsfilepath,args.foldit)
    if args.useheaderlabels:
        args.labels = headers
    if args.KStests:
        if args.neutrallabel is None:
            # Alternating pairs mode: second, fourth, sixth... rows are neutral controls
            # First check that we have an even number of datasets
            if len(data) % 2 != 0:
                print(f"Error: When using -k without -e, expecting alternating pairs.")
                print(f"Found {len(data)} datasets, but need an even number for pairs.")
                print(f"Available labels: {args.labels}")
                sys.exit(1)
            
            print(f"Using alternating pairs mode: {len(data)//2} selected-neutral pairs")
            
            ksresults = []
            for di in range(len(data)):
                if di % 2 == 0:
                    # This is a selected dataset (even indices: 0, 2, 4, ...)
                    # Its neutral control is at di + 1
                    neutral_index = di + 1
                    if neutral_index < len(data):
                        p, stat, direction = kstest(data[di][1:], data[neutral_index][1:], 
                                       alternative='greater' if args.KStests==1 else "two-sided")
                        ksresults.append(", p={:.3g}, {}".format(p, direction))
                    else:
                        print(f"Error: Selected dataset at index {di} has no corresponding neutral control")
                        sys.exit(1)
                else:
                    # This is a neutral dataset (odd indices: 1, 3, 5, ...)
                    ksresults.append("")  # No p-value for neutral datasets
        else:
            # Single neutral mode: find the specified neutral label
            if args.useheaderlabels:
                # If using header labels, find neutral in headers which matches data order
                try:
                    ni = headers.index(args.neutrallabel)
                except ValueError:
                    print(f"Error: Neutral label '{args.neutrallabel}' not found in data headers")
                    print(f"Available labels: {headers}")
                    sys.exit(1)
            else:
                # If using custom labels, find neutral in args.labels
                try:
                    ni = args.labels.index(args.neutrallabel)
                except ValueError:
                    print(f"Error: Neutral label '{args.neutrallabel}' not found in provided labels")
                    print(f"Available labels: {args.labels}")
                    sys.exit(1)
            
            ksresults = []
            for di,d in enumerate(data):
                if di == ni:
                    ksresults.append("")
                else:
                    p, stat, direction = kstest(d[1:],data[ni][1:], alternative='greater' if args.KStests==1 else "two-sided")
                    ksresults.append(", p={:.3g}, {}".format(p, direction))

        plot_data(data, counts, args.labels,args,ksresults=ksresults)    
    else:
        plot_data(data,counts, args.labels,args,ksresults=None)
