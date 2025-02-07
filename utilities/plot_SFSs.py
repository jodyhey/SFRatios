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
    
    return headers, SFSs


def calculate_proportional_cumulative_sum(numbers):
    cumsum = np.cumsum(numbers)
    return cumsum / cumsum[-1]

def calculate_custom_sum(datai,numbers, xaxislowerlimit, cumulative, proportional):
    if xaxislowerlimit > 1:
        numbers = numbers[xaxislowerlimit-1:]    
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
        if proportional:
            return np.array(numbers) / numbers[0]
        else:
            return np.array(numbers)

def plot_data(data, labels, args,ksresults=None):

    if len(labels) < len(data):
        print("problem,  len(labels) != len(data)")
        print(" was -b invoked ?")
        print("headers :",headers)
        exit()
        # for j in range(len(labels),len(data)):
        #     labels.append("dataset_{}".format(j))
    if ksresults:
        labels = [l + ksresults[i] for i,l in enumerate(labels)]
    plt.rcParams.update({'font.size': 15})  # Set default font size to 15
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define a list of colors and line styles
    colors = [
        "#000000",  # Black
        "#FF1493",  # Deep Pink
        "#0000CC",  # Strong Dark Blue
        "#FF7F00",  # organgs
        # "#FFD700",  # Gold
        "#4CAF50",  # Green
        "#9C27B0",  # Purple
        "#FF5722",  # Deep Orange
        "#00BCD4"  # Cyan
        
    ]

    line_styles = [
        '-',            # solid
        '--',           # dashed
        (0, (8,2)),  # Even longer dashes
        '-.',           # dash-dot
        (0, (3, 1)),   # more densely dashed
        ':',            # dotted
        (0, (7, 3)),   # more sparsely dashed
        (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
        (0, (1, 2))    # densely dotted
    ]    
    # colors = [
    #     "#000000",  # Black
    #     "#FF1493",  # Deep Pink
    #     "#00BCD4",  # Cyan
    #     "#FFD700",  # Gold
    #     "#4CAF50",  # Green
    #     "#9C27B0",  # Purple
    #     "#FF5722",  # Deep Orange
    #     "#2196F3"   # Blue
    # ]
    # # line_styles = ['-', '--', ':', '-.']
    # line_styles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]

    # Sort labels alphabetically and rearrange data accordingly
    sorted_indices = sorted(range(len(labels)), key=lambda x: labels[x])
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_data = [data[i] for i in sorted_indices]

    for i, (numbers, label) in enumerate(zip(sorted_data, sorted_labels)):
        plotvals = calculate_custom_sum(i,numbers, args.xaxislowerlimit, args.plotcumulative, args.plotproportional)
        x = range(args.xaxislowerlimit, len(plotvals) + args.xaxislowerlimit)
        
        # Always cycle through colors and line styles in the same order
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        ax.plot(x, plotvals, label=label, color=color, linestyle=line_style,linewidth=3)

    # Set the labels and title
    ax.set_xlabel('Index', fontsize=15)
    if args.plotcumulative:
        if args.plotproportional:
            if args.yaxislimit is not None:
                ax.set_ylim(args.yaxislimit, 1.001)
            ax.set_ylabel('Proportional Cumulative Sum', fontsize=15)
        else:
            ax.set_ylabel('Cumulative Sum', fontsize=15)
    else:
        if args.plotproportional:
            if args.yaxislimit is not None:
                ax.set_ylim(0.0,args.yaxislimit)
            ax.set_ylabel("Proportional to lowest frequency bin", fontsize=15)
        else:
            if args.yaxislimit is not None:
                ax.set_ylim(0.0,args.yaxislimit)
            ax.set_title('SNP Count', fontsize=15)    
    # Additional plot settings
    if args.plotcumulative:
        ax.legend(loc='lower right', fontsize=15, frameon=True)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
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
        for i in range(round(round(count))):
            d2.append(val)            
    res = ks_2samp(d1,d2,alternative='greater' if args.KStests==1 else "two-sided")
    return res.pvalue, res.statistic

def parsecommandline():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s",dest="sfsfilepath",type = str, required=True,help="path and filename for SFSs")
    parser.add_argument("-o",dest="plotfilepath",type = str, required=True,help="path and filename for plot figure")
    parser.add_argument("-y",dest="yaxislimit",type = float, default = None,help="if '-m ' y axis lower limit, else upper limit")
    parser.add_argument("-x",dest="xaxislowerlimit",type = int, default = 1,help="lowest x axis bin to include, default = 1")
    parser.add_argument("-f",dest="foldit",action="store_true",default=False,help="fold the sfss")
    parser.add_argument("-b",dest="userheaderlabels",action="store_true",default=False,help="user the text in the file as the plot legend text")
    parser.add_argument("-m",dest="plotcumulative",action="store_true",default=False,help="plot the cumulative SFS,  default is regular")
    parser.add_argument("-r",dest="plotproportional",action="store_true",default=False,help="plot the SFS, whether reg or cumulative, proportional to the lowest bin, default is regular")
    parser.add_argument("-w",dest="plot_to_screen",action="store_true",default=False,help="show the plot on the screen")
    parser.add_argument("-L",dest="labels",nargs="+",default= [],help="a series of labels, typically the same number as the number SFSs in the sfs file")
    parser.add_argument("-k",dest="KStests",type = int, default = 0,help="do Kolmogorov-Smirnov test, with intergenic as neutral, -k 1 one sided -k 2 two sided,default = 0")
    parser.add_argument("-a",dest="neutrallabel",default=None,help = "if using -k, the neutral SFS label")
    parser.add_argument("-g", dest="gridlines", action="store_true", default=False, help="add gridlines")
        

    args  =  parser.parse_args(sys.argv[1:])   
 
    args.commandstring = " ".join(sys.argv[1:])

    return args

    # return parser

if __name__ == '__main__':
    """

    """
    args = parsecommandline()
    headers, data = readSFS(args.sfsfilepath,args.foldit)
    if args.userheaderlabels:
        args.labels = headers
    if args.KStests:
        ni = args.labels.index(args.neutrallabel)
        ksresults = []
        for di,d in enumerate(data):
            if di == ni:
                ksresults.append("")
            else:
                p,stat = kstest(d[1:],data[ni][1:], alternative='greater' if args.KStests==1 else "two-sided")
                ksresults.append(", p={:.3g}".format(p))

        plot_data(data, args.labels,args,ksresults=ksresults)    
    else:
        plot_data(data, args.labels,args,ksresults=None)    
