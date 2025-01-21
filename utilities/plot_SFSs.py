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

def readSFS(fn,foldit):
    """
        data text file format, alternating row of text or empty line followed by row of SFS,  SFS includes 0 bin: 
            for example, and SFRatios.py input file looks like:
                line 1:  arbitrary text
                line 2:  neutral SFS beginning with 0 bin (which is ignored)
                line 3:  blank
                line 4:  selected SFS beginning with 0 bin (which is ignored)

    """
    lines = open(fn,'r').readlines()
    while len(lines[-1]) < 2:
        lines = lines[:-1]
    assert len(lines) % 2 == 0
    headers = []
    SFSs  = []
    for i in range(0,len(lines),2):
        headers.append(lines[i].strip())
        if '.' in lines[i+1]:
            sfs = list(map(float,lines[i+1].strip().split()))
        else:
            sfs = list(map(int,lines[i+1].strip().split()))
        nc = len(sfs)
        if i == 0:
            ncall = nc
        else:
            assert ncall == nc # all sfss should be same length 
        if foldit:  
            
            nciseven = nc % 2 ==0
            if nciseven:
                sfs = sfs[0] + [sfs[j]+sfs[nc-j] for j in range(1,nc//2)] + [sfs[nc//2]]
            else:
                sfs = sfs[0] + [sfs[j]+sfs[nc-j] for j in range(1,1 + nc//2)] 
        SFSs.append(sfs)
    return headers,SFSs 

def calculate_proportional_cumulative_sum(numbers):
    cumsum = np.cumsum(numbers)
    return cumsum / cumsum[-1]

def calculate_custom_sum(numbers, xaxislowerlimit, cumulative, proportional):
    if xaxislowerlimit > 1:
        numbers = numbers[xaxislowerlimit-1:]    
    if cumulative:
        result = np.cumsum(numbers)
        if proportional:
            return result / result[-1]
        else:
            return result
    else:
        if proportional:
            return np.array(numbers) / numbers[0]
        else:
            return np.array(numbers)

def plot_data(data, labels, args):

    if len(labels) < len(data):
        for j in range(len(labels),len(data)):
            labels.append("dataset_{}".format(j))
    plt.rcParams.update({'font.size': 15})  # Set default font size to 15
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define a list of colors and line styles
    colors = [
        "#000000",  # Black
        "#FF1493",  # Deep Pink
        "#00CED1",  # Dark Turquoise
        "#FFD700",  # Gold
        "#32CD32",  # Lime Green
        "#800080",  # Purple
        "#FF4500",  # Orange Red
        "#1E90FF"   # Dodger Blue
    ]
    # line_styles = ['-', '--', ':', '-.']
    line_styles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]

    # Sort labels alphabetically and rearrange data accordingly
    sorted_indices = sorted(range(len(labels)), key=lambda x: labels[x])
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_data = [data[i] for i in sorted_indices]

    for i, (numbers, label) in enumerate(zip(sorted_data, sorted_labels)):
        plotvals = calculate_custom_sum(numbers, args.xaxislowerlimit, args.plotcumulative, args.plotproportional)
        x = range(args.xaxislowerlimit, len(plotvals) + args.xaxislowerlimit)
        
        # Always cycle through colors and line styles in the same order
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        ax.plot(x, plotvals, label=label, color=color, linestyle=line_style)

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
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Save the figure
    plt.savefig(args.plotfilepath, dpi=300, bbox_inches='tight')
    
    # Show the plot if necessary
    if args.plot_to_screen:
        plt.show()


def parsecommandline():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s",dest="sfsfilepath",type = str, required=True,help="filename for SFSs")
    parser.add_argument("-o",dest="plotfilepath",type = str, required=True,help="plot file path")
    parser.add_argument("-y",dest="yaxislimit",type = float, default = None,help="if '-m ' y axis lower limit, else upper limit")
    parser.add_argument("-x",dest="xaxislowerlimit",type = int, default = 1,help="lowest x axis bin to include, default = 1")
    parser.add_argument("-f",dest="foldit",action="store_true",default=False,help="fold the sfss")
    parser.add_argument("-b",dest="userheaderlabels",action="store_true",default=False,help="user the text in the file as the plot legend text")
    parser.add_argument("-m",dest="plotcumulative",action="store_true",default=False,help="plot the cumulative SFS,  default is regular")
    parser.add_argument("-r",dest="plotproportional",action="store_true",default=False,help="plot the SFS, whether reg or cumulative, proportional to the lowest bin, default is regular")
    parser.add_argument("-w",dest="plot_to_screen",action="store_true",default=False,help="show the plot on the screen")
    parser.add_argument("-L",dest="labels",nargs="+",default= [],help="a series of labels, typically the same number as the number SFSs in the sfs file")

        

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
    plot_data(data, args.labels,args)    
