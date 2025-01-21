import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import sys
import os 
import argparse


def is_file_open(file_path):
    """
    Checks if the file exists and if it is open by another process.
    
    :param file_path: Path to the file to check.
    :return: Tuple (exists, is_open), where:
             - exists: True if the file exists, False otherwise.
             - is_open: True if the file is open by another process, False if not.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        return False, False
    
    # Check if the file is open by another process
    try:
        # Attempt to open the file in exclusive mode
        with open(file_path, "a+"):
            pass  # If we can open the file, it's not open by another process
        return True, False
    except IOError:
        # If an IOError occurs, the file is likely open by another process
        return True, True
    
def modify_plotfilename(file_path):
    """
    Modify the filename to add or increment a numeral (_#) before the extension.

    :param file_path: The full path of the file to modify.
    :return: The modified filename.
    """
    # Extract directory, filename, and extension
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)

    # Check if the base name already ends with _#
    if "_" in name and name.split("_")[-1].isdigit():
        # Increment the existing numeral
        parts = name.rsplit("_", 1)
        base_name = parts[0]
        number = int(parts[1]) + 1
        new_name = f"{base_name}_{number}"
    else:
        # Add a numeral suffix if none exists
        new_name = f"{name}_1"

    # Combine the new name with the extension and directory
    return os.path.join(directory, f"{new_name}{ext}")
    


def boxplot_2d(x, y, ax, color="blue", whis=1.5, includeoutliers=False):
    """Create a 2D boxplot preserving negative values"""
    x = np.array(x)
    y = np.array(y)
    
    # Debug output
    # print(f"\nDebug for {color} dataset:")
    # print(f"X range: {np.min(x):.2f} to {np.max(x):.2f}")
    # print(f"Y range: {np.min(y):.2f} to {np.max(y):.2f}")
    
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]
    
    # print(f"X quartiles: {xlimits}")
    # print(f"Y quartiles: {ylimits}")
    
    # Box - handle negative coordinates correctly
    box = Rectangle(
        (xlimits[0], ylimits[0]),  # Lower left corner
        xlimits[2]-xlimits[0],     # Width
        ylimits[2]-ylimits[0],     # Height
        ec="black",
        color=color,
        alpha=0.3,
        zorder=0
    )
    ax.add_patch(box)
    
    # Medians
    vline = Line2D(
        [xlimits[1], xlimits[1]], [ylimits[0], ylimits[2]],
        color=color,
        zorder=1
    )
    ax.add_line(vline)
    
    hline = Line2D(
        [xlimits[0], xlimits[2]], [ylimits[1], ylimits[1]],
        color=color,
        zorder=1
    )
    ax.add_line(hline)
    
    # Calculate whiskers
    iqr_x = xlimits[2]-xlimits[0]
    iqr_y = ylimits[2]-ylimits[0]
    
    # X whiskers
    left = np.min(x[x > xlimits[0]-whis*iqr_x])
    right = np.max(x[x < xlimits[2]+whis*iqr_x])
    
    # Y whiskers - handle negative values
    # bottom = np.min(y[y > ylimits[0]-whis*iqr_y])
    # top = np.max(y[y < ylimits[2]+whis*iqr_y])

    # Y whiskers - handle negative values
    y_mask = y > ylimits[0]-whis*iqr_y
    if np.any(y_mask):
        bottom = np.min(y[y_mask])
    else:
        bottom = ylimits[0]  # Use the lower quartile if no points found

    y_mask = y < ylimits[2]+whis*iqr_y
    if np.any(y_mask):
        top = np.max(y[y_mask])
    else:
        top = ylimits[2]  # Use the upper quartile if no points found
    
    # Draw whiskers - handle negative coordinates
    for xx, side in [(left, xlimits[0]), (right, xlimits[2])]:
        ax.add_line(Line2D([xx, side], [ylimits[1], ylimits[1]], color=color, zorder=1))
        ax.add_line(Line2D([xx, xx], [ylimits[0], ylimits[2]], color=color, zorder=1))
    
    for yy, side in [(bottom, ylimits[0]), (top, ylimits[2])]:
        ax.add_line(Line2D([xlimits[1], xlimits[1]], [yy, side], color=color, zorder=1))
        ax.add_line(Line2D([xlimits[0], xlimits[2]], [yy, yy], color=color, zorder=1))
    
    if includeoutliers:
        mask = (x<left)|(x>right)|(y<bottom)|(y>top)
        ax.scatter(x[mask], y[mask], facecolors='none', edgecolors=color)

def make2Dboxplot(alldata, truevals, xparamlabel, yparamlabel, filename, includeoutliers, ylimb=None, ylimt=None, xliml=None, xlimr=None,logx = None):
    colors = ["red","blue","green","cyan","magenta","black"]
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate global ranges including negative values
    all_y = []
    for data in alldata:
        all_y.extend(data[1])
    ymin, ymax = np.min(all_y), np.max(all_y)
    # print(f"\nOverall Y range: {ymin:.2f} to {ymax:.2f}")
    
    n = len(alldata)
    for i in range(n):
        
        x = alldata[i][0]
        y = alldata[i][1]
        # if args.fastdfegamma: # obsolete probably
        #     y = [v/-2 for v in y]
        truevalx = float(truevals[i][0])
        truevaly = float(truevals[i][1])
        # print(f"\nPlotting dataset {i+1}")
        # print(f"True values: x={truevalx}, y={truevaly}")
        
        color = colors[i]
        boxplot_2d(x, y, ax=ax, color=color, whis=1, includeoutliers=includeoutliers)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.plot(truevalx, truevaly, marker="o", markersize=10, markeredgecolor="black", markerfacecolor=color)
    
    plt.xlabel(xparamlabel, fontsize=14)
    plt.ylabel(yparamlabel, fontsize=14)
    # Set axis limits to show full range including negatives
    padding = 0.1
    if ylimb is not None:
        ax.set_ylim(bottom=ylimb)
    else:
        if ymin < 0:
            bottom_pad = abs(ymin) * padding 
            ax.set_ylim(bottom=ymin - bottom_pad)
        else:
            ax.set_ylim(bottom=0)

        
    if ylimt is not None:
        ax.set_ylim(top=ylimt)
    else:
        top_pad = abs(ymax) * padding
        ax.set_ylim(top=ymax + top_pad)
    
    if xliml is not None:
        ax.set_xlim(left=xliml)
    if xlimr is not None:
        ax.set_xlim(right=xlimr)
    
    # print(f"\nPlot Y limits: {ax.get_ylim()}")
    if logx:
        plt.xscale('symlog')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()    

    exists, is_open = is_file_open(filename)
    if is_open == False:
        if exists:
            if args.overwrite == False:
                filename = modify_plotfilename(filename)
    else:
        newfilename = modify_plotfilename(filename)
        print("filename is open in another application,  writing to {}".format(newfilename))
        filename = newfilename
    plt.savefig(filename)
    return

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",dest="overwrite",action="store_true",help="overwrite plot file if it exists")
    parser.add_argument("-f",dest="fname",required=True,type=str,help="filename")
    parser.add_argument("-d",dest="densitymodel",default=None,type=str,help="density model")
    parser.add_argument("-j",dest="plotjpeg",action="store_true",help="make jpeg plot, default is pdf")
    parser.add_argument("-o",dest="includeoutliers",action="store_true",default=False,help="include outliers")
    parser.add_argument("-b",dest="ylimb",default=None,type=float,help="ylim bottom")
    parser.add_argument("-t",dest="ylimt",default=None,type=float,help="ylim top")
    parser.add_argument("-l",dest="xliml",default=None,type=float,help="xlim left")
    parser.add_argument("-r",dest="xlimr",default=None,type=float,help="xlim right")
    parser.add_argument("-x",dest="logx",action="store_true",help="use log scale for x axis")
    # parser.add_argument("-z",dest="fastdfegamma",action="store_true",help="if fastdfe gamma model")
    args = parser.parse_args(sys.argv[1:])
    args.commandstring = " ".join(sys.argv[1:])
    return args

if __name__ == '__main__':
    args = parsecommandline()
    includeoutliers = args.includeoutliers
    if args.plotjpeg:
        plotfname = args.fname[:-4] + "_with_outliers_alt2dplot.pdf" if includeoutliers else args.fname[:-4] + "_alt2dplot.jpeg"
    else:
        plotfname = args.fname[:-4] + "_with_outliers_alt2dplot.pdf" if includeoutliers else args.fname[:-4] + "_alt2dplot.pdf"
    
    # Read and process input file preserving negative values
    with open(args.fname, 'r') as gf:
        allgs = []
        truevals = []
        while True:
            line1 = gf.readline().strip().split()
            if len(line1) <= 1:
                break
                
            truevals.append([float(line1[0])])
            # Convert all values to float maintaining signs
            g1s = [float(x) for x in line1[1:]]
            
            line2 = gf.readline().strip().split()
            truevals[-1].append(float(line2[0]))
            g2s = [float(x) for x in line2[1:]]
            
            allgs.append([g1s, g2s])
            gf.readline()  # Skip empty line
    if args.densitymodel == 'lognormal' or 'lognormal' in args.fname[:-4].split('_'):
        make2Dboxplot(allgs, truevals, r'$\mu$', r'$\sigma$', plotfname, includeoutliers, 
                     ylimb=args.ylimb, ylimt=args.ylimt, xliml=args.xliml, xlimr=args.xlimr, logx = args.logx)
    elif args.densitymodel == "gamma" or 'gamma' in args.fname[:-4].split('_'):
        make2Dboxplot(allgs, truevals, r'mean', r'shape', plotfname, includeoutliers,
                     ylimb=args.ylimb, ylimt=args.ylimt, xliml=args.xliml, xlimr=args.xlimr,logx = args.logx)