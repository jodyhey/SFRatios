"""
written by claude with this prompt:
I need to do a test of a hypothesis that the area under the curve of one cumulative 
distributions of counts for a series of bins, where the bins are indexed from 1 to k, 
is greater than the area under the curve for a second cumulative distribution that is 
also of discrete counts for bins 1 thru k.   The total counts are different,  so the test must account for this.
Does a one-sided test that the second SFS has greater AUC than the first. 
Reads an SF_Ratios.py input file, i.e. text on line 1,  neutral SFS on line 2 and selected SFS on line 3. Ignore the 0 bin. 

"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import sys

def getSFSs(fn):
    """
        data text file format: 
            line 1:  arbitrary text
            line 2:  neutral SFS beginning with 0 bin (which is ignored)
            line 3:  blank
            line 4:  selected SFS beginning with 0 bin (which is ignored)
    """
    lines = open(fn,'r').readlines()
    datafileheader = lines[0].strip()
    sfss = []
    for line in [lines[1],lines[3]]: # neutral, skip a line, then selected 
        if "." in line:
            # vals = list(map(float,line.strip().split()))
            # sfs = list(map(stochastic_round,vals))
            sfs = list(map(float,line.strip().split()))
        else:
            sfs = list(map(int,line.strip().split()))
        sfs[0] = 0
        sfss.append(sfs)       
    if len(sfss[0]) != len(sfss[1]) :
        print("Exception, Neutral and Selected SFSs are different lengths ")
        exit()
    return datafileheader,sfss[0][1:],sfss[1][1:] # skip the zero bin 

def compute_normalized_cdf(counts):
    """
    Convert bin counts to cumulative proportions
    """
    total = np.sum(counts)
    cumulative = np.cumsum(counts)
    return cumulative / total

def area_between_cdfs(cdf1, cdf2):
    """
    Calculate area between two CDFs using trapezoidal rule
    Returns both total area and area where cdf1 > cdf2
    """
    diff = cdf1 - cdf2
    # Area where cdf1 > cdf2
    positive_diff = np.maximum(diff, 0)
    positive_area = np.trapz(positive_diff)
    # Total absolute area
    total_area = np.trapz(np.abs(diff))
    return positive_area, total_area

def test_cdf_dominance(counts1, counts2, n_permutations=10000):
    """
    Test whether CDF1 is significantly greater than CDF2
    Uses permutation test approach
    
    Parameters:
    counts1, counts2: arrays of counts for each bin
    n_permutations: number of permutations for the test
    
    Returns:
    pvalue: p-value for the test
    observed_stat: observed area where cdf1 > cdf2
    null_stats: distribution of test statistics under null
    """
    # Convert to CDFs
    cdf1 = compute_normalized_cdf(counts1)
    cdf2 = compute_normalized_cdf(counts2)
    
    # Calculate observed statistic (area where cdf1 > cdf2)
    observed_stat, _ = area_between_cdfs(cdf1, cdf2)
    
    # Combine data for permutation test
    combined = np.concatenate([counts1, counts2])
    n1, n2 = len(counts1), len(counts2)
    
    # Permutation test
    null_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Randomly permute the combined data
        permuted = np.random.permutation(combined)
        # Split into two groups of original sizes
        perm1, perm2 = permuted[:n1], permuted[n1:]
        # Calculate CDFs
        perm_cdf1 = compute_normalized_cdf(perm1)
        perm_cdf2 = compute_normalized_cdf(perm2)
        # Calculate test statistic
        null_stats[i], _ = area_between_cdfs(perm_cdf1, perm_cdf2)
    
    # Calculate p-value (proportion of permuted stats >= observed)
    pvalue = np.mean(null_stats >= observed_stat)
    
    return pvalue, observed_stat, null_stats

def plot_cdfs_comparison(counts1, counts2, title="Comparison of CDFs"):
    """
    Plot both CDFs and their difference
    """
    cdf1 = compute_normalized_cdf(counts1)
    cdf2 = compute_normalized_cdf(counts2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot CDFs
    x = np.arange(len(counts1))
    ax1.plot(x, cdf1, label='Selected CDF', marker='o')
    ax1.plot(x, cdf2, label='Neutral CDF', marker='o')
    ax1.set_title(title)
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Cumulative Proportion')
    ax1.legend()
    ax1.grid(True)
    
    # Plot difference
    diff = cdf1 - cdf2
    ax2.plot(x, diff, color='purple', marker='o')
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.fill_between(x, diff, 0, where=(diff > 0), color='green', alpha=0.3, 
                     label='Area where SCDF > NCDF')
    ax2.fill_between(x, diff, 0, where=(diff < 0), color='red', alpha=0.3,
                     label='Area where SCDF > NCDF')
    ax2.set_title('Difference between CDFs (SCDF - NCDF)')
    ax2.set_xlabel('Bin Index')
    ax2.set_ylabel('Difference in Proportions')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def parsecommandline():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="sfsfilename",required=True,type = str, help="Path for SFS file")
    parser.add_argument("-l",dest="label",default = "", type=str, help="putput file label")    
    parser.add_argument("-o",dest="outdir",default = "", type=str, help="output directory")    
    args  =  parser.parse_args(sys.argv[1:]) 
    return args 
# Example usage:
if __name__ == "__main__":
    # Example data (replace with your actual data)
    args = parsecommandline()
    datafileheader,Ncounts,Scounts = getSFSs(args.sfsfilename)
    # counts1 = np.array([10, 20, 15, 8, 5])
    # counts2 = np.array([8, 15, 18, 12, 7])
    
    # Run test
    pvalue, observed_stat, null_stats = test_cdf_dominance(Ncounts, Scounts)
    
    # Print results
    print(f"Observed area where SCDF > NCDF2: {observed_stat:.4f}")
    print(f"P-value: {pvalue:.4f}")
    
    # Create and save plots
    fig = plot_cdfs_comparison(Scounts,Ncounts)
    plt.savefig('cdf_comparison.pdf')
    plt.close()
    
    # Plot null distribution
    plt.figure(figsize=(8, 6))
    plt.hist(null_stats, bins=50, density=True)
    plt.axvline(observed_stat, color='red', linestyle='--', 
                label=f'Observed statistic: {observed_stat:.4f}')
    plt.title('Null Distribution from Permutation Test')
    plt.xlabel('Test Statistic (Area where SCDF > NCDF)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('null_distribution.pdf')
    plt.close()