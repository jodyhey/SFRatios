"""
reads a text file with SNP allele counts formatted like this:
    167	142
    162	11
    165	133
    165	164
    164	21
    163	33
    165	77
    162	20
    165	113

the first column is the total allele count and the second is the count of derived alleles 
return an SFS with at least 
"""
import sys
import argparse
import os.path as op
import numpy as np 
import random
from scipy.stats import hypergeom


def stochastic_round(number):
    floor_number = int(number)
    # Probability of rounding up is the fractional part of the number
    if np.random.uniform() < (number - floor_number):
        return floor_number + 1
    else:
        return floor_number
    
def readfile(infilename):
    snpcounts = {}
    minnc = float('inf')
    with open(infilename, "r") as f:
        for line in f:
            if len(line) > 2:
                l = line.strip().split()
                if minnc > int(l[0]):
                    minnc = int(l[0])
                snc = int(l[0])
                sa = int(l[1])
                if minnc > snc:
                    minnc = snc
                if snc not in snpcounts:
                    snpcounts[snc] = {}
                if sa not in snpcounts[snc]:
                    snpcounts[snc][sa] = 0
                snpcounts[snc][sa] += 1
    return snpcounts,minnc

def make_SFS_with_downsampling(args,snpcounts):
    sfs = [0 for i in range(nc+1)]
    for snc in snpcounts:
        for sa in snpcounts[snc]:
            pi = sa
            numg = snc
            for si in range(args.nc+1):
                prob = hypergeom.pmf(si,numg,pi,args.nc)
                sfs[si] += prob*snpcounts[snc][sa]
    if args.dostochasticround:
        sfs = list(map(stochastic_round,sfs))
    if args.fixedbin == False:
        return sfs[:-1]
    else:
        return sfs

def make_SFS_with_subsampling(args,snpcounts):
    sfs = [0 for i in range(nc+1)]
    for snc in snpcounts:
        for sa in snpcounts[snc]:
            pi = sa
            numg = snc    
            samples = np.random.hypergeometric(pi,numg-pi,args.nc,snpcounts[snc][sa])
            for c in samples:
                sfs[c] += 1
    if args.fixedbin == False:
        return sfs[:-1]
    else:
        return sfs

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",dest="dostochasticround",default = False,action="store_true",help = "if downsampling (-d) apply stochastic rounding to get an integer")
    parser.add_argument("-d",dest="downsample",action="store_true", default = False,help=" use hypergeometic downsampling")    
    parser.add_argument("-e",dest="seed",type = int, default = 1,help=" random number seed for sub-sampling and misspecification")    
    parser.add_argument("-f",dest="foldit",action="store_true",help="fold the resulting SFS")    
    parser.add_argument("-i",dest="infilename",required=True,type = str, help="input file path/name")  
    parser.add_argument("-n",dest="nc",type = int, default = None,help=" new sample size")   
    parser.add_argument("-o", dest="sfsfilename",required=True,type = str, help="Path for output SFS file")
    parser.add_argument("-s",dest="subsample",action="store_true", default =False,help=" use random subsampling")   
    parser.add_argument("-y",dest="fixedbin",default = False,action="store_true",help=" SFSs file includes fixed sites,  i.e. last value is count of # of fixed sites") 
    parser.add_argument("-z",dest="nozerobin",default = False,action="store_true",help=" SFSs in file begin at count 1,  not 0 ") 
    
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    if args.downsample is True and args.subsample is True:
        parser.error(' cannot do both -d and -s')
    if args.downsample is False and args.subsample is False:
        parser.error(' must use either -d or -s')
    if args.downsample is False  and args.dostochasticround is True:
        parser.error(' cannot do -c without -d')
    return args

if __name__ == '__main__':
    args = parsecommandline()
    random.seed(args.seed)
    np.random.seed(args.seed)
    snpcounts,nc = readfile(args.infilename)
    if args.nc == None:
        args.nc = nc
    if args.downsample:
        sfs = make_SFS_with_downsampling(args,snpcounts)
    elif args.subsample:
        sfs = make_SFS_with_subsampling(args,snpcounts)

    nciseven = args.nc % 2 == 0
    if args.foldit:
        if nciseven:
            fsfs = [0] + [sfs[j]+sfs[args.nc-j] for j in range(1,args.nc//2)] + [sfs[args.nc//2]]
        else:
            fsfs = [0] + [sfs[j]+sfs[args.nc-j] for j in range(1,1 + args.nc//2)] 
        if args.fixedbin: # fold bins 0 and -1
          fsfs[0] = sfs[0] + sfs[-1]
        else:
          fsfs[0] = sfs[0]
        sfs = fsfs
    if args.nozerobin:
        sfs = sfs[1:]
    fout = open(args.sfsfilename,'w')
    fout.write("make_sfs_from_SNP_allele_counts.py output,  command line {}:\n".format(args.commandstring))
    # print(sfs)
    if args.downsample and args.dostochasticround == False:  # values are floats
        sfsline = " ".join(f"{x:.1f}" for x in sfs)
    else:
        sfsline = " ".join(map(str,sfs))
    fout.write(sfsline + "\n")
    fout.close()

