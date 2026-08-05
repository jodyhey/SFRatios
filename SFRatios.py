"""
Program:  SFRatios.py
Author: Jody Hey

reads a file with SFSs 
    data text file format: 
        line 1:  arbitrary text
        line 2:  neutral SFS beginning with 0 bin (which is ignored)
        line 3:  blank
        line 4:  selected SFS beginning with 0 bin (which is ignored)

models (-d) :
    fixed2Ns :  single  value of 2Ns
    gamma : flipped gamma distribution,  max set by -m  (e.g. 0 or 1), or estimated (-t)
    lognormal : flipped lognormal distribution,  max set by -m  (e.g. 0 or 1), or estimated (-t)
    normal : regular gaussian 

    model additions: -m (default=0, gamma, lognormal), -t (gamma, lognormal), (-z gamma, lognormal,normal,fixed2Ns) 

usage: SFRatios.py [-h] -a SFSFILENAME [-c FIX_THETA_RATIO] [-d DENSITYOF2NS] [-g] -f FOLDSTATUS [-m SETMAX2NS] [-M MAXI] [-p POPLABEL] [-t] [-r OUTDIR] [-x] [-z] [-Q THETARATIORANGE [THETARATIORANGE ...]]

options:
  -h, --help            show this help message and exit
  -a SFSFILENAME        Path for SFS file
  -c FIX_THETA_RATIO    set the fixed value of thetaS/thetaN (i.e. mutation rate ratio)
  -d DENSITYOF2NS       gamma, lognormal, normal, fixed2Ns
  -g                    turn on optimization using basinhopping and dualannealing (very slow, often finds better optimum)
  -f FOLDSTATUS         usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded'
  -m SETMAX2NS          optional setting for 2Ns maximum, default = 0, use with -d lognormal or -d gamma
  -M MAXI               the maximum bin index to include in the calculations, default=None
  -p POPLABEL           a population name and/or data type or other label for the start of the output filename
  -t                    if -d lognormal or -d gamma, estimate the maximum 2Ns value
  -r OUTDIR             results directory
  -x                    if true and output file already exists, the run is stopped, else a new numbered output file is made
  -z                    include a proportion of the mass at zero in the density model
  -Q THETARATIORANGE [THETARATIORANGE ...]
                        optional range for thetaratio (i.e. mutation rate ratio), low end followed by high end
    
deprecated models and options (accessible by setting ):
    uni3fixed : uniform in each of 3 intervals  ((-1000,-1),(-1,1),(1,10)), parameters are the proportions in the two left bins  (right bin is 1- sum of those proportions)
    uni3float : uniform in each of 3 intervals  ((-1000, c0),(c0, c1),(c1,10)), parameters are the proportions in the two left bins (right bin is 1- sum of those proportions) as well as c0 and c1  

    other model additions and options: -e, -o, -z, -w, -M

"""
import numpy as np
from scipy.optimize import minimize,minimize_scalar,OptimizeResult
from scipy.optimize import basinhopping, brentq,dual_annealing
from scipy.stats import chi2
import os
import os.path as op
import math
import random
import time
import argparse
import sys
# Get the absolute directory of the script currently running
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Go up one level from the script)
parent_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, parent_dir)
# Build the path to SFRatios.py
import SFRatios_functions
import SFRatios_functions as SRF 
from scipy.optimize import OptimizeWarning
import warnings
warnings.filterwarnings("ignore", category=OptimizeWarning)
from functools import lru_cache

starttime = time.time()

#fix random seeds to make runs with identical settings repeatable 
random.seed(1) 
np.random.seed(2) 
defaultnumberofbasicoptimizatinos = 3
kluge_likelihood_difference_bug_patch = 10 

deprecated_options_OFF = True # use this to turn off obscure options not for the release of this program. 
# deprecated_options_OFF  = False # set to False to turn obscure options and debug 

# to turn various things off or on when debugging, set to True by setting  deprecated_options_OFF  = False and -D on the command line
miscDebug = False 


def stochastic_round(number):
    floor_number = int(number)
    # Probability of rounding up is the fractional part of the number
    if random.random() < (number - floor_number):
        return floor_number + 1
    else:
        return floor_number

def find_subranges(neusfs, args):
    """
    Identifies subranges in a list where each subrange sums to at least a minimum value.
    
    Parameters:
    neusfs (list): List of values (counts) where index corresponds directly to bin number (0-indexed)
    args: Object with attributes:
        maxi (int): Lowest bin number where value is less than mindenom
        mindenom (float): Minimum sum required for each subrange
    
    Returns:
    list: Array of subranges, each represented as (start_bin, end_bin+1) following Python convention
          of half-open intervals [start, end)
    """
    subranges = []
    
    # Check if args.maxi + 1 is beyond the array length
    if args.maxi + 1 >= len(neusfs):
        return subranges  # No subranges possible
    
    # Start from args.maxi + 1
    current_index = args.maxi + 1
    
    while current_index < len(neusfs):
        sum_values = 0
        start_bin = current_index
        end_bin = start_bin
        
        # Continue adding bins until sum reaches or exceeds mindenom
        while end_bin < len(neusfs):
            # Add the value at this bin
            sum_values += neusfs[end_bin]
            
            if sum_values >= args.mindenom:
                # We've reached the minimum denominator
                subranges.append((start_bin, end_bin + 1))  # Python-style half-open interval
                # Move to the next bin for the next subrange
                current_index = end_bin + 1
                break
            
            # If we haven't reached mindenom, move to the next bin
            end_bin += 1
        
        # If we didn't find a valid subrange starting from current_index
        # or if we've reached the end of the array, exit the loop
        if end_bin >= len(neusfs) or sum_values < args.mindenom:
            break
    
    return subranges



def getSFSratios(args,fn,dofolded,isfolded = False): 
    """
        data text file format: 
            line 1:  arbitrary text
            line 2:  neutral SFS beginning with 0 bin (which is ignored)
            line 3:  blank
            line 4:  selected SFS beginning with 0 bin (which is ignored)

        dofolded and isfolded conditions are based on args.foldstatus
            isfolded = args.foldstatus == "isfolded" 
            dofolded = (args.foldstatus == "isfolded"  or args.foldstatus == "foldit")            

        isfolded == False  means is not folded (either foldit or unfolded )
        isfolded == True  means isfolded  don't bother making folded SFSs,  regardless of dofolded, return the SFSs as loaded

        dofolded == True  return folded SFSs (means foldit must have been true) 
        dofolded == False  return the SFSs as is 

        (dofolded == True and isfolded == True) not allowed

    """
    seps = [' ', '\t', ',']
    lines = open(fn,'r').readlines()
    datafileheader = lines[0].strip()
    sfss = []
    for line in [lines[1],lines[3]]: # neutral, skip a line, then selected 
        

# normalize
        for sep in seps:
            line = line.replace(sep, ' ')
        if "." in line:
            sfs = list(map(float,line.strip().split()))
        else:
            sfs = list(map(int,line.strip().split()))
        sfs[0] = 0
        sfss.append(sfs)       
    if len(sfss[0]) != len(sfss[1]) :
        print("Exception, Neutral and Selected SFSs are different lengths ")
        exit()
    if isfolded == False:
        f_sfss = []
        nc  = len(sfss[0])
        for sfs in sfss:
            f_sfs = [0] + ([sfs[i] + sfs[nc-i] for i in range(1,nc//2)] + [sfs[nc//2]] if nc % 2 == 0 else  [sfs[i] + sfs[nc-i] for i in range(1,1+nc//2)])
            f_sfss.append(f_sfs)
        if dofolded: # return the folded, i.e. f_sfss
            neusfs = f_sfss[0]
            selsfs = f_sfss[1]
        else: # return original, i.e. sfss 
            neusfs = sfss[0]
            selsfs = sfss[1]
    else: # return original, i.e. sfss 
        neusfs = sfss[0]
        selsfs = sfss[1]
        nc = 2*(len(neusfs) - 1) 
    # this was previously "if dontkeepzeroratios == False:"  and the function call had this on by default,  even though args.dontkeepzeroratios had been defined as true  - i.e. a bug 
    if args.dontkeepzeroratios == False:
        ratios = [math.inf if  neusfs[j] <= 0.0 else selsfs[j]/neusfs[j] for j in range(len(neusfs))]
    else:
        ratios = [math.inf if  (neusfs[j] <= 0.0 or selsfs[j] <= 0.0) else selsfs[j]/neusfs[j] for j in range(len(neusfs))]
    thetaNest = sum(neusfs)/sum([1/i for i in range(1,nc )]) # this should work whether or not the sfs is folded 
    thetaSest = sum(selsfs)/sum([1/i for i in range(1,nc )]) # this should work whether or not the sfs is folded 
    if thetaNest <= 0.0 or not math.isfinite(thetaNest):
        raise ValueError("Neutral SFS has zero or non-finite total count; cannot estimate thetaNspace")
    if thetaSest < 0.0 or not math.isfinite(thetaSest):
        raise ValueError("Selected SFS has non-finite total count")    
    # thetaNspace used for integrating over thetaN 
    thetaNspace = np.logspace(np.log10(thetaNest/math.sqrt(args.thetaNspacerange)), np.log10(thetaNest*math.sqrt(args.thetaNspacerange)), num=101) # used for likelihood calculation,  logspace integrates better than linspace
    args.subranges = None
    if args.mindenom is not None:
        args.maxi = len(neusfs)-1
        for i,c in enumerate(neusfs[1:]):
            if c < args.mindenom:
                args.maxi = i
                break
        if args.sumbins:
            args.subranges = find_subranges(neusfs, args)
            ratios = [math.inf if  neusfs[j] <= 0.0 else selsfs[j]/neusfs[j] for j in range(0,args.maxi+1)]
            tempneusfs = neusfs[:args.maxi + 1]
            tempselsfs = selsfs[:args.maxi + 1]
            for subrange in args.subranges:
                nm = sum(selsfs[subrange[0]:subrange[1]])
                dm = sum(neusfs[subrange[0]:subrange[1]])
                if args.dontkeepzeroratios == False:
                    rtemp = math.inf if dm <= 0.0 else nm/dm 
                else:
                    rtemp = math.inf if (dm <= 0.0 or nm <= 0.0) else nm/dm
                ratios.append(rtemp)
                tempneusfs.append(dm)
                tempselsfs.append(nm)
            args.maxi = None  # use subranges,  not maxi 
            neusfs = tempneusfs
            selsfs = tempselsfs

    return args,datafileheader,nc,neusfs,selsfs,ratios,thetaNest,thetaSest,thetaNspace

def update_table(X, headers, new_data, new_labels):
    # Update headers and columns for table of observed and expected values 
    # called by buildSFStable()
    headers.extend(new_labels)
    # Format and add new data
    for i in range(len(new_data[0])):
        if len(X) <= i:
            X.append([])
        ratiotemp = "na" if math.isinf(float(new_data[2][i])) else f"{float(new_data[2][i]):.4g}"
        # formatted_row = [f"{float(new_data[0][i]):.1f}", f"{float(new_data[1][i]):.1f}", f"{float(new_data[2][i]):.4g}"]
        formatted_row = [f"{float(new_data[0][i]):.1f}", f"{float(new_data[1][i]):.1f}", ratiotemp]
        X[i].extend(formatted_row)
    return X, headers

def calcexpectedSFS(args,paramdic,pm0tempval,nc):
    tempthetaratio = paramdic['thetaratio'] if args.estimate_both_thetas == False else None
    tempmisspec = None if args.includemisspec==False else paramdic["misspec"]
    if args.densityof2Ns=="lognormal":
        params = (paramdic["mu"],paramdic["sigma"])
        if args.estimatemax2Ns:   
            neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params,pm0tempval, True, tempthetaratio,subranges = args.subranges)
        else:
            neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],args.setmax2Ns,nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params,pm0tempval, True, tempthetaratio,subranges = args.subranges)
    elif args.densityof2Ns=='gamma':
        # params = (paramdic["alpha"],paramdic["beta"])
        params = (paramdic["mean"],paramdic["shape"])
        if args.estimatemax2Ns:
            neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],paramdic["max2Ns"],nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio,subranges = args.subranges)
        else:
            neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],args.setmax2Ns,nc ,None,args.dofolded,
                tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio,subranges = args.subranges)
    elif args.densityof2Ns=="normal":
        params = (paramdic["mu"],paramdic["sigma"])
        neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,args.dofolded,
            tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio,subranges = args.subranges)
    elif args.densityof2Ns == "uni3fixed":
        params = (paramdic["p0"],paramdic["p1"])
        neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,args.dofolded,
            tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio,subranges = args.subranges)                
    elif args.densityof2Ns == "uni3float":
        params = (paramdic["p0"],paramdic["p1"],paramdic["c0"],paramdic["c1"])
        neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,args.dofolded,
            tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio,subranges = args.subranges)                
    elif args.densityof2Ns =="fixed2Ns":
        params = (paramdic["2Ns"],)
        neusfs,selsfs,ratios = SRF.simsfsratio(paramdic["thetaN"],paramdic["thetaS"],None,nc ,None,args.dofolded,
            tempmisspec,args.densityof2Ns,params, pm0tempval, True, tempthetaratio,subranges = args.subranges)    
    return neusfs,selsfs,ratios

def buildSFStable(args,paramdic,pm0tempval,pmmasstempval,pmvaltempval,X,headers,nc):
    """
        generates expected values using estimated parameters
        calls update_table()
    """
    neusfs,selsfs,ratios = calcexpectedSFS(args,paramdic,pm0tempval,nc)
    if args.estimate_both_thetas == False:
        X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["Fit*_N","Fit*_S","Fit_Ratio"])
    else:
        X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["{}_N","{}_S","{}_Ratio"])
    return X,headers


def check_filename_make_new_numbered_as_needed(filename,filecheck):
    """
        if file does not exist return False and the filename
        if a file exists  make a new numbered version
        if a numbered version exists,  make one with a higher number 

    """
    if  op.exists(filename):
        if filecheck:
            return False,filename
        if "." in filename:
            base = filename[:filename.rfind(".")]
            extension = filename[filename.rfind("."):]
        else:
            base = filename
            extension = ""
        if base[-1] == ")" and base[-2] in "0123456789":
            base = base[:base.rfind("(")]
        k = 1
        while op.exists(base + "({})".format(k) + extension): 
            k += 1
        return True,base + "({})".format(k) + extension
    else:
        return True,filename

def makeresultformatstrings(args):    
    """
    stuff for formatting output under various models
    """
    resultlabels =["likelihood"]
    resultformatstrs = ["{}\t{:.3f}"]
    if args.estimate_both_thetas == False:
        if args.fix_theta_ratio is None:
            resultlabels += ["thetaratio"]
            resultformatstrs += ["{}\t{:.4f}\t({:.4f} - {:.4f})"]
    else:
        resultlabels += ["thetaN","thetaS"]
        resultformatstrs += ["{}\t{:.2f}\t({:.2f} - {:.2f})","{}\t{:.2f}\t({:.2f} - {:.2f})"]
    if args.densityof2Ns in ("lognormal","normal"):
        resultlabels += ["mu","sigma"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})"]
    elif args.densityof2Ns == "gamma":
        resultlabels += ["mean","shape"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})"]
    elif args.densityof2Ns == "uni3fixed":
        resultlabels += ["p0","p1"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})"]    
    elif args.densityof2Ns == "uni3float":
        resultlabels += ["p0","p1","c0","c1"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})","{}\t{:.5g}\t({:.5g} - {:.5g})"]                 
    elif args.densityof2Ns == "fixed2Ns":
        resultlabels += ["2Ns"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    if args.estimate_pointmass0:
        resultlabels += ["pm0"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    if args.estimatemax2Ns:
        resultlabels += ["max2Ns"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    if args.includemisspec:
        resultlabels += ["misspec"]
        resultformatstrs += ["{}\t{:.5g}\t({:.5g} - {:.5g})"]
    paramlabels = resultlabels[1:] # skip the likelihood in position 0 
    resultlabels += ["expectation","mode"]
    resultformatstrs += ["{}\t{:.3f}","{}\t{:.3f}"]
    return resultlabels,resultformatstrs,paramlabels

def set_bounds_and_start_possibilities(args,thetaNest,thetaSest,ntrials):
    """
        specify bounds for each model
            boundary values are based on a lot of hunches, trial and error 
        specify start values for each optimization trial 
        
    """
    bounds = []
    startvals = [[] for _ in range(ntrials)]
    if not args.fix_theta_ratio:
        if args.estimate_both_thetas == False:
            thetaratioest = thetaSest/thetaNest
            if args.thetaratiorange is not None:
                bounds.append((args.thetaratiorange[0],args.thetaratiorange[1]))
                for sv in startvals: sv.append(random.uniform(args.thetaratiorange[0],args.thetaratiorange[1]))                
            else:
                bounds.append((thetaratioest/20,thetaratioest*20))
                for sv in startvals: sv.append(random.uniform(thetaratioest/3,thetaratioest*3))
        else:
            bounds.append((thetaNest/30,thetaNest*30))
            for sv in startvals: sv.append(random.uniform(thetaNest/10,thetaNest*10))
            bounds.append((thetaSest/30,thetaSest*30))
            for sv in startvals: sv.append(random.uniform(thetaSest/10,thetaSest*10))
    if args.densityof2Ns == "lognormal":
        bounds += [(-5,10),(0.01,5)]
        for sv in startvals: sv.append(random.uniform(0.3,3))
        for sv in startvals: sv.append(random.uniform(0.5,1.5))
    elif args.densityof2Ns =="gamma":
        # bounds += [(-10000,0.0),(0.5,10)]       
        bounds += [(-10000,0.0),(0.1,5)]       
        for sv in startvals: sv.append(random.uniform(-10,-1))
        for sv in startvals: sv.append(random.uniform(0.5,2))
    elif args.densityof2Ns=="normal":
        bounds += [(-50,20),(0.1,20)]
        for sv in startvals: sv.append(random.uniform(-15,1))
        for sv in startvals: sv.append(random.uniform(0.2,4))
    elif args.densityof2Ns=="uni3fixed":
        bounds += [(0.0,1.0),(0.0,1.0)]
        for sv in startvals: sv.append(random.uniform(0.1,0.4))
        for sv in startvals: sv.append(random.uniform(0.1,0.4))
    elif args.densityof2Ns=="uni3float":
        bounds += [(0.0,1.0),(0.0,1.0),(-999,9),(-999,9)]
        for sv in startvals: sv.append(random.uniform(0.1,0.4))
        for sv in startvals: sv.append(random.uniform(0.1,0.4))        
        for sv in startvals: sv.append(random.uniform(-100,-2))    #c0
        for sv in startvals: sv.append(random.uniform(-1,5))        #c1
    else:# otherwise density == "fixed2Ns"
        bounds += [(-10000,100)]
        for sv in startvals: sv.append(random.uniform(-10,1))
    if args.estimate_pointmass0:
        bounds += [(0.0,0.5)] # max mass at 0 is 0.5 
        for sv in startvals: sv.append(random.uniform(0.01,0.49))
    if args.estimatemax2Ns:
        bounds += [(-20.0,20.0)]
        for sv in startvals: sv.append(random.uniform(-5,1))
    if args.includemisspec:
        bounds += [(0.0,0.2)] # assume misspecification rate is less than 0.2 
        for sv in startvals: sv.append(random.uniform(0.001,0.1))
    return bounds,startvals 

def buildoutpaths(args):
    """
        makes the results directory as needed
        generates an output filename from the command line options 
    """
    os.makedirs(args.outdir, exist_ok=True)
    fnameparts = []
    if args.poplabel != "":
        fnameparts.append(args.poplabel)
    if args.estimate_both_thetas == False:
        fnameparts.append("Qratio")
    else:
        fnameparts.append("QNQS")
    fnameparts.append("{}".format(args.densityof2Ns))
    fnameparts.append("nc{}".format(args.nc))
    if args.estimatemax2Ns == True:
        fnameparts.append("MX{}".format(args.setmax2Ns))
    elif  args.setmax2Ns is not None and args.densityof2Ns in ('lognormal','gamma'):
        fnameparts.append("M{}".format(args.setmax2Ns))
    if args.fix_theta_ratio != None:
        fnameparts.append("FXQR{}".format(args.fix_theta_ratio))
    if args.estimate_pointmass0:
        fnameparts.append("PM0")
    if args.includemisspec:
        fnameparts.append("MS")
    fnamebase = "_".join(fnameparts)
    outpathstart = op.join(args.outdir,fnamebase)           
    outfilename = outpathstart + "_estimates.out"
    fileok, outfilename = check_filename_make_new_numbered_as_needed(outfilename,args.filecheck)
    if not fileok:
        print(outfilename," already exists")
        exit()
    return outfilename

def countparameters(args):
    numparams = 0
    numparams += 0 if args.fix_theta_ratio else (1 if (args.estimate_both_thetas == False) else 2) # 0 if theta ratio is fixed, otherwise 1 if using the theta ratio parameters,  else 2 when using thetaN and thetaS 
    numparams += 2 if args.densityof2Ns in ("normal","lognormal","gamma","uni3fixed") else (1 if args.densityof2Ns=="fixed2Ns" else (4 if args.densityof2Ns=="uni3float" else 0))
    numparams += 1 if args.estimatemax2Ns else 0 
    numparams += 1 if args.estimate_pointmass0 else 0 
    numparams += 1 if args.includemisspec else 0
    return numparams    

def makeparamdic(args,paramlabels,resultx,thetaNest):
    paramdic = dict(zip(paramlabels,resultx))
    if args.estimate_both_thetas == False:
        paramdic['thetaN']=thetaNest
        if args.fix_theta_ratio is not None:
            paramdic["thetaratio"] = args.fix_theta_ratio
        paramdic['thetaS'] = paramdic["thetaratio"]*thetaNest
    return paramdic

def writeresults(args,numparams,thetaNest,paramlabels,resultlabels,resultformatstrs,resultx,likelihood,confidence_intervals,outfilename,message):
    """
    """
    paramdic = makeparamdic(args,paramlabels,resultx,thetaNest)
    #get expectation
    pm0tempval = pmmasstempval = pmvaltempval = None
    densityof2Nsadjust = None
    if args.densityof2Ns == "lognormal":
        expectation,mode,negsd,densityof2Nsadjust,xvals = SRF.getXrange(args.densityof2Ns,(paramdic['mu'],paramdic['sigma']),(paramdic['max2Ns'] if args.estimatemax2Ns else args.setmax2Ns))
    elif args.densityof2Ns == "gamma" :
        expectation,mode,negsd,densityof2Nsadjust,xvals = SRF.getXrange(args.densityof2Ns,(paramdic['mean'],paramdic['shape']),(paramdic['max2Ns'] if args.estimatemax2Ns else args.setmax2Ns))
    elif args.densityof2Ns == "uni3fixed":
        expectation = -(11/2)* (-1 + 92*paramdic['p0'] + paramdic['p1'])
        mode = np.nan
    elif args.densityof2Ns == "uni3float":
        expectation,mode,negsd,densityof2Nsadjust,xvals = SRF.getXrange(args.densityof2Ns,(paramdic['p0'],paramdic['p1'],paramdic['c0'],paramdic['c1']),(paramdic['max2Ns'] if args.estimatemax2Ns else args.setmax2Ns))

    elif args.densityof2Ns == "normal": 
        expectation,mode,negsd,densityof2Nsadjust,xvals = SRF.getXrange(args.densityof2Ns,(paramdic['mu'],paramdic['sigma']),(paramdic['max2Ns'] if args.estimatemax2Ns else args.setmax2Ns))
    else:
        expectation = mode = np.nan
    if args.estimate_pointmass0:
        expectation *= (1-paramdic['pm0'])
        pm0tempval = paramdic['pm0']
    AIC = 2*numparams - 2*likelihood
    paramwritestr = ["AIC\t{:.3f}".format(AIC)] 
    paramwritestr+=[resultformatstrs[0].format(resultlabels[0],likelihood)]
    paramwritestr += [resultformatstrs[i+1].format(val,resultx[i],confidence_intervals[i][0],confidence_intervals[i][1]) for i,val in enumerate(resultlabels[1:-2])]
    if expectation == expectation: # this is how you check if it is not nan (i.e. if a is nan  then a != a)
        paramwritestr += [resultformatstrs[-2].format(resultlabels[-2],expectation)]
    if mode == mode: # this is how you check if it is not nan (i.e. if a is nan  then a != a)
        paramwritestr += [resultformatstrs[-1].format(resultlabels[-1],mode)]
    
    outf = open(outfilename, "a")
    outf.write(message)
    if densityof2Nsadjust:
        outf.write("\tNumerical integration of {} distribution check: {:.2g}  (near 1 is good, large departures from 1 suggest a numerical integration difficulty)\n".format(args.densityof2Ns,densityof2Nsadjust))
    outf.write("\n".join(paramwritestr)+"\n")
           
    outf.close()
    return pm0tempval,pmmasstempval,pmvaltempval,paramdic,expectation,mode

def generate_confidence_intervals(func,p_est, arglist, maxLL,bounds,alpha=0.05):
    """
    These are not very useful as there is very strong covariation in parameter values 

    Generates 95% confidence intervals for each parameter in p_est.
    Find bounds where the likelihood drops by an ammount given by chi-square distribution, 1df 
    Find confidence interval bounds by searching for values that cross the threshold

    Args:
        p_est: List of estimated parameter values.
        f: Function that returns the log-likelihood for a given parameter set.
        maxLL:  the log likelihood at p_est 
        arglist : a tuple or list with all the rest of the arguments,  after the parameter values, to pass to the f 
        alpha: Significance level for confidence intervals (default: 0.05).

    Returns:
        List of tuples, where each tuple contains (lower bound, upper bound) for a parameter.
    """

    confidence_intervals = []
    chiinterval = chi2.ppf(1 - alpha, df=1)/2  
    likelihood_threshold = maxLL - chiinterval
    for i in range(len(p_est)):
        # Create a minimization function with fixed parameters except for the i-th one

        # replace this block with codex suggested hardened code 
        # def neg_log_likelihood_fixed(p_i):
        #     p_fixed = p_est.copy()
        #     p_fixed[i] = p_i
        #     return -func(p_fixed, *arglist)  # Minimize negative log-likelihood
        # lbtemp = bounds[i][0]/2
        # if lbtemp  <   p_est[i] and  maxLL - neg_log_likelihood_fixed(lbtemp) > chiinterval:
        #     lower_bound = brentq(lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold, lbtemp, p_est[i])
        # else:
        #     lower_bound = np.nan # this should still be able to be written to the file
        # ubtemp = bounds[i][1]*2
        # if ubtemp  >   p_est[i] and  maxLL - neg_log_likelihood_fixed(ubtemp) > chiinterval:
        #     upper_bound = brentq(lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold, p_est[i], ubtemp)
        # else:
        #     upper_bound = np.nan # this should still be able to be written to the file

        # confidence_intervals.append((lower_bound, upper_bound))
    # codex suggested hardened code
        def neg_log_likelihood_fixed(p_i):
            p_fixed = p_est.copy()
            p_fixed[i] = p_i
            val = func(p_fixed, *arglist)
            if not math.isfinite(val):
                return math.inf
            return -val

        lower_bound = np.nan
        upper_bound = np.nan

        lbtemp = bounds[i][0] / 2
        try:
            if lbtemp < p_est[i]:
                lbval = neg_log_likelihood_fixed(lbtemp)
                if math.isfinite(lbval) and maxLL - lbval > chiinterval:
                    lower_bound = brentq(
                        lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold,
                        lbtemp,
                        p_est[i],
                    )
        except (ValueError, ArithmeticError):
            lower_bound = np.nan

        ubtemp = bounds[i][1] * 2
        try:
            if ubtemp > p_est[i]:
                ubval = neg_log_likelihood_fixed(ubtemp)
                if math.isfinite(ubval) and maxLL - ubval > chiinterval:
                    upper_bound = brentq(
                        lambda p_i: neg_log_likelihood_fixed(p_i) - likelihood_threshold,
                        p_est[i],
                        ubtemp,
                    )
        except (ValueError, ArithmeticError):
            upper_bound = np.nan

        confidence_intervals.append((lower_bound, upper_bound))
    return confidence_intervals

def calcdistances(X):
    """
        calculate distances between observed and expected ratio lists
        X can be a table built for output or a list of two lists of ratios
    """
    sumsq = 0.0
    c = 0
    if len(X)==2:
        for i in range(1,len(X[0])):
            c+=1
            sumsq += pow(X[0][i] - X[1][i],2)
    else:
        for i,row in enumerate(X):
            if i > 0:
                c += 1
                try: # might be some "na" values that need to be skipped
                    sumsq += pow(float(row[5]) -float(row[2]),2)
                except:
                    pass
    RMSE = math.sqrt(sumsq/c)
    EucDis = math.sqrt(sumsq)
    return EucDis,RMSE

def run(args):
    isfolded = args.foldstatus == "isfolded" 
    args.dofolded = args.foldstatus == "isfolded"  or args.foldstatus == "foldit"

    #GET RATIO DATA 
    args,datafileheader,nc,neusfs,selsfs,ratios,thetaNest,thetaSest,thetaNspace = getSFSratios(args,args.sfsfilename,args.dofolded,isfolded=isfolded)
    args.nc = nc 
    args.datafileheader = datafileheader
    args.numparams = countparameters(args)
    args.local_optimize_method="Nelder-Mead" # this works better, more consistently and faster than Powell
    

    # START RESULTS FILE
    outfilename = buildoutpaths(args)
    outf = open(outfilename, "w") # write run info to outfile
    outf.write("SFRatios.py   @Jody Hey 2024\n=============================\n")
    outf.write("Command line: " + args.commandstring + "\n")
    outf.write("Arguments:\n")
    for key, value in vars(args).items():
        outf.write("\t{}: {}\n".format(key,value))
    outf.write("\n")
    outf.close()

    #SET STUFF UP FOR RECORDING RESULTS
    X = [] # Initialize the table 
    headers = [] # Initialize the table headers
    X, headers = update_table(X, headers,[neusfs,selsfs,ratios], ["DataN","DataS","DataRatio"])
    resultlabels,resultformatstrs,paramlabels = makeresultformatstrings(args)    
    #SET OPTIMIZATION FUNCTIONS AND TERMS
    boundsarray,startvals = set_bounds_and_start_possibilities(args,thetaNest,thetaSest,args.optimizetries)
    if args.estimate_both_thetas == False:
        if args.sumbins == False:
            func = SRF.NegL_SFSRATIO_estimate_thetaratio
            arglist = (nc,args.dofolded,args.includemisspec,args.densityof2Ns,args.fix_theta_ratio,args.setmax2Ns,args.estimate_pointmass0,args.maxi,thetaNspace,ratios)
                            # p,nc,dofolded,includemisspec,densityof2Ns,fix_theta_ratio,max2Ns,estimate_pointmass0,maxi,thetaNspace,zvals): 
        else:
            func = SRF.NegL_SFSRATIO_estimate_thetaratio_with_subranges
            arglist = (nc,args.dofolded,args.includemisspec,args.densityof2Ns,False,args.setmax2Ns,args.estimate_pointmass0,thetaNspace,args.subranges,ratios)
    else:
        func = SRF.NegL_SFSRATIO_estimate_thetaS_thetaN
        arglist = (nc,args.dofolded,args.includemisspec,args.densityof2Ns,False,args.setmax2Ns,args.estimate_pointmass0,args.maxi,ratios)     


    #RUN MINIMIZE TRIALS
    outf = open(outfilename, "a")
    for ii in range(args.optimizetries):
        if ii == 0:
            resvals = []
            rfunvals = []
        startarray = startvals[ii]
        result = minimize(func,np.array(startarray),args=arglist,bounds = boundsarray,method=args.local_optimize_method,options={"disp":False,"maxiter":1000*4})    
        resvals.append(result)
        rfunvals.append(-result.fun)
    outf.close()
    besti = rfunvals.index(max(rfunvals))
    OPTresult = resvals[besti]
    OPTlikelihood = rfunvals[besti]
    paramdic = makeparamdic(args,paramlabels,OPTresult.x,thetaNest)
    neuSFS,selSFS,expratios = calcexpectedSFS(args,paramdic,None if 'pm0' not in paramdic else paramdic['pm0'],nc)
    opteucdis,optRMSE = calcdistances([ratios,expratios])
    #basinhopping
    if args.moreoptimization:
        if args.optimizetries>0:
            startarray = startvals[besti] # start at the best value found previously,  is this helpful ? works better than an arbitrary start value 
        else:
            startarray = [(boundsarray[i][0] + boundsarray[i][1])/2.0 for i in range(len(boundsarray))]
        if miscDebug:
            # trying to catch a bug here   
            for bi,b in enumerate(boundsarray):
                if not (boundsarray[bi][0] < startarray[bi] < boundsarray[bi][1]):
                    print("basinhopping bounds problem: i:{} boundsarray[i]:{}  startarray[i]:{}".format(bi,boundsarray[bi],startarray[bi]))
                    print(boundsarray)
                    print(startarray)
                    exit()
        try:
            BHresult = basinhopping(func,np.array(startarray),T=10.0,
                                    minimizer_kwargs={"method":args.local_optimize_method,"bounds":boundsarray,"args":tuple(arglist)})
            BHlikelihood = -BHresult.fun
            paramdic = makeparamdic(args,paramlabels,BHresult.x,thetaNest)
            neuSFS,selSFS,expratios = calcexpectedSFS(args,paramdic,None if 'pm0' not in paramdic else paramdic['pm0'],nc)
            BHeucdis,BHRMSE = calcdistances([ratios,expratios])
        except Exception as e:
            BHlikelihood = -np.inf
            BHresult = None
            outf = open(outfilename, "a")
            outf.write("\nbasinhopping failed with message : {}\n".format(e))
            outf.close()
        if (BHlikelihood - OPTlikelihood > kluge_likelihood_difference_bug_patch and BHeucdis > opteucdis): #BH seems not to have worked
            BHlikelihood = -np.inf
            BHresult = None
    else:
        BHlikelihood = -np.inf
        BHresult = None
    #dualanneal - use two starting temperatures, default and 20000
    if args.moreoptimization:
        try:
            DA1result = dual_annealing(func, boundsarray, args=tuple(arglist))
            DA1likelihood = -DA1result.fun
            paramdic = makeparamdic(args,paramlabels,DA1result.x,thetaNest)
            neuSFS,selSFS,expratios = calcexpectedSFS(args,paramdic,None if 'pm0' not in paramdic else paramdic['pm0'],nc)
            DA1eucdis,DA1RMSE = calcdistances([ratios,expratios])
            if (DA1likelihood - OPTlikelihood > kluge_likelihood_difference_bug_patch and DA1eucdis > opteucdis): #DA1 seems not to have worked
                DA1likelihood = -np.inf
                DA1result = None
        except Exception as e:
            DA1likelihood = -np.inf
            DA1result = None
            outf = open(outfilename, "a")
            outf.write("\ndualannealing1 failed with message : {}\n".format(e))
            outf.close()
        try:
            DA2result = dual_annealing(func, boundsarray, args=tuple(arglist),initial_temp=20000.0)
            DA2likelihood = -DA2result.fun
            paramdic = makeparamdic(args,paramlabels,DA2result.x,thetaNest)
            neuSFS,selSFS,expratios = calcexpectedSFS(args,paramdic,None if 'pm0' not in paramdic else paramdic['pm0'],nc)
            DA2eucdis,DA2RMSE = calcdistances([ratios,expratios])
            if (DA2likelihood - OPTlikelihood > kluge_likelihood_difference_bug_patch and DA2eucdis > opteucdis): #DA1 seems not to have worked
                DA2likelihood = -np.inf
                DA2result = None
        except Exception as e:
            DA2likelihood = -np.inf
            DA2result = None
            outf = open(outfilename, "a")
            outf.write("\ndualannealing2 failed with message : {}\n".format(e))
            outf.close()
    else:
        DA1likelihood = DA2likelihood =-np.inf   
        DA1result = DA2result = None  
    # replace this with code that traps nonfinite values    
    # [likelihood,result,outstring]= sorted([[OPTlikelihood,OPTresult,"Optimize"],[BHlikelihood,BHresult,"Basinhopping"],
    #                                        [DA1likelihood,DA1result,"Dualannealing1"],[DA2likelihood,DA2result,"Dualannealing2"]], 
    #                                        key=lambda x: x[0] if x[0] != -np.inf else float('-inf'))[-1]
    candidates = [
                    (OPTlikelihood, OPTresult, "Optimize"),
                    (BHlikelihood, BHresult, "Basinhopping"),
                    (DA1likelihood, DA1result, "Dualannealing1"),
                    (DA2likelihood, DA2result, "Dualannealing2"),
                ]
    filtered = []
    for c in candidates:
        if c[1] is not None and math.isfinite(c[0]):
            filtered.append(c)
    candidates = filtered    

    if not candidates:
        raise RuntimeError("All optimization attempts failed or returned non-finite likelihoods")

    likelihood, result, outstring = max(candidates, key=lambda x: x[0])    
    confidence_intervals = generate_confidence_intervals(func,list(result.x), arglist, likelihood,boundsarray)
    pm0tempval,pmmasstempval,pmvaltempval,paramdic,expectation,mode = writeresults(args,args.numparams,thetaNest,paramlabels,resultlabels,resultformatstrs,result.x,likelihood,confidence_intervals,outfilename,"\nOptimization: {}\nMaximized Likelihood, AIC, Parameter Estimates, 95% Confidence Intervals:\n".format(outstring))
    X,headers = buildSFStable(args,paramdic,pm0tempval,pmmasstempval,pmvaltempval,X,headers,nc)
    if args.sumbins == False:
        EucDis, RMSE = calcdistances(X)
    else:
        EucDis, Rmse = ("na","na")
    # WRITE a TABLE OF DATA and RATIOS UNDER ESTIMATED MODELS        
    outf = open(outfilename, "a")
    if args.sumbins == False:
        outf.write("\nCompare data and optimization estimates\n\tEuclidean Distance: {:.4f} RMSE: {:.5f}\n".format(EucDis,RMSE))
    if args.estimate_both_thetas == False:
        outf.write("\t*Expected counts generated with Wright-Fisher theta estimates (SFRatios does not provide theta estimates)\n")
        outf.write("--------------------------------------------------------------------------------------------------------------\n") 
    else:
        outf.write("------------------------------------------------------------------------------------------\n") 
    if args.subranges is not None:
        maxi_single_bin = args.subranges[0][0]
        outf.write("i\t#bins\t" + "\t".join(headers) + "\n")
        j = 0
        for i,row in enumerate(X):
            if i < maxi_single_bin:
                outf.write("{}\t".format(i) + "1\t" + "\t".join(row) + "\n")    
            else:
                outf.write("{}\t".format(i) + "{}\t".format(len(range(args.subranges[j][0],args.subranges[j][1]))) + "\t".join(row) + "\n")    
                j += 1
    else:
        outf.write("i\t" + "\t".join(headers) + "\n")
        for i,row in enumerate(X):
            outf.write("{}\t".format(i) +"\t".join(row) + "\n")            
    #clear caches and write cache usage to
    SRF.clear_cache(outf = outf if miscDebug else False)
    # WRITE THE TIME AND CLOSE
    endtime = time.time()
    total_seconds = endtime-starttime
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    outf.write(f"\nTime taken: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n")            
    outf.close()
    print("done ",outfilename)

def parsecommandline():
    global miscDebug
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest="sfsfilename",required=True,type = str, help="Path for SFS file")
    parser.add_argument("-c",dest="fix_theta_ratio",default=None,type=float,help="set the fixed value of thetaS/thetaN (i.e. mutation rate ratio)")
    if deprecated_options_OFF  == False:
        parser.add_argument("-d",dest="densityof2Ns",default = "fixed2Ns",type=str,help="gamma, lognormal, normal, uni3fixed,uni3float,fixed2Ns")
        parser.add_argument("-e",dest="includemisspec",action="store_true",default=False,help=" for unfolded, include a misspecification parameter") 
        parser.add_argument("-D",dest ="debugmode",default = False,action="store_true", help = "turn on debug mode,  only works if deprecated_options_OFF  == True")
        parser.add_argument("-q",dest="thetaNspacerange",default=100,type=int,help="optional setting for the range of thetaNspace, alternatives e.g. 25, 400")
        parser.add_argument('--profile', action='store_true', help="Enable profiling")    
        parser.add_argument("-w",dest="estimate_both_thetas",action="store_true",default=False,help="estimate both thetas, not just the ratio") 
    else:
        parser.add_argument("-d",dest="densityof2Ns",default = "fixed2Ns",type=str,help="gamma, lognormal, normal, fixed2Ns")

    parser.add_argument("-g",dest="moreoptimization",default=False,action="store_true",help=" turn on optimization using basinhopping  and dualannealing (very slow, often finds better optimum)") 
    parser.add_argument("-f",dest="foldstatus",required=True,help="usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded' ")    
    parser.add_argument("-m",dest="setmax2Ns",default=0,type=float,help="optional setting for 2Ns maximum, default = 0, use with -d lognormal or -d gamma")
    parser.add_argument("-M",dest="maxi",default=None,type=int,help="the maximum bin index to include in the calculations, default=None")
    parser.add_argument("-p",dest="poplabel",default = "", type=str, help="a population name and/or data type or other label for the start of the output filename")    
    parser.add_argument("-t",dest="estimatemax2Ns",default=False,action="store_true",help=" if  -d lognormal or -d gamma,  estimate the maximum 2Ns value") 
    parser.add_argument("-r",dest="outdir",default = "", type=str, help="results directory")    
    parser.add_argument("-x",dest="filecheck",action="store_true",default=False,help=" if true and output file already exists, the run is stopped, else a new numbered output file is made") 
    parser.add_argument("-z",dest="estimate_pointmass0",action="store_true",default=False,help="include a proportion of the mass at zero in the density model")    
    parser.add_argument("-Q", dest="thetaratiorange", type=float, nargs="+", default=None, help="optional range for thetaratio (i.e. mutation rate ratio), low end followed by high end ")
    parser.add_argument("-v",dest="mindenom",default=None,type=int,help="optional setting the minimum count in a denominator, e.g. 10. If used, args.maxi is set to this bin number")
    parser.add_argument("-w",dest="sumbins",action="store_true",default=False,help=" requires -v, if true sums neutral SFS bins till mindenom is reached ") 

    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    if deprecated_options_OFF  == True: # add the things not included in args by parser.parse_args() when this is set 
        args.thetaNspacerange = 10 #100
        args.profile = False
        args.estimate_both_thetas = False
        args.includemisspec = False
    else:
        if args.debugmode: # use miscDebug because it is global and don't have to pass around args.debugmode 
            miscDebug = True  # this is global, unlike args.debugmode 
        

    if args.densityof2Ns not in ("lognormal","gamma"):
        if (args.estimatemax2Ns):
            parser.error('cannot use -t with -d normal or fixed 2Ns value')
    if deprecated_options_OFF:
        if args.densityof2Ns not in ("gamma","lognormal","normal","fixed2Ns"):
            parser.error('{} not a valid model '.format(args.densityof2Ns))
    else:
        if args.densityof2Ns not in ("gamma","lognormal","normal","fixed2Ns","uni3fixed","uni3float"):
            parser.error('{} not a valid model '.format(args.densityof2Ns))            
    if args.estimatemax2Ns:
        args.setmax2Ns = None
    if args.fix_theta_ratio and args.estimate_both_thetas == True:
        parser.error(' cannot use -c fix_theta_ratio with -w estimate_both_thetas ')
    if args.foldstatus != "unfolded" and args.includemisspec == True:
        parser.error(' cannot include a misspecification term (-e) when SFS is not folded (-f)')
    if args.sumbins and args.mindenom is None:
        parser.error('cannot use -w / sumbins without -v / mindenom')
    if args.mindenom is not None and args.mindenom <= 0:
        parser.error('-v / mindenom must be positive')
    if args.thetaratiorange is not None:
        if len(args.thetaratiorange) != 2:
            parser.error('-Q / thetaratiorange requires exactly two values: low high')
        if args.thetaratiorange[0] <= 0 or args.thetaratiorange[0] >= args.thetaratiorange[1]:
            parser.error('-Q / thetaratiorange must satisfy 0 < low < high')        
    args.optimizetries = defaultnumberofbasicoptimizatinos
    args.dontkeepzeroratios = True  # played with this but it seems to work better when we don't use ratios of zero. 
    return args

if __name__ == '__main__':

    args = parsecommandline()
    ## if --profile
    if args.profile:
        import cProfile 
        import pstats
        import io
        from pstats import SortKey
        # Set up the profiler
        profiler = cProfile.Profile()
        profiler.enable()     
    
    run(args)

    ## if --profile
    if args.profile:
        profiler.disable()
        
        # Write full program profile stats
        prffilename = 'SFRatios_stats_{}.prof'.format(args.poplabel)
        with open(prffilename, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats()
        print("profile stats written too {}".format(prffilename))
        
        # Filter and write myptools profile stats
        prffilename = 'SFRatios_functions_stats_{}.prof'.format(args.poplabel)
        with open(prffilename, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats('SFRatios_functions')
        print("SFRatios_functions profile stats written too {}".format(prffilename))
