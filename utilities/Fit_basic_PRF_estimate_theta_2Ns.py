"""
    fit basic PRF model to estimate theta and 2Ns 
    runs on files with vep consequence labels, built using scripts for building SFSs from large human data sets 
    if sfs file format varies,  adjust  read_data()

    

"""
import numpy as np
from scipy.optimize import minimize,minimize_scalar,OptimizeResult
from scipy.optimize import basinhopping, brentq
from scipy.stats import chi2

import os
import os.path as op
import math
import random
import time
import argparse
import sys
sys.path.append("./")
sys.path.append("../")
import SFRatios_functions as SFR
from tabulate import tabulate

starttime = time.time()

#fix random seeds to make runs with identical settings repeatable 
random.seed(1) 
np.random.seed(2) 

def read_data(filename, foldstatus):
    data = []
    labels = []
    ncvals = []
    f = open(filename, 'r')
    numdone = 0
    f.readline()
    f.readline()
    while True:
        line = f.readline()
        if len(line)==0:
            break
        if len(line.strip()) > 0 and len(line.split()) == 1:
            labels.append(line.split()[0] )
            sfs = list(map(float, f.readline().strip().split()))

            if foldstatus == "foldit":
                nc = len(sfs)
                nciseven = nc % 2 == 0
                if nciseven:
                    sfs = [0] + [sfs[j]+sfs[nc-j] for j in range(1,nc//2)] + [sfs[nc//2]]
                else:
                    sfs = [0] + [sfs[j]+sfs[nc-j] for j in range(1,1 + nc//2)] 
            elif foldstatus == "isfolded":
                nc = 2*(len(sfs) - 1)            
            else: #foldstatus == "unfolded":
                nc = len(sfs)
            ncvals.append(nc)
            data.append(sfs)
            numdone += 1
    return data, labels,ncvals

def run(args):
 
    args.optimizemethod="Nelder-Mead" # this works better, more consistently than Powell
    
    #GET RATIO DATA 
    SFSs,sfslabels,ncvals = read_data(args.sfsfilename, args.foldstatus)
    sfslabels = [s.ljust(max(map(len, sfslabels))) for s in sfslabels]
    

    # START RESULTS FILE
    outfilename = args.outfilename

    outf = open(outfilename, "w")

    outf.write("Fit_basic_PRF_estimate_theta_2Ns.py results\n================\n")
    outf.write("Command line: " + args.commandstring + "\n")
    outf.write("Arguments:\n")
    for key, value in vars(args).items():
        outf.write("\t{}: {}\n".format(key,value))
    outf.write("\n")
    outf.close()

    
    func = SFR.NegL_SFS_Theta_Ns # two parameters, theta and 2Ns
    forsort = [[c,sum(SFSs[i][1:]),0,0] for i,c in enumerate(sfslabels)]
    
    
    for sfsi,sfs in enumerate(SFSs):
        print(sfslabels[sfsi])
        outf = open(outfilename, "a")
        outf.write("\n{}".format(sfslabels[sfsi]))
        outf.write("trial\tlikelihood\tTheta\t2Ns\n")
        sfs[0] = 0
        nc = ncvals[sfsi]
        thetaest = sum(sfs)/sum([1/i for i in range(1,nc)]) 
        boundsarray = [(0.001, 10*thetaest),(-100,10)]
        randomstartbounds = [(thetaest/10,2*thetaest),(-20,1)]
        arglist = (nc,args.foldstatus in ('isfolded','foldit'), False,args.maxi,sfs)
        startvals = []
        for ii in range(args.optimizetries):
            if ii == 0:
                resvals = []
                rfunvals = []
            startarray = [random.uniform(randomstartbounds[0][0],randomstartbounds[0][1]),random.uniform(randomstartbounds[1][0],randomstartbounds[1][1])]
            startvals.append(startarray)
            result = minimize(func,np.array(startarray),args=arglist,bounds = boundsarray,method=args.optimizemethod,options={"disp":False,"maxiter":1000*4})    
            resvals.append(result)
            rfunvals.append(-result.fun)
            outf.write("{}\t{:.5g}\t{}\n".format(ii,-result.fun," ".join(f"{num:.5g}" for num in result.x)))
        if args.basinhoppingopt:
            besti = rfunvals.index(max(rfunvals))
            startarray = startvals[besti] # start at the best value found previously,  is this helpful ? works better than an arbitrary start value 
            try:
                # args.optimizemethod="Powell"
                BHresult = basinhopping(func,np.array(startarray),T=10.0,
                                        minimizer_kwargs={"method":args.optimizemethod,"bounds":boundsarray,"args":tuple(arglist)})
                BHlikelihood = -BHresult.fun
                outf = open(outfilename, "a")
                outf.write("{}\t{:.5g}\t{}\n".format("BH",BHlikelihood," ".join(f"{num:.5g}" for num in BHresult.x)))
                resvals.append(BHresult)
                rfunvals.append(BHlikelihood)
                outf.close()
            except Exception as e:
                BHlikelihood = -np.inf
                outf = open(outfilename, "a")
                outf.write("\nbasinhopping failed with message : {}\n".format(e))
                outf.close()            
        besti = rfunvals.index(max(rfunvals))
        result = resvals[besti]
        outf = open(outfilename, "a")
        outf.write("best\n")
        outf.write("{}\t{:.5g}\t{}\n".format("BH" if besti == args.optimizetries else besti,-result.fun," ".join(f"{num:.5g}" for num in result.x)))
        outf.close()
        forsort[sfsi][2] = result.x[-2] # put 2Ns estimates in a list with the data set labels 
        forsort[sfsi][3] = result.x[-1] # put 2Ns estimates in a list with the data set labels 

    sorted_forsort = sorted(forsort, key=lambda x: x[3])
    outf = open(outfilename, "a")
    outf.write("\nData sets sorted by 2Ns estimates\n")

    # Prepare the data for tabulate
    table_data = [[row[0], f"{row[1]:.1f}", f"{row[2]:.1f}", f"{row[3]:.4f}"] for row in sorted_forsort]
    headers = ["Consequence", "Sum", "Theta", "2Ns"]

    # Generate the formatted table
    table = tabulate(table_data, headers=headers, tablefmt="plain")

    # Write the table to the file
    outf.write(table)


    # WRITE THE TIME AND CLOSE
    endtime = time.time()
    total_seconds = endtime-starttime
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    outf = open(outfilename, "a")
    outf.write(f"\n\nTime taken: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n")            
    outf.close()
    print("done ",outfilename)

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest="sfsfilename",required=True,type = str, help="Path for SFS file")
    parser.add_argument("-m",dest="maxi",default=None,type=int,help="optional setting for the maximum bin index to include in the calculations")
    parser.add_argument("-f",dest="foldstatus",required=True,help="usage regarding folded or unfolded SFS distribution, 'isfolded', 'foldit' or 'unfolded' ")    
    parser.add_argument("-g",dest="basinhoppingopt",default=False,action="store_true",help=" turn on global optimzation using basinhopping (quite slow, sometimes improves things)") 
    parser.add_argument("-i",dest="optimizetries",type=int,default=1,help="run the minimize optimizer # times")
    parser.add_argument("-o",dest="outfilename",required=True, type=str, help="results file ")    
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])

    if args.optimizetries == 0 and args.basinhoppingopt == False:
        parser.error(' either -i must be greater than 0, or -g  or both')
    
    return args


if __name__ == '__main__':
    """

    """
   
    args = parsecommandline()
    run(args)
