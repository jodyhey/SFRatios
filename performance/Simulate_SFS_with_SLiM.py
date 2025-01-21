"""
    runs slim SFS simulations,  generates folded SFSs for neutral and for selected sites 
    sample size n is diploid (as slim expects) and the resulting SFS is folded 
"""
import os
import sys
import argparse
import csv
import time
# import shlex, subprocess
import subprocess
import numpy as np
import math
import socket
import glob 

# This might fix the RuntimeWarning with np.divide
np.seterr(divide='ignore', invalid='ignore')

#---FUNCTION DEFINITION---#

# Function to read one SFS from a file
def readSFS(file):
    for line in open (file, 'r'):
        if len(line) > 1 and line[0] != "#":
            try:
                sfs = line.strip().split()
                sfs = [int(x) for x in sfs]
            except:
                print(sfs)
                pass
    return sfs


# Function to run SLiM as a external process
def runSlim(args,simulation, mu, rec, popSize, seqLen, ns, diploidsamplesize, modelpath,density2Ns, param1, param2, outdir, donotcleandir = False):
    # Path to SLiM
    # slim = "/usr/local/bin/slim"
    # modelsdir = "/Users/tur92196/WorkDir/prfratio/slim/models"
    host = socket.gethostname()
    if host=="compute":
        slim = "/home/tuf29449/download/SLiM/build/slim"
        modelsdir ="/home/tuf29449/prfratio/models"
    else:
        slim = "/usr/bin/slim"
        # modelsdir = "/mnt/d/genemod/better_dNdS_models/popgen/prfratio/slim/models"
        modelsdir = "/mnt/d/genemod/better_dNdS_models/popgen/PRF-Ratio/slim_work/models"

    # # Model dictionary
    # avail_models = {"constant": "Constant size model",
    #                 "iexpansion": "Instantaeous expansion model",
    #                 "ibottlneck": "Instantaneous bottleneck model",
    #                 "popstructure": "Constant size, population structure model",
    #                 "popstructureN2": "N/2 population size after split, population structure model",
    #                 "OOAgravel2011": "Gravel et al. 2011 Out-of-Africa Human demography model"}
    
    # # if model in avail_models:
    # #         print("Ok, " + avail_models.get(model) + " is available!")
    # # else:
    # #     print("Sorry, model " + model + " does not exists!")
    # #     sys.exit()
    # if model not in avail_models:
    #     print("Sorry, model " + model + " does not exists!")
    #     sys.exit()        

    # Distribution of fitness effects dict
    

    # Sample a seed every function call
    seed = str(int(np.random.uniform(low=100000000, high=900000000)))
    #seed = str(123456) # debugging only
    
    # Run SLiM as a python subprocess
    run = subprocess.run([slim, "-s", seed, "-d", ("simu="+str(simulation)), "-d", ("MU="+str(mu)), "-d", ("R="+str(rec)),
                          "-d", ("N="+str(popSize)), "-d", ("L="+str(seqLen)), "-d", ("Ns="+str(ns)), "-d", ("n="+str(diploidsamplesize)),
                          "-d", ("param1="+str(param1)), "-d", ("param2="+str(param2)),
                          "-d", ("outDir="+"'"+outdir+"'"), 
                          "-d", ("maxg="+str(args.max2Ns)),
                        #   modelsdir + "/" + modelfile + ".slim"
                            modelpath
                        #   (modelsdir + "/" + model + "_" + density2Ns + ".slim")
                          ], capture_output=True)
    if run.stderr != b'':
        print("slim problem ",run.stderr)
        exit()
    neutralsfsfile = (outdir + "/" + ("sfs_neutral_" + str(simulation) + "_" + seed + ".txt"))
    selectedsfsfile = (outdir + "/" + ("sfs_selected_" + str(simulation) + "_" + seed + ".txt"))
    if not os.path.exists(neutralsfsfile):
        neutralsfsfile = (outdir + "/" + ("fsfs_neutral_" + str(simulation) + "_" + seed + ".txt"))
    if not os.path.exists(selectedsfsfile):
        selectedsfsfile = (outdir + "/" + ("fsfs_selected_" + str(simulation) + "_" + seed + ".txt"))
    neutral_sfs = readSFS(file = neutralsfsfile)
    selected_sfs = readSFS(file = selectedsfsfile)
    if args.fill0binSelfrac:
        neutral_sfs.insert(0,seqLen*(1-args.fill0binSelfrac) -sum(neutral_sfs))
        selected_sfs.insert(0,seqLen*args.fill0binSelfrac -sum(selected_sfs))
    if donotcleandir==False:
        # os.remove(neutralsfsfile)
        # os.remove(selectedsfsfile)
        # Use glob to find files matching the pattern
        files_to_remove = glob.glob(os.path.join(outdir, "*_" + seed + ".txt"))

        # Iterate and remove each file
        for file_path in files_to_remove:
            os.remove(file_path)
            # print(f"Removed: {file_path}")        
    
    return seed, neutral_sfs, selected_sfs, run.stderr, run.stdout

def combineSFSs(sfslist, nbins):
    nbins = len(sfslist[0])
    csfs = [0]*(nbins)
    for li, list in enumerate(sfslist):
        for bi, bin in enumerate(list):
            csfs[bi] += bin
    return csfs

def writeCombinedSFS(file, header, csfs):
    with open(file, 'w') as f:
        f.write(header + "\n")
        wr = csv.writer(f, delimiter=" ")
        wr.writerows(csfs)
   
# Function to combine individuals SFSs in a file (each one in a row) ## NEED to FIX it??
def readANDcombineSFSs(diploidsamplesize, filename, path):
    
    csfs = [0]*(diploidsamplesize)
    file = (path + "/" + filename + ".txt")
    for line in open(file, 'r'):
        if len(line) > 1 and line[0] != "#":
            try:
                sfs = line.strip().split()
                for i in range(len(csfs)): 
                    csfs[i] += int(sfs[i])
            except:
                print(sfs)
                pass
    return csfs

#---PROGRAM DEFINITION---#
# Pipeline to run multiple sequences (or genes, chrms, etc) for multiples simulations (or replicates)
# It is a wrapper for SLiM code and it allows run different models with different parameters, especially with Ns
# Each replicate (or simulated SFS) is actually a combination of many sequence or gene SFSs#.
def simulateSFSslim(args,nsimulations = 3, mu = 1e-6/4, rec = 1e-6/4, popSize = 1000, seqLen = 10000, 
                     ns = 0.0, density2Ns = "lognormal", nsdistargs = [1.0, 1.0], diploidsamplesize = 40,foldit = False,
                     modelpath = "constant", nSeqs = 5, output_dir = "results/prfratio", savefile = True, donotcleandir = False):

    # Constant value parameters:
    # Intron length and total intron size
    intronL = 810
    intron_totalL = (8*intronL) + 928 # Hard-coded
    
    # Exon length and total exon size
    exonL = 324
    exon_totalL = 8*exonL

    # SFS format
    if foldit:
        sfs_format = "folded"
        numbins = diploidsamplesize-1
    else:
        sfs_format = "unfolded"
        numbins = 2*diploidsamplesize-1
    
    # Define thetas 
    modelname = os.path.split(modelpath)[1][:-5] # just the name of the model file, minus '.slim'
    thetaNeutral = (4*popSize*mu)*intron_totalL*nSeqs
    if "ibottleneck" in modelname:
        thetaNeutral = thetaNeutral/10
    if "iexpansion"  in modelname:
        thetaNeutral = thetaNeutral * 10
    if "OOAgravel2011" in modelname:
        thetaNeutral = thetaNeutral * 122.24
    
    # Selected theta
    thetaSelected = (4*popSize*mu)*exon_totalL*nSeqs
    if "ibottleneck" in modelname:
        thetaSelected = thetaSelected /10
    if "iexpansion" in modelname:
        thetaSelected  = thetaSelected * 10
    if "OOAgravel2011" in modelname:
        thetaSelected = thetaSelected * 122.24

    # Second, check if the distribution exists
    # and if parameters are correct!
    avail_nsdist = {"fixed": "Fixed 2Ns values",
                    "lognormal": "2Ns values sample from a lognormal with param1=meanlog, param2=sdlog",
                    "gamma": "2Ns values sample from a gamma with param1=shape, param2=scale"}
    
    if True : #density2Ns in avail_nsdist:
        print("Ok, " + avail_nsdist.get(density2Ns) + " exists")
        if density2Ns != "fixed":
            # Combine output directory
            path = os.path.join(output_dir, modelname, (str(nsdistargs[0]) + "-" + str(nsdistargs[1]))) 
            # Prapare the headers for each model/Ns simulation
            header_neutral = "# 4Nmu(intron total length)={t} distribution={density2Ns} dist_pars={nsdistargs} n={n} Neutral {sfs} SFS".format(t=thetaNeutral, density2Ns=density2Ns, nsdistargs=nsdistargs, n=diploidsamplesize, sfs=sfs_format)
            header_selected = "# 4Nmu(exon total length)={t} distribution={density2Ns} dist_pars={nsdistargs} n={n} Selected {sfs} SFS".format(t=thetaSelected,density2Ns=density2Ns, nsdistargs=nsdistargs, n=diploidsamplesize, sfs=sfs_format)
        else:
            # Combine output directory
            path = os.path.join(output_dir, modelname, str(ns)) 

            # Prapare the headers for each model/Ns simulation
            header_neutral = "# 4Nmu(intron total length)={t} distribution={density2Ns} Ns={ns} n={n} Neutral {sfs} SFS".format(t=thetaNeutral, density2Ns=density2Ns, ns=ns, n=diploidsamplesize, sfs=sfs_format)
            header_selected = "# 4Nmu(exon total length)={t} distribution={density2Ns} Ns={ns}  n={n} Selected {sfs} SFS".format(t=thetaSelected, density2Ns=density2Ns, ns=ns, n=diploidsamplesize, sfs=sfs_format)

    else:
        print("Sorry, Ns distribution " + density2Ns + " does not exists!")
        sys.exit()

    # Check if output already exists
    if not os.path.exists(path):
        os.makedirs(path)

    if nsimulations == 1:
        simulation = 1
        list_neutral_sfss = []
        list_selected_sfss = []
        list_seeds = []

        j = 0
        while j < nSeqs:
            # RUN SLiM
            seed, neutral_sfs, selected_sfs, stderr, stdout = runSlim(args,simulation=simulation,mu=mu,rec=rec,popSize=popSize,seqLen=seqLen,ns=ns,diploidsamplesize=diploidsamplesize,modelpath=modelpath,density2Ns=density2Ns,param1=nsdistargs[0],param2=nsdistargs[1],outdir=path,donotcleandir=donotcleandir) 

            list_neutral_sfss.append(neutral_sfs)
            list_selected_sfss.append(selected_sfs)
            list_seeds.append(seed)

            j += 1

        csfs_neutral = combineSFSs(list_neutral_sfss, nbins=numbins)
        csfs_selected = combineSFSs(list_selected_sfss, nbins=numbins)
        # calculate the ratio
        csfs_ratio = np.divide(csfs_selected, csfs_neutral).tolist()
        csfs_ratio = [0 if math.isnan(x) else x for x in csfs_ratio]
        
        # insert 0 to the 0-bin
        if args.fill0binSelfrac == None:
            csfs_neutral.insert(0,0) 
            csfs_selected.insert(0,0) 
            csfs_ratio.insert(0,0)

        # Write files containing the simulation combined SFS 
        # One SFS for each simulation (that is a aggregation of subsimulations SFS)
        if savefile:
            sim_csfs_neutral = []
            sim_csfs_selected = []
            sim_csfs_neutral.append(csfs_neutral)
            sim_csfs_selected.append(csfs_selected)
            # Define the output file for the combined SFS(s):
            sims_csfs_neutralfile = (path + "/" + "csfs_neutral.txt")
            sims_csfs_selectedfile = (path + "/" + "csfs_selected.txt")
            writeCombinedSFS(sims_csfs_neutralfile, header_neutral, sim_csfs_neutral)
            writeCombinedSFS(sims_csfs_selectedfile, header_selected, sim_csfs_selected)
        
        # Return statement
        return csfs_neutral, csfs_selected, csfs_ratio, list_seeds

    else: 

        # These lists collects each simulation (combined) SFS and lists of seeds
        # for each gene/fragment/subsimulation
        sims_seeds = []
        sims_csfs_neutral = []
        sims_csfs_selected = []
        sims_csfs_ratio = []

        # Run many replicates
        for i in range(0,nsimulations):
            simulation = i
            # These lists collects the output produced by each simulated gene/fragment/subsimulation
            list_neutral_sfss = []
            list_selected_sfss = []
            list_seeds = []

            j = 0
            while j < nSeqs:
                # RUN SLiM
                seed, neutral_sfs, selected_sfs, stderr, stdout = runSlim(args,simulation=simulation,mu=mu,rec=rec,popSize=popSize,seqLen=seqLen,ns=ns,diploidsamplesize=diploidsamplesize,modelpath=modelpath,density2Ns=density2Ns,param1=nsdistargs[0],param2=nsdistargs[1],outdir=path,donotcleandir=donotcleandir) 

                list_neutral_sfss.append(neutral_sfs)
                list_selected_sfss.append(selected_sfs)
                list_seeds.append(seed)

                j += 1

            csfs_neutral = combineSFSs(list_neutral_sfss, nbins=numbins)
            csfs_selected = combineSFSs(list_selected_sfss, nbins=numbins)            

            # calculate the ratio
            csfs_ratio = np.divide(csfs_selected, csfs_neutral).tolist()
            csfs_ratio = [0 if math.isnan(x) else x for x in csfs_ratio]
            
            # insert 0 to the 0-bin
            if args.fill0binSelfrac == None:
                csfs_neutral.insert(0,0)
                csfs_selected.insert(0,0)
                csfs_ratio.insert(0,0)
            
            # Create a list of outputs (one for each simulation)
            sims_csfs_neutral.append(csfs_neutral)
            sims_csfs_selected.append(csfs_selected)
            sims_csfs_ratio.append(csfs_ratio)
            sims_seeds.append(list_seeds)
    
        # Write files containing the simulation combined SFS 
        # One SFS for each simulation (that is a aggregation of subsimulations SFS)
        if savefile:
            # Define the output file for the combined SFS(s):
            sims_csfs_neutralfile = (path + "/" + "csfs_neutral.txt")
            sims_csfs_selectedfile = (path + "/" + "csfs_selected.txt")
            writeCombinedSFS(sims_csfs_neutralfile, header_neutral, sims_csfs_neutral)
            writeCombinedSFS(sims_csfs_selectedfile, header_selected, sims_csfs_selected)
        
        # Return statement
        return sims_csfs_neutral, sims_csfs_selected, sims_csfs_ratio, sims_seeds

# Define the command line arguments
def parsecommandline():
    parser = argparse.ArgumentParser("python simulate_SFS_withSLiM.py",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", help="Set the parameters of the chosen distribution",
                        dest="nsdistargs", nargs= "+", default = [0.0, 0.0], type=float)
    parser.add_argument("-c",dest="donotcleandir",action="store_true",default=False,help="do not delete temporary sfs files, default is to delete")
    parser.add_argument("-d", dest="density2Ns", default=None, help="set a distribution, fixed, lognormal, or gamma",required=True)
    parser.add_argument("-f",dest="foldit",action="store_true",default=False,help="Make folded sfs,  default is unfolded")
    
    parser.add_argument("-g", help="Non-synonymous population selection coefficient 2Ns (for fixed values DFE)",
                        dest="ns",default=0.0, type=float)
        
    parser.add_argument("-k", help="Number of slim runs per simulation",
                        dest="nSeqs", type=int, required=True)
    parser.add_argument("-L", help="Sequence length",
                        dest="seqLen", default=10000, type=int)
    parser.add_argument("-m", help="full path and name of the slim model file,  e.g. 'constant'  for constant.slim",
                        dest="modelpath", required = True,
                        type = str)    
    parser.add_argument("-n", help="Diploid sample size, i.e. half the # of chromosomes",
                        dest="diploidsamplesize", default=40, type=int)    
    parser.add_argument("-N", help="Population census size",
                        dest="popSize", default=1000, type=int)
    parser.add_argument("-o",help="main supervening output directory (subfolder will be made if needed)",dest="output_dir",required=True)    
    parser.add_argument("-r", help="number of simulations",
                        dest="nsimulations", type=int, required=True)
    parser.add_argument("-R", help="Per site recombination rate per generation",
                        dest="rec", default=1e-6/4, type=float)
    parser.add_argument("-s", help="Save simulated SFS to a file",
                        dest="savefile",default = True, type = bool)    
    parser.add_argument("-U", help="Per site mutation rate per generation",
                        dest="mu", default=1e-6/4, type=float)
    parser.add_argument("-x",dest="max2Ns",default = 0.0, help="max of 2Ns when using lognormal or gamma,  e.g. 1")    
    parser.add_argument("-z",dest="fill0binSelfrac",default = None, type=float,help=" fraction of sequence length that is selected,  used for filling 0 bin. If None, 0 bin is set to zer0 for both Neut and Sel")
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    return args

def main(argv):
    starttime = time.time()
    args = parsecommandline()

    nsimulations = args.nsimulations
    mu = args.mu
    rec = args.rec
    popSize = args.popSize
    seqLen = args.seqLen
    ns = args.ns
    nsdistargs = args.nsdistargs
    diploidsamplesize = args.diploidsamplesize
    modelpath = args.modelpath
    
    nSeqs = args.nSeqs
    output_dir = args.output_dir
    savefile = args.savefile
    
    density2Ns = args.density2Ns

    nfsfs, sfsfs, fsfs_ratio, seeds = simulateSFSslim(args,nsimulations = nsimulations, mu = mu, rec = rec, popSize = popSize, seqLen = seqLen, 
                                                      ns = ns, density2Ns = density2Ns, nsdistargs = nsdistargs, diploidsamplesize = diploidsamplesize,foldit = args.foldit,
                                                      modelpath = modelpath, nSeqs = nSeqs, output_dir = output_dir, savefile = True,
                                                      donotcleandir=args.donotcleandir)

    print("Run finished!!!")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        main(['-h'])
    else:
        main(sys.argv[1:])
        