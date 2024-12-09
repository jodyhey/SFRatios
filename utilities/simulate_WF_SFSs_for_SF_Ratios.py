"""
    Author: Jody Hey

    basic script for simulating data sets for SF_Ratios.py
    generated under Wright-Fisher with a 2Ns value or density 

"""
import os.path as op
import random
import sys
# Get the parent directory
parent_dir = op.abspath(op.join(op.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
import SF_Ratios_functions as SRF
import numpy as np 
import argparse
import os.path as op 

def run(args):

    random.seed(args.ranseed) 
    np.random.seed(args.ranseed) 
    thetaS = args.thetas[0]
    if args.getratio or args.getneutral:
        thetaN = args.thetas[1]
    else:
        thetaN = 1000 #dummy 
    if args.densityof2Ns in ("lognormal","gamma"):
        g = args.densityparams[:2]
        max2Ns = args.densityparams[2]
    elif args.densityof2Ns in ("normal","fixed2Ns"):
        g = args.densityparams
        max2Ns = None
    else:
        print("error")
        exit()
    pointmassstring = '' if args.pointmass_loc_and_value == None else "_pmL_{}_pmV_{}".format(args.pointmass_loc_and_value[0],args.pointmass_loc_and_value[1])
    
    fname = "{}_{}_{}{}{}{}{}.txt".format(args.outfilelabel,
                                            args.nc,
                                            args.densityof2Ns,
                                            "_folded" if args.folded else "_unfolded",
                                            "_qS{}_qN{}".format(args.thetas[0],args.thetas[1]) if (args.getratio or args.getneutral) else "_q{}".format(args.thetas[0]),
                                            ''.join([f"_p{num}" for num in g]),
                                            pointmassstring
                                            )
    foutname = op.join(args.outfiledir,fname)

    if args.pointmass_loc_and_value:
        nsfs,ssfs,ratios = SRF.simsfsratio(thetaN,thetaS,max2Ns,args.nc,None,
                                        args.folded,None,args.densityof2Ns,g,None,False,None,pmmass=args.pointmass_loc_and_value[0],pmval=args.pointmass_loc_and_value[1])
    else:
        nsfs,ssfs,ratios = SRF.simsfsratio(thetaN,thetaS,max2Ns,args.nc,None,
                                        args.folded,None,args.densityof2Ns,g,None,False,None)
    
    f = open(foutname,'w')
    f.write("Generated by simulate_WF_SFSs_for_SF_Ratios.py\n  command line: {}\n".format(args.commandstring))
    f.write("Arguments:\n")
    for key, value in vars(args).items():
        f.write(" {}: {}\n".format(key,value))
    if args.clengths:
        ssfs[0] = args.clengths[0] - sum(ssfs)
        nsfs[0] = args.clengths[1] - sum(nsfs)
    f.write("Selected SFS\n")
    f.write("{}\n".format(" ".join(list(map(str,ssfs)))))
    if args.getratio or args.getneutral:
        f.write("Neutral SFS\n")
        f.write("{}\n".format(" ".join(list(map(str,nsfs)))))
    if args.getratio:
        f.write("Ratios\n")
        f.write(' '.join(f"{x:.4f}" for x in ratios) + '\n')
    f.close()

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest="densityparams", type=float, nargs="+", required=True, help="One or more integer values, fixed2Ns:2Ns lognormal: mu sigma max2Ns  gamma: scale mean max2Ns normal: mu sigma ")
    parser.add_argument("-e",dest="ranseed",default=1,type=int,help="random number seed integer, default = 1")
    parser.add_argument("-d",dest="densityof2Ns",default = "fixed2Ns",type=str,help="gamma, lognormal, normal, fixed2Ns")
        # parser.add_argument("-e",dest="includemisspec",action="store_true",default=False,help=" for unfolded, include a misspecification parameter") 
        # parser.add_argument("-o",dest="fixmode0",action="store_true",default=False,help="fix the mode of 2Ns density at 0, only works for lognormal and gamma")    
    parser.add_argument("-f",dest="folded",action="store_true", help="fold the sfs  (unfolded is default state)")    
    parser.add_argument("-l", dest="outfilelabel",required=True,type = str, help="string for start of out file name")
    parser.add_argument("-L",dest="clengths",default=None,type=int, nargs="+", help=" optional, to fill 0 bin,  selected chromosome length, then neutral chromosome length")
    parser.add_argument("-n",dest="nc",required=True, type=int,help=" number of chromosomes")
    parser.add_argument("-o", dest="outfiledir",required=True,type = str, help="Path for SFS file")
    parser.add_argument("-p", dest="pointmass_loc_and_value", type=float, nargs="+", default = None, help=" optional point mass loc and value")
    parser.add_argument("-q", dest="thetas", type=float, nargs="+", required=True, help="selected theta then neutral theta if getneutral or getratio")
    parser.add_argument("-r",dest="getratio",action="store_true", help="also simulated neutral SFS and get ratio")    
    parser.add_argument("-u",dest="getneutral",action="store_true", help="simulate neutral SFS as well ")   

    args  =  parser.parse_args(sys.argv[1:])  
    return args

if __name__ == '__main__':
    args = parsecommandline()
    args.commandstring = " ".join(sys.argv[1:])
    run(args)