"""
    simple script to combine two SFS files into one file to be input to SFRatios.py or a plotting script
"""
import os.path as op
import sys
import argparse

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a",dest="neutsfsfn",required=True,type=str, help = "neutral SFS filepath")
    parser.add_argument("-b",dest="neut_label",default = "",type=str, help = "neutral SFS header text line, optional, if not used, then the line from the SFS file is used")
    parser.add_argument("-c",dest="selectsfsfn",required=True,type=str, help = "select SFS filepath")
    parser.add_argument("-d",dest="select_label",default = "",type=str, help = "select SFS header text line, optional, if not used, then the line from the SFS file is used")
    parser.add_argument("-o",dest="outfilen",required=True,type=str, help = "output SR_Ratios.py file")
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    return args


if __name__ == '__main__':
    args = parsecommandline()
    lines = []
    temp = open(args.neutsfsfn,'r').readlines()
    for x in temp[:2]:
        lines.append(x.strip())
    temp = open(args.selectsfsfn,'r').readlines()
    for x in temp[:2]:
        lines.append(x.strip())
    if args.neut_label != '':
        lines[0] = args.neut_label
    if args.select_label != '':
        lines[2] = args.select_label
    fout = open(args.outfilen,"w")
    for line in lines:
        fout.write(line +  "\n")
    fout.close()
        
