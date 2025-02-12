"""
reads output from one more output files from SFRatios.py  
generates a csv file with one row for each file 
reports runtime terms, data set stuff, parameter estimates, etc etc
usage: get_SF_Ratio_output_summaries.py [-h] -d FILEDIRECTORY -f OUTFILENAME [-p POPLABEL] [-x SKIPFILESTRING]

options:
  -h, --help         show this help message and exit
  -d FILEDIRECTORY   path to files
  -f OUTFILENAME     outfilename
  -p POPLABEL        population/sample label
  -x SKIPFILESTRING  skip files with this string in their name, e.g. '_PM_'

"""
import os
import os.path as op
import csv
from datetime import datetime
import sys
import argparse
import math


def calcdistances(lines):
    sumsq = 0.0
    c = 0
    for i,xrow in enumerate(lines):
        if len(xrow) < 2:
            break
        c += 1
        row = xrow.strip().split()
        sumsq += pow(float(row[5]) -float(row[2]),2)
    RMSE = math.sqrt(sumsq/c)
    EucDis = math.sqrt(sumsq)
    return EucDis,RMSE


# Function to parse each .out file
def parse_file(filepath,headers):
    """
        crappy code
        files can vary a lot in terms of their model parameters 
    """
    data = {key: None for key in headers}
    with open(filepath, 'r') as file:
        lines = file.readlines()
    print(filepath)
    last_modified_datetime = op.getmtime(filepath)
    last_modified_datetime = datetime.fromtimestamp(last_modified_datetime)
    data["fileMtime"] = f"{last_modified_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    li = 0
    while True:

        if "Command line arguments:" in lines[li] or lines[li][:10] == "Arguments:":
            li += 1
            break
        li += 1
    data["PMmode"]  = "FALSE"
    data["nc"]  = "?"
    data["OptMethods"] = ''
    while len(lines[li]) > 2 :
        if "sfsfilename:" in lines[li]:
            data["File"] = op.split(filepath)[-1]
            data["NonS/Syn"] = "NONSYN" if "NONSYN" in data["File"].upper() else "SYN"
        if "fix_theta_ratio" in lines[li]:
            data["FixQR"] = lines[li].split(":")[1].strip()
        if "densityof2Ns" in lines[li]:
            data["Density"] = lines[li].split(":")[1].strip()
        if "max2Ns" in lines[li] and "estimate" not in lines[li]:
            data["SetMax2Ns"] = lines[li].split(":")[1].strip()
        if "fixmode0" in lines[li]:
            data["FixMode0"] = lines[li].split(":")[1].strip()
        if "estimate_pointmass0:" in lines[li]:
            temp = lines[li].split(":")[1].strip()
            if temp == "True":
                data["PMmode"] = "PM_at_0"
        if "estimate_pointmass:" in lines[li]:
            temp = lines[li].split(":")[1].strip()
            if temp == "True":
                data["PMmode"] = "PM_M_and_V"
        if "numparams" in lines[li]:
            data["numparams"] = int(lines[li].split(":")[1].strip())
        if "nc:" in lines[li]:
            data["nc"] = int(lines[li].split(":")[1].strip())
        if "foldstatus:" in lines[li]:
            data["FoldStatus"] = lines[li].split(":")[1].strip()
        if "thetaratiorange"  in lines[li]: 
            data["thetaRatiorange"] = lines[li].split(":")[1].strip()
        if "optimizetries:" in lines[li] and int(lines[li].split(":")[1].strip()) > 0:
            data["OptMethods"] += "," if len(data["OptMethods"]) > 0 else ""
            data["OptMethods"] += "reg{}".format(lines[li].split(":")[1].strip())
        if "basinhoppingopt:" in lines[li] and "True" in lines[li]:
            data["OptMethods"] += "," if len(data["OptMethods"]) > 0 else ""
            data["OptMethods"] += "BH"
        if "dualannealopt:" in lines[li] and "True" in lines[li]:
            data["OptMethods"] += "," if len(data["OptMethods"]) > 0 else ""
            data["OptMethods"] += "DA"
        li += 1

    while "trial\t" not in lines[li]:

        li += 1
        if li >= len(lines):
            break
    li += 1

# Qratio	p1	p2	estMax2Ns	pm0_mass	pm_mass	pm_val
    tempcounter = 0
    tempatlike = False
    while data["Mode"] == None: #"Mode is the last thing read here "
        if li >= len(lines):
            break
        line = lines[li]
        if "Optimization:" in line:
            data["BestOptMethod"] = line.strip().split()[1]
        if "AIC\t" in  line:
            data["AIC"] = float(line.strip().split()[1])
        if "likelihood" in line and "thetaratio" not in line:
            data["Lklhd"] = line.split('\t')[1].strip() if '\t' in line else None
            tempcounter = 0  # First four lines after "basinhopping" are liklihood, thetaratio, and then the two parameters
            tempatlike = True
        elif "distribution check" in line:
            start = line.find("distribution check:") + len("distribution check: ")
            end = line.find(" ", start)
            data["IntCheck"] = float(line[start:end])
        if tempatlike:
            if tempcounter == 1:
                if  data["FixQR"] in (None,"None"):
                    data["Qratio"] = line.split('\t')[1].strip() if '\t' in line else None
                    data["Qratio_CI"] = line.split('\t')[2].strip() if '\t' in line else None
                else:
                    data["Qratio"] = None
                    data["Qratio_CI"] = None
                    tempcounter += 1
            if tempcounter == 2:
                data["p1"] = line.split('\t')[1].strip() if '\t' in line else None
                data["p1_CI"] = line.split('\t')[2].strip() if '\t' in line else None                
            if tempcounter == 3:
                if data["Density"] != 'single2Ns':
                    data["p2"] = line.split('\t')[1].strip() if '\t' in line else None
                    data["p2_CI"] = line.split('\t')[2].strip() if '\t' in line else None                
                else:
                    data["p2"] = data["p2_CI"] = None
        if "max2Ns" in line:
            data["estMax2Ns"] = line.split('\t')[1].strip() if '\t' in line else None
            data["estMax2Ns_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "pm0" in line:
            data["pm0_mass"] = line.split('\t')[1].strip() if '\t' in line else None
            data["pm0_mass_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "pm_mass" in line:
            data["pm_mass"] = line.split('\t')[1].strip() if '\t' in line else None
            data["pm_mass_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "pm_val" in line:
            data["pm_val"] = line.split('\t')[1].strip() if '\t' in line else None
            data["pm_val_CI"] = line.split('\t')[2].strip() if '\t' in line else None
        elif "expectation" in line:
            data["Mean"] = line.split('\t')[1].strip() if '\t' in line else None
            if data["Mean"] == 'nan':
                if data["Density"]=='single2Ns':
                    if data["PMmode"] == "PM_at_0":
                        data["Mean"] = (1- float(data["pm0_mass"] ))* float(data["p1"]) 
                        if float(data["pm0_mass"]) > 0.5:
                            data["Mode"] = 0.0
                        else:
                            data["Mode"] = data["p1"]
                    elif data["PMmode"] == "PM_M_and_V":
                        data["Mean"] = (1- float(data["pm_mass"] ))* float(data["p1"])  + float(data["pm_mass"] ) * float(data["pm_val"]) 
                        if float(data["pm_mass"]) > 0.5:
                            data["Mode"] = data["pm_val"]
                        else:
                            data["Mode"] = data["p1"]
                    else:
                        data["Mean"] = data["p1"]
                        data["Mode"] = data["p1"]
            elif data['Density'] in ("discrete3","uni3fixed"):
                data["Mode"] = "na"
            elif data["Density"] == "uni3float":
                data["Mode"] = "na"
                
        elif "mode" in line:
            data["Mode"] = line.split('\t')[1].strip() if '\t' in line else None
        tempcounter += 1  
        li += 1
    try: # calculate euclidean distance and RMSE if oberved and expected values are in the file 
        while "i\tDataN\t" not in lines[li]:
            li += 1
        data["EucDis"],data["RMSE"] = calcdistances(lines[li+2:])
    except:
        pass        
    while data["RunTime"] == None:
        if li >= len(lines):
            break
        line = lines[li]
        if line.startswith("Time taken:"):
            time_str = line[len("Time taken:"):].strip()
            parts = time_str.split(', ')
            hours, minutes, seconds = 0, 0, 0.0
            for part in parts:
                if "hour" in part:
                    hours = int(part.split()[0])  # Extract number of hours
                elif "minute" in part:
                    minutes = int(part.split()[0])  # Extract number of minutes
                elif "second" in part:
                    seconds = float(part.split()[0])  # Extract number of seconds
            rounded_seconds = round(seconds)
            data["RunTime"] = f"{hours}h{minutes}m{rounded_seconds}s"
        li += 1


    return [data[header] for header in headers]

def run(args):
    # Define the directory and output file
    ddir = args.filedirectory
    output_file = args.outfilename
    output_path = op.join(ddir, output_file)

    # Define the headers for the CSV file
    headers = [
        "File","fileMtime","RunTime","FoldStatus","OptMethods","BestOptMethod","nc","IntCheck","NonS/Syn", "Density", "SetMax2Ns", "FixQR","FixMode0", "PMmode","numparams",
        "Lklhd","AIC","Mean", "Qratio","Qratio_CI", "p1","p1_CI", "p2","p2_CI","estMax2Ns","estMax2Ns_CI",
        "pm0_mass","pm0_mass_CI","pm_mass","pm_mass_CI","pm_val","pm_val_CI","Mode", "EucDis","RMSE","thetaRatiorange"
        ]
    if args.poplabel is not None:
        headers.insert(0,"dataset")
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for filename in os.listdir(ddir):
            if filename.endswith('.out') and (args.skipfilestring is None or args.skipfilestring not in filename):
                filepath = os.path.join(ddir, filename)
                row = parse_file(filepath,headers)
                if args.poplabel is not None:
                    # row.insert(0,args.poplabel)
                    row[0] = args.poplabel
                writer.writerow(row)
    print(f"Summary CSV file has been created at {output_path}.")
    return

def parsecommandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="filedirectory",required=True,type = str, help="path to files")
    parser.add_argument("-f",dest="outfilename",required=True,type=str,help="outfilename")
    parser.add_argument("-p",dest="poplabel",default = None,help="population/sample label")
    parser.add_argument("-x",dest="skipfilestring",default = None,help="skip files with this string in their name, e.g. '_PM_'")
    args  =  parser.parse_args(sys.argv[1:])  
    args.commandstring = " ".join(sys.argv[1:])
    return args


if __name__ == '__main__':
   
    args = parsecommandline()
    run(args)