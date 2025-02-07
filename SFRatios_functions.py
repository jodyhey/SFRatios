"""
Program:  SFRatios_functions.py
Author: Jody Hey

    poisson random field SFS work 
    a module of various functions

    models:
        fix2Ns0,fixed2Ns,normal,lognormal,gamma,uni3fixed, uni3float

    sfs lists:
        all sfs lists begin with 0 in position 0 
        there is no position for a count where all chromosomes have the allele (i.e. fixed in the sample)

    counting with k chromosomes
        with unfolded, there are k - 1 values  so n_unf = k-1
            this means an unfolded list has length 1 + k - 1 == k 
        with folded it is more complicated, 
            if k is even,  a count of k//2  has no folding partner. e.g. if k is 4  then when folding the bins for counts 1, 3 are summed
                but the bin for 2 is not added to anything.  
                so n_f  has k//2 values  i.e. n_f = k//2  in a list that has length 1 + n_f
            if k is odd, a count of k//2 does have a folding partner, e.g. if k is 5 then bins 1,3 are summed, as are 2,4 
                so n_f has k//2 values,  i.e. n_f = k//2   in a list that has length 1 + n_f
            so the folded value of n_f for even k is the same as for odd count of k + 1 
    nc : # of chromosomes
    n_unf : nc - 1
    n_f : nc // 2 

        
"""
import sys
import numpy as np
import  mpmath 
import math
import scipy
import scipy.integrate
from scipy.optimize import golden
from scipy.special import erf, gamma, gammainc,gammaincc,seterr,_sf_error
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
import os
sys.path.append("./")
current_dir = os.path.dirname(os.path.abspath(__file__)) # the directory of this python script
# Add the current directories to sys.path
sys.path.append(current_dir)
print(current_dir)
# Get the directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create error log in the same directory as the script
# Configure logging
# logging.basicConfig(
#     filename = os.path.join(current_dir, "SFS_Ratios_functions_errorlog.txt"),
#     level=logging.ERROR,
#     format='%(asctime)s - %(levelname)s\n%(message)s\nStack trace:\n%(exc_info)s\n'
#     )


# Test if we can write to the directory
log_path = os.path.join(current_dir, "SFS_Ratios_functions_errorlog.txt")
try:
    # Try to open the file for writing to test permissions
    with open(log_path, 'a'):
        pass
    
    logging.basicConfig(
        filename=log_path,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s\n%(message)s\nStack trace:\n%(exc_info)s\n'
    )
except (IOError, PermissionError) as e:
    print(f"Warning: Could not set up logging to {log_path}")
    print(f"Error: {str(e)}")
    # Fall back to logging to stderr
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s\n%(message)s\nStack trace:\n%(exc_info)s\n'
    )

def handle_error(e: Exception, context: str = ""):
    logging.error(context, exc_info=True)  # exc_info=True includes stack trace
    print(f"{context}\nError: {str(e)}", file=sys.stderr)
    sys.exit(1)

# warnings.simplefilter("error", RuntimeWarning) # make RuntimeWarnings errors so we can trap them
# seterr(all='raise')
   


# tried out alternative parameterizations, in which the true mean of the random variable is used as a parameter
# but this also requires that any simulator that is used implement this,  and SLiM would be hard to rejigger for this
lognormalparameterization = "Lmean"  # Tmean for actual mean of the variable (typically negative) or Lmean for mean of log of variable 
gammaparameterization =  "Tmean" # "m0mean"  # m0mean for the mean of the variable if the variable had a max of 0 (this value is always positive even if true mean is negative), Tmean for true mean accounting for max2Ns != 0

#constants
sqrt2 =pow(2,1/2)
sqrt_2_pi = np.sqrt(2 * np.pi)
sqrt_pi_div_2 = np.sqrt(np.pi/2)

# constants for x values in 2Ns integration see getXrange()
minimum_2Ns_location = -10000 # the lowest value for the mode  or mean of a continuous distribution that will be considered,  lower than this an the likelihood just returns inf 
discrete3lowerbound = -1000
discrete3upperbound = 10
discrete3_xvals = np.concatenate([np.linspace(discrete3lowerbound,-5,20),np.linspace(-4.9,-1 + 1e-3,10),np.linspace(-1,1 - 1e-3,20),np.linspace(1,discrete3upperbound,20)]) # only for uni3fixed distribution 
lowerbound_2Ns_integration = -100000 # exclude regions below this from integrations,  arbitrary but saves some time

# reduced the lengths of arrays used for numerical integration
# fillnegxvals=np.flip(-np.logspace(0,5, 100)) # -1 to -100000
# himodeintegraterange = np.logspace(-2,5,50)  # tried himodeintegraterange = np.logspace(-2,5,100)  but not much improvement

# shorter versions, seem to work 
himodeintegraterange = np.logspace(-2,5,30)
fillnegxvals=np.flip(-np.logspace(0,5, 50)) # -1 to -100000

#  replaced with coth_without1()
# def coth(x):
#     """
#         save a bit of time for large values of x 
#     """
#     if abs(x) > 15: # save a bit of time 
#         return -1.0 if x < 0 else 1.0
#     else:
#         return np.cosh(x)/np.sinh(x) 

@lru_cache(maxsize=100000) # helps a little bit with coth()
def coth_without1(x):
    """
        a couple relations:
            coth(-x) = -coth(x)
            when abs(x) > 10,  coth(x)~ 1+2*exp(-2x)  

        returns coth(x)  with the 1 term removed.  i.e. if x is positive 1 is subtracted,  if x is negative -1 is subtracted

        when abs(x) > 10,  coth(x) is very near sign(x)*1 (i.e. 1 with the sign of x) 
        however there is a component of coth(x) for abs(x) > 10 that is many decimals away that we need to have. 
        this is necessary for prf_selection_weight()  that has (coth(x) - 1) x H  terms in it where H is a hyp1f1 value 
        e.g. if we just converged cothx-1 to 0   for x > 10,  then the hyp1f1 terms go away,  but it turns out to matter 
        
    """
    if abs(x) >= 300:
        if x > 0:
            return 2*mpmath.exp(-2*x)
        else:
            return -2*mpmath.exp(2*x)
    elif abs(x) >= 10:
        if x > 0:
            return 2*math.exp(-2*x)
        else:
            return -2*math.exp(2*x)
    else:
        if x > 0:
            return np.cosh(x)/np.sinh(x) -1 
        else:
            return np.cosh(x)/np.sinh(x) + 1

# turns out this does not speed things up
# @lru_cache(maxsize=100000) 
# def exp_cache(x):
#     return math.exp(x) if -745 <= x <= 709 else mpmath.exp(x)

#cached error function 
@lru_cache(maxsize=100000) #helps a little bit with erf()
def erf_cache(x):
    """
        absolute values above 6 just return 1 with the sign of the x 
        else, try scipy
        else try approximations suggested by claude
        mpmath is slow 

    """
    # if abs(x) >= 6:
    #     return math.copysign(1, x)
    # else:
    #     try:
    #         return scipy.special.erf(x)
    #     except  (ValueError, ArithmeticError, _sf_error.SpecialFunctionError):
    #         return mpmath.erf(x) # for more precision 
    if abs(x) > 6:
        return math.copysign(1, x)
    try:
        return scipy.special.erf(x)
    except  (ValueError, ArithmeticError, _sf_error.SpecialFunctionError) as e:
        # For very small x, erf(x) ≈ (2/√π) * x
        if abs(x) < 1e-7:
            return (2/math.sqrt(math.pi)) * x
        # For intermediate values use log-space calculation
        sign = 1 if x >= 0 else -1
        return sign * math.sqrt(1 - math.exp(-4*x*x/math.pi))   
    except Exception as e:
        handle_error(e, "erf_cache: x {}".format(x))
    
@lru_cache(maxsize=5000000) # helps a lot,  e.g. factor of 2 or more for hyp1f1 
def cached_hyp1f1(a, b, z):
    """
        try scipy then mpmath 
    """
    # try:
    #     return scipy.special.hyp1f1(a, b, z)
    # except  (ValueError, ArithmeticError, _sf_error.SpecialFunctionError):
    #     return float(mpmath.hyp1f1(a, b, z))
    try:
        temp = scipy.special.hyp1f1(a, b, z)
    except  (ValueError, ArithmeticError, _sf_error.SpecialFunctionError):
        try:
            return mpmath.hyp1f1(a, b, z)
        except  (ValueError, ArithmeticError, _sf_error.SpecialFunctionError):
            if z < 0:
                return 0
            else:
                return math.inf
    except Exception as e:  
        handle_error(e, "cached_hyp1f1 a {} b{} z{}".format(a,b,z))
    if temp in (math.nan,math.inf,math.inf):
        return mpmath.hyp1f1(a, b, z)
    else:
        return temp

def clear_cache(outf = False):
    cache_functions = {
        # "exp": exp_cache,
        "coth": coth_without1,
        "erf": erf_cache,
        "hyp1f1": cached_hyp1f1
    }
    for name, func in cache_functions.items():
        try:
            func.cache_clear()
        except AttributeError:
            pass  # Function doesn't have cache_info or cache_clear  
        except Exception as e:
            handle_error(e, "Unexpected error")   
    if outf:
        outf.write("\nCaching results:\n")
        for name, func in cache_functions.items():
            try:
                outf.write(f"{name} cache: {func.cache_info()}\n")
            except AttributeError:
                pass  # Function doesn't have cache_info or cache_clear
            except Exception as e:
                handle_error(e, "Unexpected error") 

def logprobratio(alpha,beta,z):  # called by NegL_SFSRATIO_estimate_thetaS_thetaN(),  not up to date as of 7/1/2024
    """
        returns the log of the probability of a ratio z of two normal densities when for each normal density the variance equals the mean 
        is called from other functions,  where alpha, beta and the ratio z have been calculated for a particular frequency bin

        two versions of this function
       
        Díaz-Francés, E. and F. J. Rubio paper expression (1)

        Kuethe DO, Caprihan A, Gach HM, Lowe IJ, Fukushima E. 2000. Imaging obstructed ventilation with NMR using inert fluorinated gases. Journal of applied physiology 88:2279-2286.
        the gamma in the paper goes away because it is equal to 1/alpha^(1/2) when we assume the normal distributions have mean equal to variance 

        The functions are similar, but the Díaz-Francés, E. and F. J. Rubio  works much better overall,  e.g. LLRtest check and ROC curves are much better 
        However the Díaz-Francés and Rubio function gives -inf often enough to cause problems for the optimizer,  so we set final probability p = max(p,1e-50)

    """
    try:
        delta = beta
        beta = alpha
        z2 = z*z
        delta2 = delta*delta
        z1 = 1+z
        z2b1 = 1+z2/beta
        z2boverb = (z2+beta)/beta
        betasqroot = math.sqrt(beta)
        ratiotemp = -(1+beta)/(2*delta2)
        temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
        ratiotemp2 = mpmath.fdiv(mpmath.fneg(mpmath.power(-z-beta,2)),(2*delta2*(z2+beta)))
        temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
        temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
        temp2 = mpmath.fdiv(temp2num,temp2denom )
        p = mpmath.fadd(temp1,temp2)
        logp = math.log(p) if p > 1e-307  else float(mpmath.log(p))
        return logp 
    
    except Exception as e:
        handle_error(e, "exception in logprobratio: alpha {} beta {} z {}".format(alpha,beta,z)) 

def intdeltalogprobratio(alpha,z,thetaNspace,nc,i,foldxterm):
    """
        integrates over the delta term in the probability of a ratio of two normal distributions
    """
    def safe_log1p_exp(x): # suggested by claude so we can handle log of exp with floats 
        if x > 709:  # np.log(np.finfo(np.float64).max)
            return x  # Because log(1 + exp(x)) ≈ x for large x
        else:
            return np.log1p(np.exp(x))

    def newrpobdeltadef(z,beta, delta,betasqroot): # 12/11/2024 faster, but not as forgiving of value ranges 
        try:
            delta2 = delta*delta
            log_term1 = -(1+beta)/(2*delta2) - np.log(math.pi*z2b1*betasqroot)
            log_term2 = powz12/(2*delta2*z2b1)
            # erftemp = stable_erf(z1/(sqrt2*sqz2b1*delta))
            erftemp = erf_cache(z1/(sqrt2*sqz2b1*delta))
            log_temp3 = log_term2 + np.log(sqrt_pi_div_2*z1*abs(erftemp))
            log_result = log_term1 + safe_log1p_exp(log_temp3 - np.log(delta*sqz2b1))
            # result = math.exp(log_result)
            result = mpmath.exp(log_result) # log_result is very often out of float range 
            return result
        except  (ValueError, ArithmeticError) as e:
            error_msg = str(e).lower()
            if "overflow" in error_msg:
                if log_result < 0:  # Check if result would be large negative
                    return -np.finfo(np.float64).max
                return np.finfo(np.float64).max
            elif "underflow" in error_msg:
                return 0.0  
            return 0.0    
        except Exception as e:
            handle_error(e,"newrprobdelta z {} beta {}  delta {} betasqroot {}".format(z,beta,delta,betasqroot))

    def rprobdelta(z,beta,delta,betasqroot): # older,  slower,  can handle wider range of values 
        delta2 = delta*delta

        try:
            forexptemp = -(1+beta)/(2*delta2)
            exptemp = math.exp(forexptemp) if -745 <= forexptemp <= 709 else mpmath.exp(forexptemp)
            try:
                if betasqroot==0.0:
                    betasqroot=mpmath.sqrt(beta)
                try :
                    temp1 = exptemp/(math.pi*z2b1*betasqroot)
                except  (ValueError, ArithmeticError):
                    temp1 = mpmath.fdiv(exptemp,(math.pi*z2b1*betasqroot))
            except  (ValueError, ArithmeticError):
                temp1 = mpmath.fdiv(exptemp,(math.pi*z2b1*betasqroot))
            fortemp2 = powz12/(2*delta2*z2b1)
            temp2 = math.exp(fortemp2) if -745 <= fortemp2 <= 709 else mpmath.exp(fortemp2)
            erftemp = erf_cache(z1/(sqrt2*sqz2b1*delta))
            try:
                temp3 = temp2*sqrt_pi_div_2*z1*erftemp
                if temp3 in (0,math.inf):
                    temp3 = mpmath.fmul(temp2,sqrt_pi_div_2*z1*erftemp)
            except  (ValueError, ArithmeticError):
                temp3 = mpmath.fmul(temp2,sqrt_pi_div_2*z1*erftemp)
            try:
                temp4 = 1 + (temp3/(delta*sqz2b1))
                if temp4 in (0,math.inf):
                    temp4 = mpmath.fadd(1,mpmath.fdiv(temp3,(delta*sqz2b1)))
            except  (ValueError, ArithmeticError,ZeroDivisionError):
                temp4 = mpmath.fadd(1,mpmath.fdiv(temp3,(delta*sqz2b1)))
            except :
                pass
            p = mpmath.fmul(temp1,temp4)
            return p
        except Exception as e:
            handle_error(e,"rprobdelta z {} beta {}  delta {} betasqroot {}".format(z,beta,delta,betasqroot))
    
    beta = float(alpha)
    z2 = z*z
    z1 = 1+z
    powz12 = pow(z1,2)
    z2b1 = 1+z2/beta
    sqz2b1 = math.sqrt(z2b1)
    betasqroot = math.sqrt(beta) 
    uy = thetaNspace * nc /(i*(nc -i)) if foldxterm else thetaNspace/i    
    deltavals = 1/np.sqrt(uy)

    # using the older rprobdelta() function
    # rprob_density_values = np.array([rprobdelta(z,beta,delta,betasqroot) for delta in deltavals])
    # rprob = np.trapz(rprob_density_values,thetaNspace)
    # logrprob = math.log(rprob) if rprob > 1e-307 else float(mpmath.log(rprob))
    #using the newer rprobdelta() function
    altrprob_density_values = np.array([newrpobdeltadef(z,beta,delta,betasqroot) for delta in deltavals])
    altrprob = np.trapz(altrprob_density_values,thetaNspace)
    logrprob = math.log(altrprob) if altrprob > 1e-307 else float(mpmath.log(altrprob))
    # print(z,beta,rprob,logrprob,altlogrprob)

    return logrprob


def ratio_expectation(p,i,max2Ns,nc,dofolded,misspec,densityof2Ns):# not used in awhile 7/11/2024
    """
    get the expected ratio for bin i given a set of parameter values 
    """
    
    def ztimesprobratio(z,alpha,beta,doneg):
        """
            called by ratio_expectation()
            for getting the expected ratio
            doneg is True when called by golden()
            False when doing the integration
        """

        # Díaz-Francés, E. and F. J. Rubio 
        try:
            delta = beta
            beta = alpha
            z2 = z*z
            delta2 = delta*delta
            z1 = 1+z
            z2b1 = 1+z2/beta
            z2boverb = (z2+beta)/beta
            betasqroot = math.sqrt(beta)
            ratiotemp = -(1+beta)/(2*delta2)
            temp1 = mpmath.fdiv(mpmath.exp(ratiotemp),(math.pi*z2b1*betasqroot))
            ratiotemp2 =   (-pow(z-beta,2)/(2*delta2*(z2+beta)))
            temp2num = mpmath.fmul(mpmath.fmul(mpmath.exp(ratiotemp2),z1), mpmath.erf(z1/(sqrt2 * delta * math.sqrt(z2boverb))))
            try:
                temp2denom = sqrt_2_pi *  betasqroot * delta*pow(z2boverb,1.5)
            except RuntimeWarning:
                temp2denom = mpmath.fmul(sqrt_2_pi,mpmath.fmul(betasqroot,mpmath.fmul(delta,mpmath.power(z2boverb,1.5))))
            temp2 = mpmath.fdiv(temp2num,temp2denom )
            p = float(mpmath.fadd(temp1,temp2))
        except Exception as e:
            handle_error(e,"ztimesprobratio: z {} alpha {} beta {}  doneg {}".format(z,alpha,beta,doneg))
        if p < 0.0:
            return 0.0
        if doneg:
            return -p*z
        return  p*z    

    foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not, assumes nc is even, in which case the last bin is not folded 
    thetaN = p[0]
    thetaS = p[1]
    g = (p[2],p[3])
    ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,g,max2Ns)
    intval = integrate2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,g_xvals,densityadjust)
    ux = thetaS*intval 
    uy = thetaN*nc /(i*(nc -i)) if foldxterm else thetaN/i    
    alpha = ux/uy
    sigmay = math.sqrt(uy)
    beta = 1/sigmay
    peak = golden(ztimesprobratio,args=(alpha,beta,True), brack = (0,1000)) # use ratio interval of 0 1000 to start 
    x = scipy.integrate.quad(ztimesprobratio,-10,peak*10,args=(alpha,beta, False)) # use interval of 0 to 10* the peak location 
    return x[0]


def prf_selection_weight(nc, i, g, dofolded, misspec):
    """
        Poisson random field selection weight for g=2Ns for bin i  (folded or unfolded)
        this is the function you get when you integrate the product of two terms:
             (1) WF term for selection    (1 - E^(-2 2 N s(1 - q)))/((1 - E^(-2 2 N s)) q(1 - q))  
             (2) bionomial sampling formula for i copies,  given allele frequency q 
        over the range of allele frequencies 
        use cached hyp1f1 function
    """    
    if abs(g) < 1e-3:
        if dofolded:
            us = nc / (i * (nc - i))
        else:
            if misspec:
                us = (1 - misspec) / i + misspec / (nc - i)
            else:
                us = 1 / i
    else:
        tempc_without1 = coth_without1(g)  # coth(g) - 1 if g > 0 else coth(g) + 1 

        if dofolded:
            temph1 = cached_hyp1f1(i, nc, 2*g)
            temph2 = cached_hyp1f1(nc - i, nc, 2*g)
            temph = temph1 + temph2
            if g > 0:
                if tempc_without1 == 0 or temph == math.inf: # it seems that when temph is inf  tempc_without1 is very near 0 
                    us = (nc / (2 * i * (nc - i))) * 4 
                else:
                    us = (nc / (2 * i * (nc - i))) * (2 + 2 * (1+tempc_without1) - (tempc_without1 * temph))

            else:
                us = (nc / (2 * i * (nc - i))) * (2 + 2 * (-1+tempc_without1) - (tempc_without1 * temph - 2*temph))
        else:
            if misspec in (None, False, 0.0):
                temph = cached_hyp1f1(i, nc, 2*g)
                if g > 0:
                    if tempc_without1 == 0 or temph == math.inf: # it seems that when temph is inf  tempc_without1 is very near 0 
                        us = (nc / (2 * i * (nc - i))) * 2                 
                    else:
                        us = (nc / (2 * i * (nc - i))) * (1 + (1+tempc_without1) - (tempc_without1 * temph))
                else:
                    us = (nc / (2 * i * (nc - i))) * (1 + (-1+tempc_without1) - (tempc_without1 * temph - 2*temph))
            else:
                temph1 = cached_hyp1f1(i, nc, 2*g)
                temph2 = cached_hyp1f1(nc - i, nc, 2*g)
                temph = (1 - misspec) * temph1 + misspec * temph2
                if g > 0:
                    if tempc_without1 == 0 or temph == math.inf: # it seems that when temph is inf  tempc_without1 is very near 0 
                        us = (2) * (nc / (2 * i * (nc - i))) 
                    else:
                        us = (1 +(1+tempc_without1) + (- tempc_without1 * temph)) * (nc / (2 * i * (nc - i)))
                else:

                    us = (1 +(-1+tempc_without1) - (tempc_without1 * temph - 2*temph)) * (nc / (2 * i * (nc - i)))
    if us  < 0.0:
        return 0.0
    else:
        return us 


def getXrange(densityof2Ns,g,max2Ns,xpand = False):
    """
        get the range of integration for the density of 2Ns, build an array with values either side of the mode 
        if xpand,  get more intervals for numerical integration 
    """
    def prfdensity(xval,g,first_call=False):
        if densityof2Ns=="lognormal":   
            if lognormalparameterization == "Lmean": 
                mean = g[0]
                std_dev = g[1]
                x = float(max2Ns-xval)
                p = (1 / (x * std_dev * sqrt_2_pi)) * np.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))
                if p==0.0:
                    p= float(mpmath.fmul(mpmath.fdiv(1, (x * std_dev * sqrt_2_pi)), mpmath.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))))
            else: #"Tmean"
                mean = g[0]  # true mean of the lognormally distributed rv,  not the mean of the log of the random variable 
                sigma = g[1] 
                x = float(max2Ns-g)
                max_minus_mean = max2Ns-mean
                if first_call == True:
                    prfdensity.sigmasquare = np.square(sigma)
                    prfdensity.sigmasquare8 = 8*prfdensity.sigmasquare
                    prfdensity.sigmasqrt2pi = sqrt_2_pi*sigma
                    sigmasquareterm = prfdensity.sigmasquare
                    sigmasquare8term = prfdensity.sigmasquare8
                    sigmasqrt2piterm =prfdensity.sgimasqrt2pi
                else:
                    sigmasquareterm = prfdensity.sigmasquare
                    sigmasquare8term = prfdensity.sigmasquare8
                    sigmasqrt2piterm =prfdensity.sgimasqrt2pi
                p = np.exp(-np.square(sigmasquareterm + 2 * np.log(x)-2*np.log(max_minus_mean))/sigmasquare8term)/(sigmasqrt2piterm*x)
                if p==0.0:
                    p = mpmath.fdiv(
                        mpmath.exp(
                            -mpmath.fdiv(
                                mpmath.power(
                                    mpmath.fadd(
                                        sigmasquareterm,
                                        mpmath.fsub(
                                            mpmath.fmul(2, mpmath.log(x)),
                                            mpmath.fmul(2, mpmath.log(max_minus_mean))
                                        )
                                    ),
                                    2
                                ),
                                sigmasquare8term
                            )
                        ),
                        mpmath.fmul(sigmasqrt2piterm, x)
                    )           

        elif densityof2Ns=="gamma": # reparamaterize with shape and mean 
            if gammaparameterization == "m0mean": 
                mean = g[0]
                shape = g[1]
                x = float(max2Ns-xval)
                if first_call == True: # save some time by saving this part of the gamma density wich does not depend on x 
                    prfdensity.holdgammaterm = (mean/shape)**(-shape) / math.gamma(shape)
                    gammaterm = prfdensity.holdgammaterm 
                else:
                    gammaterm = prfdensity.holdgammaterm             
                try:
                    p = np.exp(-shape*x/mean) * (x**(shape-1)) * gammaterm
                except  (ValueError, ArithmeticError):
                    p = 0.0                 
            else: # "Tmean"
                mean = g[0]
                shape = g[1]
                max_minus_mean = max2Ns-mean
                x = float(max2Ns-xval)
                if first_call == True: # save some time by saving this part of the gamma density wich does not depend on x 
                    prfdensity.holdgammaterm = (max_minus_mean/shape)**(-shape) / math.gamma(shape)
                    gammaterm = prfdensity.holdgammaterm 
                else:
                    gammaterm = prfdensity.holdgammaterm             
                try:
                    p = np.exp(-shape*x/max_minus_mean) * (x**(shape-1)) * gammaterm
                except  (ValueError, ArithmeticError):
                    p = 0.0        

        elif densityof2Ns == "normal":
            mean = g[0]
            std_dev = g[1]
            p = (1 / (std_dev * sqrt_2_pi)) * math.exp(-(1/2)*((xval- mean)/std_dev)**2)
        elif densityof2Ns=="uni3fixed":
            if xval < -1:
                p=g[0]/999
            elif xval<= 1:
                p = g[1]/2
            else:
                p = (1-g[0] - g[1])/9
        elif densityof2Ns=="uni3float":
            """
            g[0] - proportion of lowest bin
            g[1] - proportion of middle bin
            g[2] - upper cutoff for lowest bin
            g[3] - upper cutoff for middle bin
            (g[2] - discrete3lowerbound) = lower uniform divisor 
            (g[3] - g[2]) = middle uniform divisor 
            (discrete3upperbound - g[3]) = upper uniform divisor 

            """
            if xval < g[2]:
                lower_divisor = (g[2] - discrete3lowerbound)
                p=g[0]/lower_divisor
            elif xval<= g[3]:
                middle_divisor = (g[3] - g[2])
                p = g[1]/middle_divisor
            else:
                upper_divisor = (discrete3upperbound - g[3])
                p = (1-g[0] - g[1])/upper_divisor
        return p

    if densityof2Ns not in ("uni3fixed","uni3float"):
        if xpand:
            numSDs = 7
            himodeR = np.logspace(-4,5,500)
            lowmodenumint = 20
        else:
            numSDs = 5 
            himodeR = himodeintegraterange
            lowmodenumint = 8
            # lowmodenumint = 5

        if densityof2Ns=="normal":
            mode = ex = g[0]
            sd = g[1]
        elif densityof2Ns == "lognormal":
            if lognormalparameterization == "Lmean": 
                stdev_squared = g[1]*g[1]
                sd = math.sqrt((math.exp(stdev_squared) - 1) * math.exp(2*g[0] + stdev_squared))
                ex =  -math.exp(g[0] + stdev_squared/2)
                mode = -math.exp(g[0] - stdev_squared)
                if max2Ns:
                    ex += max2Ns
                    mode += max2Ns   
            else: #"Tmean"   
                mean = g[0]
                sigmasq = g[1]*g[1]
                sd = math.sqrt( (math.exp(sigmasq)-1)* math.pow(max2Ns-mean,2))
                ex = mean
                mode = math.exp(-3*sigmasq/2)*(mean + max2Ns*(math.exp(3*sigmasq/2)-1))                    

        elif densityof2Ns == "gamma":
            if gammaparameterization == "m0mean": 
                mean = g[0]
                shape = g[1]
                sd = math.sqrt(mean*mean/shape)
                ex = -mean
                mode = 0.0 if shape < 1 else -(shape-1)*mean/shape
                if max2Ns:
                    mode += max2Ns 
                    ex += max2Ns                
            else: # "Tmean"
                mean = g[0]
                shape = g[1]
                ex = mean
                sd = math.sqrt(math.pow(max2Ns-mean,2)/shape)
                if shape <= 1:
                    mode = max2Ns
                else:
                    mode = (max2Ns + (shape-1)*mean)/shape



        #build an array of 2Ns values for integration
        listofarrays = []    
        # print("in getxrange ",ex,sd,mode)
        if sd > 1000: # if variance is very large,  use a fixed log spaced array on both sides of the mode
            temp = np.flip(mode - himodeR)
            temp[0] = lowerbound_2Ns_integration # put lowerbound_2Ns_integration in there. it will get sorted later
            listofarrays = [temp,np.array([mode]),mode + himodeR]
        else:
            #build using np.linspace over chunks of the range. The closer to mode, the more finely spaced
            # range is from lowerbound_2Ns_integration to (max2Ns - 1e-8)
            sd10 = min(10,sd/10)
            sd100 = min(1,sd/100)
            listofarrays = [np.linspace(mode-sd,mode-sd10,10),np.linspace(mode-sd10,mode-sd100,10),np.linspace(mode-sd100,mode,10),np.linspace(mode,mode+sd100,10),np.linspace(mode+sd100,mode+sd10,10),np.linspace(mode+sd10,mode+sd,10)]
            for i in range(2,numSDs+1):
                listofarrays.insert(0,np.linspace(mode-i*sd,mode-(i-1)*sd,lowmodenumint))
                listofarrays.append(np.linspace(mode+(i-1)*sd,mode+ i*sd,lowmodenumint))
        # Concatenate the arrays
        xvals = np.concatenate(listofarrays)
        # Sort the concatenated array
        xvals = np.sort(xvals)
        # Remove duplicates 
        xvals = np.unique(xvals)
        if densityof2Ns in ("lognormal","gamma"):
            # Filter for values less than or equal to max2Ns
            xvals = xvals[xvals <= max2Ns]
            # Replace the highest value with max2Ns - 1e-8
            upperlimitforintegration = max2Ns - 1e-8
            try:
                if xvals[-1] >= upperlimitforintegration:  
                    xvals[-1] = upperlimitforintegration
                else: # append upperlimitforintegration
                    xvals = np.append(xvals, upperlimitforintegration)
            except  (ValueError, ArithmeticError):
                pass
        # remove any values less than lowerbound_2Ns_integration
        xvals = xvals[xvals >= lowerbound_2Ns_integration]
        # tack on log scaled negative values down to lowerbound_2Ns_integration
        # only inlclude those less than xvals[0]*1.1
        # the 1.1 factor is to avoid something being right next to xvals[0]
        #scipy.integrate.simpson() seems to handle xvals arrays that are even in length
        # compares well with np.trapz().   integrands closer to 1,  about the same speed 
        # more testing showed it simpson is worse. 
        if len(xvals) >= 10:
            xvals = np.concatenate([fillnegxvals[fillnegxvals < (xvals[0]*1.1)], xvals])
        else: 
            xvals = np.concatenate([fillnegxvals, xvals[xvals > -1]])
    elif densityof2Ns=="uni3fixed":
        xvals = discrete3_xvals
        ex = -11*(-1 + 92*g[0] + g[1])/2
        m2 = (-110*g[1]/3) + 37 + 333630*g[0]
        sd = math.sqrt(m2 - ex*ex)
        mode = np.nan
    elif densityof2Ns=="uni3float":
        xvals = np.concatenate([discrete3_xvals,[g[2],g[3]]])
        # Sort the concatenated array
        xvals = np.sort(xvals)
        # Remove duplicates 
        xvals = np.unique(xvals)
        mean_density_values = np.array([x*prfdensity(x,g, first_call=None) for x in xvals])
        ex = np.trapz(mean_density_values,xvals)
        var_density_values = np.array([x*x*prfdensity(x,g, first_call=None) for x in xvals])
        var = np.trapz(var_density_values,xvals)
        try:
            sd = math.sqrt(var - ex*ex)
        except  (ValueError, ArithmeticError):
            sd = np.nan
        mode = np.nan
    density_values = np.array([prfdensity(x,g,first_call=True if j == 0 else False) for j,x in enumerate(xvals)])
    densityadjust = np.trapz(density_values,xvals)
    return ex,mode,sd,densityadjust,xvals
 
def prfdensityfunction(g,densityadjust,nc ,i,args,max2Ns,densityof2Ns,foldxterm,misspec,first_call=True):
    """
    returns the product of poisson random field weight for a given level of selection (g) and a probability density for g 
    used for integrating over g 
    if foldxterm is true,  then it is a folded distribution AND two bins are being summed
    """
    us = prf_selection_weight(nc ,i,g,foldxterm,misspec)
    if densityof2Ns=="lognormal":   
        if lognormalparameterization == "Lmean": 
            mean = args[0]  # true mean of the lognormally distributed rv,  not the mean of the log of the random variable 
            std_dev = args[1] 
            x = float(max2Ns-g)
            p = ((1 / (x * std_dev * sqrt_2_pi)) * np.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2)))/densityadjust
            if p==0.0:
                p= float(mpmath.fmul(mpmath.fdiv(1, (x * std_dev * sqrt_2_pi)), mpmath.exp(-(np.log(x)- mean)**2 / (2 * std_dev**2))))/densityadjust
        else: #"Tmean"
            mean = args[0]  # true mean of the lognormally distributed rv,  not the mean of the log of the random variable 
            sigma = args[1] 
            x = float(max2Ns-g)
            max_minus_mean = max2Ns-mean
            if first_call == True:
                prf_selection_weight.sigmasquare = np.square(sigma)
                prf_selection_weight.sigmasquare8 = 8*prf_selection_weight.sigmasquare
                prf_selection_weight.sigmasqrt2pi = sqrt_2_pi*sigma*densityadjust
                sigmasquareterm = prf_selection_weight.sigmasquare
                sigmasquare8term = prf_selection_weight.sigmasquare8
                sigmasqrt2piterm =prf_selection_weight.sgimasqrt2pi
            else:
                sigmasquareterm = prf_selection_weight.sigmasquare
                sigmasquare8term = prf_selection_weight.sigmasquare8
                sigmasqrt2piterm =prf_selection_weight.sgimasqrt2pi
            p = np.exp(-np.square(sigmasquareterm + 2 * np.log(x)-2*np.log(max_minus_mean))/sigmasquare8term)/(sigmasqrt2piterm*x)
            if p==0.0:
                p = mpmath.fdiv(
                    mpmath.exp(
                        -mpmath.fdiv(
                            mpmath.power(
                                mpmath.fadd(
                                    sigmasquareterm,
                                    mpmath.fsub(
                                        mpmath.fmul(2, mpmath.log(x)),
                                        mpmath.fmul(2, mpmath.log(max_minus_mean))
                                    )
                                ),
                                2
                            ),
                            sigmasquare8term
                        )
                    ),
                    mpmath.fmul(sigmasqrt2piterm, x)
                )           

    elif densityof2Ns=="gamma":
        if gammaparameterization == "m0mean": 
            mean = args[0]
            shape = args[1]
            x = float(max2Ns-g)
            if first_call == True: # save some time by saving this part of the gamma density wich does not depend on x 
                prf_selection_weight.holdgammaterm =(mean/shape)**(-shape) / (math.gamma(shape)*densityadjust)
                gammaterm = prf_selection_weight.holdgammaterm 
            else:
                gammaterm = prf_selection_weight.holdgammaterm 
            try:
                p = np.exp(-shape*x/mean) * (x**(shape-1)) * gammaterm
            except  (ValueError, ArithmeticError):
                p = float(mpmath.exp(-shape*x/mean) * mpmath.power(x,shape-1)) * gammaterm

        else: # "Tmean"
            mean = args[0]
            shape = args[1]
            max_minus_mean = max2Ns-mean
            x = float(max2Ns-g)
            if first_call == True: # save some time by saving this part of the gamma density wich does not depend on x 
                prf_selection_weight.holdgammaterm =(max_minus_mean/shape)**(-shape) / (math.gamma(shape)*densityadjust)
                gammaterm = prf_selection_weight.holdgammaterm 
            else:
                gammaterm = prf_selection_weight.holdgammaterm 
            try:
                p = np.exp(-shape*x/max_minus_mean) * (x**(shape-1)) * gammaterm
            except  (ValueError, ArithmeticError):
                p = float(mpmath.exp(-shape*x/max_minus_mean) * mpmath.power(x,shape-1) * gammaterm)
     
    elif densityof2Ns=="normal": # shouldn't need densityadjust for normal
        mu = args[0]
        std_dev= args[1]
        p = np.exp((-1/2)* ((g-mu)/std_dev)**2)/(std_dev *sqrt_2_pi)
    elif densityof2Ns=="uni3fixed":
        if g < -1:
            p=args[0]/999
        elif g<= 1:
            p = args[1]/2
        else:
            p = (1-args[0]-args[1])/9
    elif densityof2Ns=="uni3float":
            """
            g[0] - proportion of lowest bin
            g[1] - proportion of middle bin
            g[2] - upper cutoff for lowest bin
            g[3] - upper cutoff for middle bin
            (g[2] - discrete3lowerbound) = lower uniform divisor 
            (g[3] - g[2]) = middle uniform divisor 
            (discrete3upperbound - g[3]) = upper uniform divisor 
            """
            if g < -args[2]:
                lower_divisor = (args[2] - discrete3lowerbound)
                p=args[0]/lower_divisor
            elif g <= args[3]:
                middle_divisor = (args[3] - args[2])
                p = args[1]/middle_divisor
            else:
                upper_divisor = (discrete3upperbound - args[3])
                p = (1-args[0] - args[1])/upper_divisor            
    pus = p*us
    if pus < 0.0 or np.isnan(p):
        return 0.0
    return pus


def integrate2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,xvals,densityadjust):
    """
        xvals is a numpy array 
    """
    density_values = np.array([prfdensityfunction(x,densityadjust,nc ,i,g,max2Ns,
                                                  densityof2Ns,foldxterm,misspec,
                                                    first_call=True if j == 0 else False) for j,x in enumerate(xvals)])
    intval = np.trapz(density_values,xvals)
    return intval
    
def NegL_SFS_Theta_Ns(p,nc,dofolded,includemisspec,maxi,counts): 
    """
        for fisher wright poisson random field model,  with with selection or without
        if p is a float,  then the only parameter is theta and there is no selection
        else p is a list (2 elements) with theta and Ns values 
        counts begins with a 0
        returns the negative of the log of the likelihood for a Fisher Wright sample 
    """
    def L_SFS_Theta_Ns_bin_i(i,count): 
        if isinstance(p,(float, int)): # p is simply a theta value,  no g  
            theta = p
            if theta <= 0:
                return -math.inf
            un = theta*nc/(i*(nc - i)) if dofolded else theta/i
            temp = -un +  math.log(un)*count - math.lgamma(count+1)
        else:
            theta = p[0]
            if theta <= 0:
                return -math.inf
            g = p[1]
            misspec = p[2] if includemisspec else 0.0 
            us = theta * prf_selection_weight(nc,i,g,dofolded,misspec)
            try:
                temp = -us +  math.log(us)*count - math.lgamma(count+1)
            except Exception as e:
                handle_error(e,"L_SFS_Theta_Ns_bin_i problem: nc {} i {} g {} us {} theta {} count {}".format(nc,i,g,us,theta,count))
        return temp     

    assert(counts[0]==0)
    sum = 0
    k = len(counts) if maxi in (None,False) else min(len(counts),maxi)
    for i in range(1,k):
        # sum += L_SFS_Theta_Ns_bin_i(p,i,nc,dofolded,counts[i])
        sum += L_SFS_Theta_Ns_bin_i(i,counts[i])
    return -sum 


def NegL_SFS_ThetaS_densityNs(p,max2Ns,nc ,dofolded,includemisspec,densityof2Ns,counts):
    """
        basic PRF likelihood
        returns negative of likelihood for the SFS 
        unknowns:
            thetaS
            terms for 2Ns density
    """
    sum = 0
    thetaS = p[0]
    term1 = p[1]
    term2 = p[2]
    misspec = p[3] if includemisspec else 0.0 
    ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,(term1,term2),max2Ns)
    for i in range(1,len(counts)):
        intval = integrate2Ns(densityof2Ns,max2Ns,(term1,term2),nc,i,dofolded,misspec,g_xvals,densityadjust)
        us = thetaS*intval 
        sum += -us + math.log(us)*counts[i] - math.lgamma(counts[i]+1)        
    return -sum    
 
def NegL_SFSRATIO_estimate_thetaS_thetaN(p,nc,dofolded,includemisspec,densityof2Ns,onetheta,max2Ns,estimate_pointmass0,maxi,zvals): 
    """
        returns the negative of the log of the likelihood for the ratio of two SFSs
        estimates Theta values,  not their ratio

        densityof2Ns in fix2Ns0,fixed2Ns,normal,lognormal,gamma,uni3fixed 
        onetheta in True, False
        max2Ns  is either None,  or a fixed max value 
        estimate_pointmass0 in True, False
        
        replaces:
            def NegL_SFSRATIO_thetaS_thetaN_fixedNs(p,nc ,dofolded,zvals,nog,estimate_pointmass0)
            def NegL_SFSRATIO_thetaS_thetaN_densityNs_max2Ns(p,max2Ns,nc ,maxi,dofolded,densityof2Ns,zvals)
            
        returns negative of likelihood using the probability of the ratio 
        unknown     # params
        thetaN,thetaS 1 if onetheta else 2 
        Ns terms    2 if densityNs is not fixed2Ns 1 
        max2Ns      1 if densityof2Ns is in ("lognormal","gamma") and max2Ns is None else 0 
        pointmass0  1 if estimate_pointmass0 else 0 

        handles dofolded 
    """
    def calc_bin_i(i,z): 
        if densityof2Ns in ("fix2Ns0","fixed2Ns"):
            try:
                if z==math.inf or z==0.0:
                    return 0.0
                uy = thetaN*nc /(i*(nc -i)) if dofolded else thetaN/i     
                if g == 0:
                    alpha = thetaS/thetaN
                else:
                    ux = thetaS*prf_selection_weight(nc,i,g,dofolded,misspec)
                    if densityof2Ns == "fixed2Ns":
                        if estimate_pointmass0:
                            ux = thetaS*pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux # mass at 0 times neutral weight + (1- mass at 0) times selection weight

                    alpha = ux/uy
                sigmay = math.sqrt(uy)
                beta = 1/sigmay
                return logprobratio(alpha,beta,z)
            except (ValueError, ArithmeticError) as e:
                return -math.inf            
            except Exception as e:
                handle_error(e,"calc_bin_i: densityof2Ns {} i {} z {}".format(densityof2Ns,i,z))
        else:
            try:
                ux = thetaS*integrate2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,g_xvals,densityadjust)
                if estimate_pointmass0:
                    ux = thetaS*pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux
                uy = thetaN*nc /(i*(nc -i)) if foldxterm else thetaN/i    
                alpha = ux/uy
                sigmay = math.sqrt(uy)
                beta = 1/sigmay
                return logprobratio(alpha,beta,z)   
            except (ValueError, ArithmeticError) as e:
                return -math.inf
            except Exception as e:
                handle_error(e,"calc_bin_i: densityof2Ns {} i {} z {}".format(densityof2Ns,i,z))
            
    try:
        p = list(p)
    except :
        p = [p]
        pass  # if this function is called by scipy minimize then p is just a scalar 
    if onetheta:
        thetaN = thetaS = p[0]
        unki = 1
    else:
        thetaN = p[0]
        thetaS = p[1]
        unki = 2
    if densityof2Ns == "fixed2Ns":
        g = p[unki]
        unki += 1
    elif densityof2Ns == "fix2Ns0":
        g = 0
    else:
        g = (p[unki],p[unki+1])
        if densityof2Ns=="uni3fixed":
            if ((0 < g[0] < 1) == False) or ((0 < g[1] < 1) == False) or ((g[0] + g[1]) >= 1):
                return math.inf
        holdki = unki
        unki += 2
    if estimate_pointmass0:
        pm0 = p[unki]
        unki += 1
       
    if max2Ns==None and densityof2Ns in ("lognormal","gamma"):
        max2Ns = p[unki]
        unki += 1
      
    if includemisspec:
        misspec = p[unki] 
        unki += 1
    else:
        misspec = 0.0        
    # if densityof2Ns not in ("fixed2Ns","uni3fixed","fix2Ns0"):
    if densityof2Ns not in ("fixed2Ns","fix2Ns0"):
        ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,g,max2Ns)

    sum = 0
    summaxi = maxi if maxi not in (None,False) else len(zvals)
    # for i in range(1,len(zvals)):
    for i in range(1,summaxi):
        foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not 
        temp =  calc_bin_i(i,zvals[i])
        sum += temp
        if sum == -math.inf:
            return math.inf
    return -sum   

def NegL_SFSRATIO_estimate_thetaratio(p,nc,dofolded,includemisspec,densityof2Ns,fix_theta_ratio,max2Ns,estimate_pointmass0,maxi,thetaNspace,zvals): 
    """
        returns the negative of the log of the likelihood for the ratio of two SFSs
        first parameter is the ratio of mutation rates
        sidesteps the theta terms by integrating over thetaN in the probability of the ratio (i.e. calls intdeltalogprobratio())

        densityof2Ns in fix2Ns0,fixed2Ns,normal,lognormal,gamma,uni3fixed 
        fixthetaratio is either None, or a fixed value for the ratio 
        max2Ns  is either None,  or a fixed max value 
        estimate_pointmass0 in True, False

        replaces:
            NegL_SFSRATIO_ratio_fixedNs
            NegL_SFSRATIO_ratio_densityNs
            NegL_SFSRATIO_ratio_densityNs_pointmass0

        returns negative of likelihood using the probability of the ratio 
        unknown     # params
        ratio       0 if fix_theta_ratio is not None else 1 
        Ns terms    2 if densityNs is not None else 1 
        max2Ns      1 if densityof2Ns is in ("lognormal","gamma") and max2Ns is None else 0 
        pointmass0  1 if estimate_pointmass0 else 0 

        handles dofolded 
    """
    def calc_bin_i(i,z): 
        if densityof2Ns in ("fixed2Ns","fix2Ns0"):
            try:
                # if z==math.inf or z==0.0:
                #     return 0.0
                if z==math.inf:
                    return 0.0                
                if g == 0:
                    alpha = thetaratio
                else:
                    ux = prf_selection_weight(nc,i,g,foldxterm,misspec)
                    if densityof2Ns == "fixed2Ns":
                        if estimate_pointmass0:
                            ux = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux # mass at 0 times neutral weight + (1- mass at 0) times selection weight

                    alpha = thetaratio*ux/(nc /(i*(nc -i)) if foldxterm else 1/i )

                returnval = intdeltalogprobratio(alpha,z,thetaNspace,nc,i,foldxterm)      

                return returnval

            except (ValueError, ArithmeticError) as e:
                return -math.inf
            except Exception as e:
                handle_error(e,"calc_bin_i: densityof2Ns {} i {} z {}".format(densityof2Ns,i,z))

        else:
            try:
                ux = integrate2Ns(densityof2Ns,max2Ns,g,nc,i,foldxterm,misspec,g_xvals,densityadjust)
                if estimate_pointmass0:
                    ux = pm0*(nc /(i*(nc -i)) if foldxterm else 1/i ) + (1-pm0)*ux # mass at 0 times neutral weight + (1- mass at 0) times selection weight
                alpha = thetaratio*ux/(nc /(i*(nc -i)) if foldxterm else 1/i )
                return intdeltalogprobratio(alpha,z,thetaNspace,nc,i,foldxterm)        
            except (ValueError, ArithmeticError) as e:
                return -math.inf
            except Exception as e:
                handle_error(e,"calc_bin_i: densityof2Ns {} i {} z {}".format(densityof2Ns,i,z))

    if isinstance(p,(int,float)):
        p = [p]
    else:
        p = list(p)
    unki = 0
    if fix_theta_ratio in (None,False):
        thetaratio = p[0]
        unki = 1
    else:
        thetaratio = fix_theta_ratio
    if densityof2Ns == "fixed2Ns":
        g = p[unki]
        unki += 1
    elif densityof2Ns  == "fix2Ns0":
        g = 0.0
    else:
        if densityof2Ns=="uni3fixed":
            g = (p[unki],p[unki+1])
            if ((0 < g[0] < 1) == False) or ((0 < g[1] < 1) == False) or ((g[0] + g[1]) >= 1):
                return math.inf
            unki += 2
        elif densityof2Ns=="uni3float":
            g = (p[unki],p[unki+1],p[unki+2],p[unki+3])
            if ((0 < g[0] < 1) == False) or ((0 < g[1] < 1) == False) or ((g[0] + g[1]) >= 1) or  not (discrete3lowerbound < g[2] < g[3] < discrete3upperbound):
                return math.inf
            unki += 4 
        elif densityof2Ns in ("normal","gamma","lognormal"):
            g = (p[unki],p[unki+1])
            unki += 2 


    if estimate_pointmass0:
        pm0 = p[unki]
        unki += 1
    
    if max2Ns==None and densityof2Ns in ("lognormal","gamma"):
        max2Ns = p[unki]
        unki += 1

    if includemisspec:
        misspec = p[unki] 
        unki += 1
    else:
        misspec = 0.0        
    # if densityof2Ns in ("normal","lognormal","gamma"):
    if densityof2Ns in ("normal","lognormal","gamma","uni3fixed","uni3float"):
        ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,g,max2Ns)
    else:
        densityadjust = 1.0
    sum = 0
    summaxi = maxi if maxi not in (None,False) else len(zvals)
    # for i in range(1,len(zvals)):
    for i in range(1,summaxi):
        foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not 
        temp =  calc_bin_i(i,zvals[i])
        sum += temp
        # print("{:.4f} ".format(temp),end="")
        if sum == -math.inf:
            return math.inf
    if densityof2Ns in ("normal","lognormal","gamma"): 
        # kludgy,  penalize if ex or mode is too low, penalty is 10^6 times the difference 
        if ex <  minimum_2Ns_location:
            sum -=  (minimum_2Ns_location - ex)*1e6
        elif mode <  minimum_2Ns_location :
            sum -= (minimum_2Ns_location - mode)*1e6
  
    return -sum   

def NegL_CodonPair_SFSRATIO_estimate_thetaratio(p,nc,dofolded,includemisspec,fix_theta_ratio,neg2Ns,thetaNspace,zvals): 
    """
        returns the negative of the log of the likelihood for the ratio of two codonpair SFSs
        i.e. both numerater and denominator of the ratio are from codon SFSs 
        put one with more counts in the denominator 

        as with a neutral denominator first parameter is the ratio of mutation rates

        assumes a single 2Ns value that can range from neg to pos 
        fixthetaratio is either None, or a fixed value for the ratio 

        returns negative of likelihood using the probability of the ratio 
        unknown     # params
        ratio       0 if fix_theta_ratio is not None else 1 
        2Ns value   1 
        handles dofolded 
    """
    def calc_bin_i(i,z): 
        try:
            if z==math.inf or z==0.0:
                return 0.0
            if g == 0:
                alpha = thetaratio
            else:
                if neg2Ns:
                    numweight = prf_selection_weight(nc,i,-g,foldxterm,False) # don't do misspecification now
                    denomweight = prf_selection_weight(nc,i,g,foldxterm,False) # don't do misspecification now
                else:
                    numweight = prf_selection_weight(nc,i,g,foldxterm,False) # don't do misspecification now
                    denomweight = prf_selection_weight(nc,i,-g,foldxterm,False) # don't do misspecification now

                weightratio = (((1-misspec)*numweight + misspec*denomweight)/((1-misspec)*denomweight + misspec*numweight)) if includemisspec else numweight/denomweight

                alpha = thetaratio*weightratio
            return intdeltalogprobratio(alpha,z,thetaNspace,nc,i,foldxterm)   
        except (ValueError, ArithmeticError) as e:
            return -math.inf
        except Exception as e:
            handle_error(e,"calc_bin_i: i {} z {}".format(i,z))            
            
    if isinstance(p,(int,float)):
        p = [p]
    else:
        p = list(p)
    unki = 0
    if fix_theta_ratio in (None,False):
        thetaratio = p[0]
        unki = 1
    else:
        thetaratio = fix_theta_ratio
    g = p[unki]
    unki += 1
    if includemisspec:
        misspec = p[unki] 
        unki += 1
    else:
        misspec = 0.0        
    sum = 0
    for i in range(1,len(zvals)):
        foldxterm = dofolded and i < nc //2 # True if summing two bins, False if not 
        temp =  calc_bin_i(i,zvals[i])
        sum += temp
        if sum == -math.inf:
            return math.inf
    return -sum   


def simsfs_continuous_gdist(theta,max2Ns,nc,misspec,maxi,densityof2Ns, params,pm0, returnexpected,pmmass = None,pmval = None):
    """
    nc  is the # of sampled chromosomes 

    simulate the SFS under selection, assuming a PRF Wright-Fisher model 
    uses a distribution of g (2Ns) values 
    gdist is "lognormal" or "gamma" ,params is two values

    return folded and unfolded    
    """
    sfs = [0]*nc 
    for i in range(1,nc):
        ex,mode,sd,densityadjust,g_xvals = getXrange(densityof2Ns,params,max2Ns)
        ux = integrate2Ns(densityof2Ns,max2Ns,tuple(params),nc,i,False,misspec,g_xvals,densityadjust)        
        if pm0 not in (False,None):
            ux = pm0/i + (1-pm0)*ux
        elif pmmass not in (False,None):
            ux = pmmass * prf_selection_weight(nc ,i,pmval,False,misspec) + (1-pmmass) * ux
        sfsexp = theta*ux
        assert sfsexp>= 0
        if returnexpected:
            sfs[i] = sfsexp
        else:
            sfs[i] = np.random.poisson(sfsexp)

    sfsfolded = [0] + ([sfs[i] + sfs[nc-i] for i in range(1,nc//2)] + [sfs[nc//2]] if nc % 2 == 0 else  [sfs[i] + sfs[nc-i] for i in range(1,1+nc//2)])
    if maxi:
        assert maxi < nc , "maxi setting is {} but nc  is {}".format(maxi,nc )
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded

def simsfs(theta,g,nc , misspec,maxi, returnexpected,pm0=None,pmmass=None,pmval=None):
    """
        nc  is the # of sampled chromosomes 
        pm0 is a point mass at 0
        pmmass is a point mass at pmval 

        simulate the SFS under selection, assuming a PRF Wright-Fisher model 
        uses just a single value of g (2Ns), not a distribution
        if returnexpected,  use expected values, not simulated
        generates,  folded and unfolded for Fisher Wright under Poisson Random Field
        return folded and unfolded 
    """

    if g==0:
        if misspec in (None,False, 0.0):
            sfsexp = [0]+[theta/i for i in range(1,nc )]
        else:
            sfsexp = [0]+[theta*((1-misspec)/i +(misspec/(nc-i)) ) for i in range(1,nc )]
    else:
        sfsexp = [0]
        for i in range(1,nc ):
            ux = theta*prf_selection_weight(nc,i,g,False,misspec)
            if isinstance(pm0,float):
                ux = (theta* (pm0 /i)) + (1-pm0)*ux
            elif isinstance(pmmass,float):
                ux = theta * pmmass * prf_selection_weight(nc,i,pmval,False,misspec) +  (1-pmmass)*ux
            sfsexp.append(ux)
            # sfsexp.append(u*theta)    
    if returnexpected:
        sfs = sfsexp
    else:    
        try:
            sfs = [np.random.poisson(expected) for expected in sfsexp]
        except Exception as e:
            handle_error(e,"simsfs theta {} g {} sfsexp {} z {}".format(theta,g,simsfs))   

    sfsfolded = [0] + ([sfs[i] + sfs[nc-i] for i in range(1,nc//2)] + [sfs[nc//2]] if nc % 2 == 0 else  [sfs[i] + sfs[nc-i] for i in range(1,1+nc//2)])
    if maxi:
        assert maxi < nc , "maxi setting is {} but nc  is {}".format(maxi,nc )
        sfs = sfs[:maxi+1]
        sfsfolded = sfsfolded[:maxi+1]            
    return sfs,sfsfolded


def simsfsratio(thetaN,thetaS,max2Ns,nc ,maxi,dofolded,misspec,densityof2Ns,params,pm0, returnexpected, thetaratio,pmmass = None,pmval = None):
    """
     nc  is the # of sampled chromosomes 

    simulate the ratio of selected SFS to neutral SFS
    if returnexpected,  use expected values, not simulated
    if gdist is None,  params is just a g value,  else it is a list of distribution parameters
    if a bin of the neutral SFS ends up 0,  the program stops

    if ratio is not none, thetaS = thetaratio*thetaN

    pm0 is point mass 0,  as of 2/4/2024 used only by run_one_pair_of_SFSs.py
    """
    
    nsfs,nsfsfolded = simsfs(thetaN,0,nc ,misspec,maxi,returnexpected)

    if thetaratio is not None:
        thetaS = thetaN*thetaratio
    if densityof2Ns == "fixed2Ns": 
        ssfs,ssfsfolded = simsfs(thetaS,params[0],nc ,misspec,maxi,returnexpected,pm0=pm0,pmmass=pmmass,pmval=pmval)
    else:
        # ssfs,ssfsf = SRF.simsfs_continuous_gdist(theta,max2Ns,nc,None,None,densityof2Ns,g,None,False)
        ssfs,ssfsfolded = simsfs_continuous_gdist(thetaS,max2Ns,nc ,misspec,maxi,densityof2Ns,params,pm0,returnexpected,pmmass = pmmass,pmval = pmval)
    if dofolded:
        ratios = [math.inf if nsfsfolded[j] <= 0.0 else ssfsfolded[j]/nsfsfolded[j] for j in range(len(nsfsfolded))]
        return nsfsfolded,ssfsfolded,ratios
    else:
        ratios = [math.inf if nsfs[j] <= 0.0 else ssfs[j]/nsfs[j] for j in range(len(nsfs))]
        return nsfs,ssfs,ratios



