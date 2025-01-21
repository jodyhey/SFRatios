import sys
import numpy as np
import  mpmath 
import math
import scipy
import scipy.integrate
import scipy.special
from scipy.special import erf, gamma, gammainc,gammaincc,seterr
from functools import lru_cache
# from scipy import integrate
import warnings
import traceback
import os
import mpmath as mp
import math
import random
random.seed(1)
np.random.seed(1)
sqrt2 =pow(2,1/2)
sqrt_2_pi = np.sqrt(2 * np.pi)
sqrt_pi_div_2 = np.sqrt(np.pi/2)

   
def safe_log1p_exp(x):
    if x > 709:  # np.log(np.finfo(np.float64).max)
        return x  # Because log(1 + exp(x)) ≈ x for large x
    else:
        return np.log1p(np.exp(x))

def stable_erf(x):
    if abs(x) > 6:
        return math.copysign(1, x)
    try:
        return scipy.special.erf(x)
    except:
        # For very small x, erf(x) ≈ (2/√π) * x
        if abs(x) < 1e-7:
            return (2/math.sqrt(math.pi)) * x
        # For intermediate values use log-space calculation
        sign = 1 if x >= 0 else -1
        return sign * math.sqrt(1 - math.exp(-4*x*x/math.pi))   
    
def xnewrpobdeltadef (z,beta, delta,betasqroot):
    try:
        delta = float(delta)
        beta = float(beta)
        delta2 = delta*delta
        z2 = z*z
        z1 = 1+z
        powz12 = pow(z1,2)
        z2b1 = 1+z2/beta
        sqz2b1 = math.sqrt(z2b1)        
        
        log_term1 = -(1+beta)/(2*delta2) - np.log(math.pi*z2b1*betasqroot)
        log_term2 = powz12/(2*delta2*z2b1)
        erftemp = stable_erf(z1/(sqrt2*sqz2b1*delta))
        log_temp3 = log_term2 + np.log(sqrt_pi_div_2*z1*abs(erftemp))
        log_result = log_term1 + safe_log1p_exp(log_temp3 - np.log(delta*sqz2b1))
        result = np.exp(log_result)
        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "overflow" in error_msg:
            if log_result < 0:  # Check if result would be large negative
                return -np.finfo(np.float64).max
            return np.finfo(np.float64).max
        elif "underflow" in error_msg:
            return 0.0  
        return 0.0    
       


def erf_cache(x):
    """
        absolute values above 6 just return 1 with the sing of the x 
        else, try scipy
        else try mpmath 

    """
    if abs(x) >= 6:
        return math.copysign(1, x)
    else:
        try:
            return scipy.special.erf(x)
        except:
            return mpmath.erf(x)
 


def rprobdelta(z,beta,delta,betasqroot):
    
    delta = float(delta)
    beta = float(beta)
    delta2 = delta*delta
    z2 = z*z
    z1 = 1+z
    powz12 = pow(z1,2)
    z2b1 = 1+z2/beta
    sqz2b1 = math.sqrt(z2b1)       

    try:
        forexptemp = -(1+beta)/(2*delta2)
        exptemp = math.exp(forexptemp) if -745 <= forexptemp <= 709 else mpmath.exp(forexptemp)
        try:
            if betasqroot==0.0:
                betasqroot=mpmath.sqrt(beta)
            try:
                temp1 = exptemp/(math.pi*z2b1*betasqroot)
            except:
                temp1 = mpmath.fdiv(exptemp,(math.pi*z2b1*betasqroot))
        except:
            temp1 = mpmath.fdiv(exptemp,(math.pi*z2b1*betasqroot))
        fortemp2 = powz12/(2*delta2*z2b1)
        temp2 = math.exp(fortemp2) if -745 <= fortemp2 <= 709 else mpmath.exp(fortemp2)
        erftemp = erf_cache(z1/(sqrt2*sqz2b1*delta))
        try:
            temp3 = temp2*sqrt_pi_div_2*z1*erftemp
            if temp3 in (0,math.inf):
                temp3 = mpmath.fmul(temp2,sqrt_pi_div_2*z1*erftemp)
        except:
            temp3 = mpmath.fmul(temp2,sqrt_pi_div_2*z1*erftemp)
        try:
            temp4 = 1 + (temp3/(delta*sqz2b1))
            if temp4 in (0,math.inf):
                temp4 = mpmath.fadd(1,mpmath.fdiv(temp3,(delta*sqz2b1)))
        except:
            temp4 = mpmath.fadd(1,mpmath.fdiv(temp3,(delta*sqz2b1)))
        p = mpmath.fmul(temp1,temp4)
        return p
    except:
        return 0
    
while True:
    z = random.uniform(1,1000)
    beta = random.uniform(0,100)
    delta = random.uniform(0,1000)
    # z=0.33864013266998344
    # beta=2.11550579204571
    # delta=0.0038221936742467806 
    z=0.23333333333333334
    beta=0.710353543760264
    delta=0.03114591894361798
    z=0.25862068965517243
    beta=0.886959224357771
    delta=0.025
    betasqroot = math.sqrt(beta)
    print(z,beta,delta)
    a = rprobdelta(z,beta,delta,betasqroot)
    # na = newrpobdeltadef(z,beta,delta,betasqroot)
    xna= xnewrpobdeltadef(z,beta,delta,betasqroot)
    print(z,beta,delta)
    print(a,xna,math.isclose(float(a),float(xna)))
    pass