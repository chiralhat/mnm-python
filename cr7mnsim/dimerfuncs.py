# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:54:24 2016

These functions set up the Hamiltonians for various ways of dealing with Cr7Mn
dimers, including the full Spin-1 treatment, the truncated Spin-1/2 treatment,
and the Spin-1/2 rotating frame treatment.

@author: ccollett
"""

import qutip as qt
import numpy as np
import scipy.constants as sc

ubG=sc.physical_constants['Bohr magneton in Hz/T'][0]/1e9/1e4

def spin_system(S):
    sx, sy, sz = [x/S for x in qt.jmat(S)]
    si = qt.qeye(2*S + 1)
    return (sx, sy, sz, si)

def two_spin_system(S):
    sx,sy,sz,si=spin_system(S)
    sx1,sy1,sz1=[qt.tensor(s,si) for s in [sx,sy,sz]]
    sx2,sy2,sz2=[qt.tensor(si,s) for s in [sx,sy,sz]]
    return sx1,sy1,sz1,sx2,sy2,sz2

def rotating_states(t, E, S=1/2):
    sx, sy, sz, si = spin_system(S)
    c = np.cos(E * t / 2)
    s = np.sin(E * t / 2)
    cs = 2 * c * s
    c2s2 = c**2 - s**2
    syp = cs*sz + c2s2*sy
    szp = c2s2*sz + cs*sy
    return [sx, syp, szp, si]

def spin_rotators(operators, theta=np.pi/4, S=1/2):
    sx, sy, sz, si = operators
    if S == 1/2:
        Rx = np.cos(theta)*si - 1j*np.sin(theta)*sx
        Ry = np.cos(theta)*si - 1j*np.sin(theta)*sy
        Rz = np.cos(theta)*si - 1j*np.sin(theta)*sz
    else:
        Rx = si + 2j*np.sin(theta)*np.cos(theta)*sx + 1/2*(2j*np.sin(theta)*sx)**2
        Ry = si + 2j*np.sin(theta)*np.cos(theta)*sx + 1/2*(2j*np.sin(theta)*sy)**2
        Rz = si + 2j*np.sin(theta)*np.cos(theta)*sx + 1/2*(2j*np.sin(theta)*sz)**2
    return Rx, Ry, Rz

def two_spin_rotators(E1, E2, theta=np.pi / 4, t=0, S=1/2):
    all_operators = [rotating_states(t, E, S) for E in [E1, E2]]
    Rs = [spin_rotators(operators, theta) for operators in all_operators]
    si = qt.qeye(2*S + 1)
    R1s = [qt.tensor(R,si) for R in Rs[0]]
    R2s = [qt.tensor(si,R) for R in Rs[1]]
    return R1s + R2s

def cr_h_shalf(E1,E2,J,Jp):
    sx1,sy1,sz1,sx2,sy2,sz2=two_spin_system(1/2)
    return E1*sx1 + E2*sx2 + J*sz1*sz2+Jp*(sx1*sx2+sy1*sy2)

def cr_h_s1(D1,D2,E1,E2,J,Jp):
    sx1,sy1,sz1,sx2,sy2,sz2=two_spin_system(1)
    return D1*sz1**2+D2*sz2**2 + E1*(sx1**2-sy1**2) + E2*(sx2**2-sy2**2)+J*sz1*sz2+Jp*(sx1*sx2+sy1*sy2)

def cr_h_rot(E1,E2,J,Jp):
    sx1,sy1,sz1,sx2,sy2,sz2=two_spin_system(1/2)
    U1=(1j*E1*sx1).expm()
    U2=(1j*E2*sx2).expm()
    y1,z1=[U1.dag()*o*U1 for o in [sy1,sz1]]
    y2,z2=[U2.dag()*o*U2 for o in [sy2,sz2]]
    return J*z1*z2+Jp*(sx1*sx2+y1*y2)

#Define single qubit Hamiltonian
def cr_ham_single(h,D=24.2,E=1.95,g=1.96,theta=0):
    hscale=ubG*g
    sx,sy,sz=qt.jmat(1)
    return -D*sz**2+E*(sx**2-sy**2)+h*hscale*(np.cos(theta*np.pi/180)*sz+np.sin(theta*np.pi/180)*sx)

def cr_ham_single_shalf(h,E=1.95,g=1.96,theta=0):
    hscale=ubG*g
    sx,sy,sz=qt.jmat(1/2)
    return E*(sx**2-sy**2)+h*hscale*(np.cos(theta*np.pi/180)*sz+np.sin(theta*np.pi/180)*sx)

#Quantum control functions: these go into the setup of the time-dep hamiltonian
def j_evolve(theta,Ham,J,tst=0,npts=500):
    tend=tst+theta/2/J
    def H1c(t,args):
        return 0
    H1=[Ham,H1c]
    return H1,tend

def e_evolve(fHam,E1,E2,tau,tst=0,npts=500):
    tend=tst+tau
    def H1c(t,args):
        if t>=tst and t<tend:
            return 1
        else:
            return 0
    H1=[fHam,H1c]
    return H1,tend

def r2_spin(axis,ops,theta,qubit,args,Ham,tst=0,nrot=13):
    sp=qubit-1
    sx1,sy1,sz1,sx2,sy2,sz2=ops
    w0=2*np.array(args['Es'])
    if axis=='z':
        phi=0
    else:
        phi=np.pi/2
    tend=tst+nrot*theta/args['w1']/2
    def H1coeff(t,args,fun,spt):
        if t>=tst and t<tend:
            return args['w1']*fun((w0[sp]-w0[spt])*t+phi)
        else:
            return 0
    def H1z_coeff(t,args):
        return H1coeff(t,args,np.cos,0)
    def H2z_coeff(t,args):
        return H1coeff(t,args,np.cos,1)
    def H1y_coeff(t,args):
        return H1coeff(t,args,np.sin,0)
    def H2y_coeff(t,args):
        return H1coeff(t,args,np.sin,1)
#    tlist=[tend]
    H1s=[[sz1,H1z_coeff],[sy1,H1y_coeff],[sz2,H2z_coeff],[sy2,H2y_coeff]]
    return H1s,tend

# This formalism comes from Vandersypen and Chuang, RMP 76, 1037
def r2_spin_rot(axis,ops,theta,qubit,args,Ham,tst=0,nrot=13):
    sp=qubit-1
    sx1,sy1,sz1,sx2,sy2,sz2=ops
    w0=2*np.array(args['Es'])
    if axis=='z':
        phi=0
    else:
        phi=np.pi/2
    tend=tst+nrot*theta/args['w1']/2
    def H1coeff(t,args,fun,spt):
        if t>=tst and t<tend:
            return args['w1']*fun((w0[sp]-w0[spt])*t+phi)
        else:
            return 0
    def H1z_coeff(t,args):
        return H1coeff(t,args,np.cos,0)
    def H2z_coeff(t,args):
        return H1coeff(t,args,np.cos,1)
    def H1y_coeff(t,args):
        return H1coeff(t,args,np.sin,0)
    def H2y_coeff(t,args):
        return H1coeff(t,args,np.sin,1)
#    tlist=[tend]
    H1s=[[sz1,H1z_coeff],[sy1,H1y_coeff],[sz2,H2z_coeff],[sy2,H2y_coeff]]
    return H1s,tend

def spin_echo(axis,spin,args,Ham,fullHam,E1,E2,tau,tst=0,nrot=13):
    H1,t1=r2_spin_rot('z',np.pi/2,1,args,Ham,tst=tst)
    H2,t2=e_evolve(fullHam,E1,E2,tau,tst=t1)
    H3,t3=r2_spin_rot('z',np.pi,1,args,Ham,tst=t2)
    H4,t4=e_evolve(fullHam,E1,E2,tau,tst=t3)
#    H1s=[*H1,H2,*H3,H4]
    H1s=H1+[H2]+H3+[H4]
    return H1s,t4
