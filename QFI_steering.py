#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:40:09 2021

@author: samue
TODO: exact expressions
"""
  
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator)
import random
from sympy import LeviCivita

##Pauli matrices
s0 = qt.qeye(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

s=[s0,sx,sy,sz]

##main program runs simulations with fixed parameters and plots results
def main():
    ##Parameters
    Nqb = 2                                             ##Number of qubits
    DeltaT = 0.2                                        ##time step length
    #J = [1, 0.99, 1.01, 1.005, 0.995, 1.003, 0.997, 1.007]        ##coupling strength
    #J = [1-rng.random()/10 for j in range(Nqb)]
    J = [1]*Nqb                                         ##coupling strength
    if DeltaT == 0.2:
        N = 500
    elif DeltaT == 0.1:
        N=800                                           ##nr of time steps
    M = 100                                             ##number of trajectories
    Nst = 1                                             ##every Nst´th step is saved
    
    ##couplings
    K = 6                                               ##nr of couplings: 3, 4, 6, 8, 9, 10, 11, 12, 14
    params = [N, Nst, DeltaT, J[:Nqb], K]
    
    ##Observable for O_i=n*(X,Y,Z)
    #n = np.array([1,1,1])
    #print('n={0}'.format(n))
    #O = sum([n[0]*plaqS([int(4**(j-i)%4) for i in range(Nqb)])+n[1]*plaqS([int(2*4**(j-i)%4) for i in range(Nqb)])+n[2]*plaqS([int(3*4**(j-i)%4) for i in range(Nqb)]) for j in range(Nqb)])/2/np.linalg.norm(n)
    #O = sum([plaqS([int(3*4**(i-j)%4) for j in range(Nqb)]) for i in range(Nqb)])/2                                                            #O_i=Z
    O = sum([plaqS([int(4**(j-i)%4) for i in range(Nqb)])+plaqS([int(3*4**(j-i)%4) for i in range(Nqb)]) for j in range(Nqb)])/2/np.sqrt(2)     #O_i=X+Z
 
    ##Initial state
    psi0 = qt.qstate(Nqb*'d')               ##starting in 00...0 state
    psi0 = psi0.unit()
    #psi0 = qt.rand_ket(2**Nqb, dims=[[2]*Nqb,[1]*Nqb])             ##random state
    
    print('psi0={0}'.format(psi0))
    print('O={0}'.format(O))
    print(r'[N, Nst, DeltaT, J, K]={0}, M={1}'.format(params, M))
    
    ##parallel running M trajectories
    results = qt.parallel_map(trajec, [Nqb]*M, task_kwargs=dict(psi0=psi0, O=O, param=params), progress_bar=True)#, num_cpus=2)
    result_array = np.array(results, dtype=object)
    if Nqb == 2:
        k_List, xi_eta_List, psi_List, QFI, phList, S_List = result_array.T
    elif Nqb < 5:
        k_List, xi_eta_List, psi_List, QFI, phList = result_array.T
    else:
        k_List, xi_eta_List, QFI, phList = result_array.T    
    
    ##rearrange Distributions by Metropolis temperatures
    QFIDistr = np.reshape(np.concatenate(QFI), (M, int(N/Nst)+1))
    phDistr = np.reshape(np.concatenate(phList), (M, int(N/Nst)+1))
    if Nqb == 2:
        SDistr = np.reshape(np.concatenate(S_List), (M, int(N/Nst)+1))
    if Nqb < 5:
        psiDistr = np.reshape(np.concatenate(psi_List), (M, int(N/Nst)+1))
        np.savetxt('psi_List', np.concatenate(psi_List), fmt='%s')
    
    ##save data to files
    np.savetxt('coupl_list', np.concatenate(k_List))
    np.savetxt('xi_eta_List', np.concatenate(xi_eta_List), fmt='%s')
    
############################################################Plots       
    ##phase ensemble at final time
    fig, ax = plt.subplots(figsize=(8,6))
    ph = phDistr[:,-1]
    np.savetxt('phHist', ph)
    plt.hist(ph, bins=np.arange(-np.pi,np.pi+0.01,2*np.pi/50), alpha=0.5, edgecolor='grey')
    plt.xlim(left=-np.pi, right=np.pi)
    
    plt.minorticks_on()
    plt.xlabel(r'$\phi$', fontsize=35)
    plt.ylabel(r'Number of final states', fontsize=35)
    xticks = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    xlabels = [r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.text(-3.9,1.1*ax.get_ylim()[1], r'(a)', fontsize=35)
    
    plt.tight_layout()    
    plt.savefig('rel ph hist.pdf')
    plt.savefig('rel ph hist.svg')
    
    ##phase single traj
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.plot(np.arange(0,N+1,Nst), np.mean(phDistr, axis=0), label=r'average phase', linewidth=3)
    ax.plot(np.arange(0,N+1,Nst), phDistr[0], '--', label=r'single phase 1', linewidth=2, alpha=0.7)
    
    np.savetxt('phMean', np.mean(phDistr, axis=0))
    np.savetxt('phTraj1', phDistr[0])
    
    plt.ylim(bottom=-3.2, top=3.2)
    plt.xlim(left=200, right=N)
    plt.minorticks_on()
    plt.xlabel(r'$n_t$', fontsize=35)
    plt.ylabel(r'$\phi$', fontsize=35)
    yticks = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ylabels = [r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$']
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    ax.text(-(N/500)*75+200,2.8, r'(b)', fontsize=35)
    
    plt.tight_layout()    
    plt.savefig('rel ph traj.pdf')
    plt.savefig('rel ph traj.svg')
    
    ##final qfi histogram
    fig, ax = plt.subplots(figsize=(8,6))
    plt.hist(QFIDistr[:,-1], bins=np.arange(0,Nqb**2+Nqb**2/100,Nqb**2/100), alpha=0.5, label=r'$[p_r]$', edgecolor='grey')
    
    np.savetxt('qfiHist', QFIDistr[:,-1])
    
    plt.xlim(left=0, right=Nqb**2)
    plt.minorticks_on()
    
    plt.xlabel(r'$QFI final$', fontsize=25)
    plt.ylabel(r'Number of trajectories', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()    
    plt.savefig('qfi hist.pdf')
    plt.savefig('qfi hist.svg')
    
    ##coupling histogram
    fig, ax = plt.subplots(figsize=(8,6))
    plt.hist(np.concatenate(k_List), bins=np.arange(0,K**2), alpha=0.5, label=r'$[p_r]$', edgecolor='grey')
    
    plt.xlim(left=0, right=K**2)
    plt.minorticks_on()
    plt.xlabel(r'$K$', fontsize=25)
    plt.ylabel(r'Number of trajectories', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()    
    plt.savefig('coupl hist.pdf')
    plt.savefig('coupl hist.svg')
    
    ##purity of average
    if Nqb < 5 and type(psiDistr[0,0]) == qt.qobj.Qobj:
        fig, ax = plt.subplots(figsize=(8,6))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.ylim(bottom=0, top=1.01)
        plt.xlim(left=0, right=N)
        avgPur = [(sum(psiDistr[:,i])**2).tr()/M**2 for i in range(int(N/Nst)+1)]
        np.savetxt('avgPurList', avgPur)
        
        ax.plot(np.arange(0,N+1,Nst), avgPur, label=r'average purity', linewidth=3)
        ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*(1/2**Nqb), 'k:', label=r'minimum', linewidth=2)
        ax.set_xlabel(r'$n_t$',fontsize=35)
        ax.set_ylabel(r'$P(t)$',fontsize=35)
        ax.minorticks_on()
        ax.tick_params(length=8, right=True)
        ax.tick_params(which='minor', length=4, right=True) 
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=6)
        ax.xaxis.set_minor_locator(MultipleLocator(N/20))
        
        ax.text(-(N/500)*95,0.95, r'(c)', fontsize=35)
        
        plt.tight_layout()
        plt.savefig('avg pur.pdf', format='pdf')
        plt.savefig('avg pur.svg', format='svg')
        
        ##average state
        rhoFin = sum(psiDistr[:,-1])/M
        fig, ax = qt.hinton(rhoFin)
        plt.show()
        
        plt.savefig('avg fin state.pdf', format='pdf')
        plt.savefig('avg fin state.svg', format='svg')
    
    ##QFI plot
    fig, ax = plt.subplots(figsize=(8,6))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylim(bottom=0, top=Nqb**2+0.01)
    plt.xlim(left=0, right=N)
    avgQFI = np.mean(QFIDistr, axis=0)
    np.savetxt('avgQFIList', avgQFI)
    np.savetxt('QFItraj1', QFIDistr[0])
    np.savetxt('QFItraj2', QFIDistr[1])
    np.savetxt('QFItraj3', QFIDistr[2])
    
    print('final qfi500={0}'.format(avgQFI[int(500/Nst)]))
    if N>=800:
        print('final qfi800={0}'.format(avgQFI[int(800/Nst)]))
    print('final qfi{0}={1}'.format(N, avgQFI[int(N/Nst)]))
    
    ax.plot(np.arange(0,N+1,Nst), avgQFI, label=r'$\overline{{F_{{Q}}}}$', linewidth=3)
    ax.plot(np.arange(0,N+1,Nst), QFIDistr[0], '--', label=r'Single trajectory', linewidth=2, alpha=0.7)
    ax.plot(np.arange(0,N+1,Nst), QFIDistr[1], '--', label=r'Single trajectory', linewidth=2, alpha=0.7)
    ax.plot(np.arange(0,N+1,Nst), QFIDistr[2], '--', label=r'Single trajectory', linewidth=2, alpha=0.7)
    ax.set_xlabel(r'$n_t$',fontsize=35)
    ax.set_ylabel(r'$F_Q$',fontsize=35)
    ax.minorticks_on()
    ax.tick_params(length=8)
    ax.tick_params(which='minor', length=4)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=6)
    ax.xaxis.set_minor_locator(MultipleLocator(N/20))
    plt.legend(fontsize=30)
    
    ax.text(-(N/500)*85,0.95*Nqb**2, r'(a)', fontsize=35)
    
    plt.tight_layout()
    
    plt.savefig('qfi.pdf')
    
    print('step90', np.where(np.array(avgQFI)>=0.9*Nqb**2)[0])
    print('step95', np.where(np.array(avgQFI)>=0.95*Nqb**2)[0])

    ##entanglement plots
    if Nqb==2:
        ##Average entanglement entropy plot    
        fig, ax = plt.subplots(figsize=(8,6))
        avgS = np.mean(SDistr, axis=0)
        np.savetxt('avgSList', avgS)
        np.savetxt('Straj', SDistr[0])
    
        ax.plot(np.arange(0,N+1,Nst), avgS, linewidth=3, label=r'average')
        ax.plot(np.arange(0,N+1,Nst), SDistr[0], '--', label=r'single traj', linewidth=2, alpha=0.7)
        ax.plot(np.arange(0,N+1,Nst), np.ones(int(N/Nst)+1)*np.log(2), 'k:', label=r'maximally entangled')
        ax.set_xlabel(r'$n_t$',fontsize=25)
        ax.set_ylabel('$S(t)$',fontsize=25)
        plt.xlim(0,N)
        plt.ylim(0,0.73)
        plt.minorticks_on()
        ax.tick_params(length=8)
        ax.tick_params(which='minor', length=4)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        ax.xaxis.set_minor_locator(MultipleLocator(N/20))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        
        plt.savefig('avg EE.pdf')
        plt.savefig('avg EE.svg')
       
####################################################################################        
##Trajectory simulator
def trajec(Nqb, psi0, O, param):
    N, Nst, DeltaT, J, K = param            ##unpack parameters
    Gamma = [j**2*DeltaT for j in J]        ##jump rate, default: J*J*DeltaT
    
    ##2 qubit correlator Q indices
    indList2 = [0]
    indList2.extend([a*b for b in [4**i for i in range(Nqb)] for a in [1,2,3]])
    indList2.extend([alpha*4**j+beta*4**k for alpha in range(1,4) for beta in range(1,4) for j in range(Nqb-1) for k in range(j+1,Nqb)])
      
    #random number generator
    rng = np.random.default_rng()
    
    ##phase states
    st1 = qt.basis(2,0)-(1-np.sqrt(2))*qt.basis(2,1)
    st2 = (1-np.sqrt(2))*qt.basis(2,0)+qt.basis(2,1)
    state1 = (qt.tensor([st1]*Nqb)).unit()
    state2 = (qt.tensor([st2]*Nqb)).unit()
    
    ##Measured quantities
    k_List = np.zeros(N*int(Nqb/2))
    qfi = np.zeros(int(N/Nst)+1)
    phList = np.zeros(int(N/Nst)+1)
    xi_eta_List = np.zeros(N, dtype=tuple)
    if Nqb < 5:
        psi_List = np.zeros(int(N/Nst+1), dtype=object)
    
    ##entanglement entropy for 2 qubits:
    if Nqb == 2:
        S_list = np.zeros(int(N/Nst)+1)
        S_list[0] = qt.entropy_vn(psi0.ptrace((0)))
        
    ##coupling list
    if K == 3:          ##beta=x; alpha=x,y,z; s=+
        slist = [1,1,1]
        aList = [1,2,3]
        bList = [1,1,1]
    elif K == 4:          ##beta=x; alpha=0,x,y,z; s=+
        slist = [1,1,1,1]
        aList = [0,1,2,3]
        bList = [1,1,1,1]
    elif K == 6:          ##beta=x,z; alpha=x,y,z; s=+
        slist = [1,1,1,1,1,1]
        aList = [1,2,3,1,2,3]
        bList = [1,1,1,3,3,3]
        
        ##beta=x,y; alpha=x,y,z; s=+
        #slist = [1,1,1,1,1,1]
        #aList = [1,2,3,1,2,3]
        #bList = [1,1,1,2,2,2]
    elif K == 8:        ##beta=x,y; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1]
        aList = [0,1,2,3,0,1,2,3]
        bList = [1,1,1,1,2,2,2,2]
    elif K == 9:        ##beta=x,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1]
        
        ##beta=x,y,z; alpha=x,y,z; s=+
        #slist = [1,1,1,1,1,1,1,1,1]
        #aList = [1,2,3,1,2,3,1,2,3]
        #bList = [1,1,1,2,2,2,3,3,3]
    elif K == 10:        ##beta=x,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1]
    elif K == 11:        ##beta=x,y,z; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,1,1,1,1,2,2,2,2]
    elif K == 12:        ##beta=x,y,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,2,2,2]
    elif K == 14:        ##beta=x,y,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1,2,2,2,2]    
    
    ##operator list
    pauliPlaqList = np.zeros(int(1+3*Nqb+9/2*Nqb*(Nqb-1)), dtype=object)
    
    Spsi = np.zeros(int(1+3*Nqb+9/2*Nqb*(Nqb-1)))
    SO = np.zeros(int(1+3*Nqb+9/2*Nqb*(Nqb-1)))
    
    #compute initial values
    for l,k in enumerate(indList2):
        pauliPlaqList[l] = plaqS([int(k/(4**j)%4) for j in range(Nqb)])
        Spsi[l] = qt.expect(pauliPlaqList[l],psi0)
        SO[l] = qt.expect(pauliPlaqList[l],O)
      #  Spsi[l] = qt.expect(plaqS([int(k/(4**j)%4) for j in range(Nqb)]),psi0)     #for N>15 to save memory
      #  SO[l] = qt.expect(plaqS([int(k/(4**j)%4) for j in range(Nqb)]),O)          #for N>15 to save memory
    
    ##Initial QFI
    phList[0] = np.angle((state1).overlap(psi0)*psi0.overlap(state2))
    qfi[0] = (SO[0:1+3*Nqb]@SO[0:1+3*Nqb]+2*sum([np.sum(np.kron(SO[1+3*k:1+3*(k+1)],SO[1+3*j:1+3*(j+1)])*Spsi[int(1+3*Nqb+(2*Nqb-j-1)*j/2)+k-j-1::int(Nqb*(Nqb-1)/2)]) for j in range(Nqb-1) for k in range(j+1,Nqb)])-(SO[0:1+3*Nqb]@Spsi[0:1+3*Nqb])**2)/2**(2*(Nqb-1))       ##starting point for 2 corr: (Nqb-1)+(Nqb-2)+...+(Nqb-j)
    
    ##Density matrix tracking
    if Nqb < 5:
        psi_List[0] = qt.ket2dm(psi0)
    
    #qbList = np.arange(0,Nqb)                  ##needed for fully coupled chain
    
    ##Time step loop
    psi = psi0
    for i in range(1, N+1):
        nStart1 = int(N*rng.random())
        nStart2 = (nStart1+1)%Nqb
       
        #random.shuffle(qbList)                 ##needed for fully coupled chain
        
        ##steering coupled qubit pairs
        for nPair in range(int(Nqb/2)):
            ##Nearest neighbor coupling
            n1 = (nStart1+2*nPair)%Nqb
            n2 = (nStart2+2*nPair)%Nqb
            
            ##for fully coupled chain
           # n1 = int(qbList[2*nPair])
           # n2 = int(qbList[2*nPair+1])
            
            ##closed boundary
           # if n1>n2:
           #     #break
            
            J1, J2 = J[n1], J[n2]
            G1, G2 = Gamma[n1], Gamma[n2]
            
            ##decision-making
            deltaQFI = expQFIchgSparse(Spsi, SO, J, Gamma, DeltaT, n1, n2, Nqb, K)
                
            klis = rng.choice(np.where(deltaQFI==np.nanmax(deltaQFI))[0])
            k_List[(i-1)*int(Nqb/2)+nPair] = klis
        
            ##chosen couplings
            s1 = slist[int(klis%K)]
            s2 = slist[int(klis/K)]
            alpha1 = aList[int(klis%K)]
            alpha2 = aList[int(klis/K)]
            beta1 = bList[int(klis%K)]
            beta2 = bList[int(klis/K)]
            
            ##chosen Pauli operators
            sig1 = pauliPlaqList[alpha1+3*n1]
            sig2 = pauliPlaqList[alpha2+3*n2]
            
           # sig1 = plaqS([int(alpha1*4**(n1-j)%4) for j in range(Nqb)])     #for N>15 to save memory
           # sig2 = plaqS([int(alpha2*4**(n2-j)%4) for j in range(Nqb)])     #for N>15 to save memory
            
            ##Time step
            #beta1=beta2=z
            if beta1==3 and beta2==3:
                xi_eta_List[i-1] = (0,(-1)**int(2*rng.random()))
                H = s1*J1*sig1+s2*J2*sig2
                psi = schroesol(psi, DeltaT, H)
            
            #beta1=z, beta2!=z
            elif beta1==3:
                c_op, H = np.sqrt(G2)*sig2, s1*J1*sig1
                psi, xi = unitsol(psi, DeltaT, H, c_op, G2*DeltaT)
                xi_eta_List[i-1] = (xi,(-1)**int(2*rng.random()))
                
            #beta1!=z, beta2=z
            elif beta2==3:
                c_op, H = np.sqrt(G1)*sig1, s2*J2*sig2
                psi, xi = unitsol(psi, DeltaT, H, c_op, G1*DeltaT)
                xi_eta_List[i-1] = (xi,(-1)**int(2*rng.random()))
            
            #beta1=beta2=x/y  
            elif beta1==beta2:
                c_op = [np.sqrt(G1/2)*sig1+np.sqrt(G2/2)*sig2, np.sqrt(G1/2)*sig1-np.sqrt(G2/2)*sig2]
                psi, xi_eta_List[i-1] = ent_swap_sol(psi, DeltaT, c_op)
            
            #(beta1=x, beta2=y) or (beta1=y, beta2=x)
            else:
                eta = (-1)**int(2*rng.random())
                if beta1==1:
                    c_op, H = np.sqrt(G1)*sig1+eta*1j*np.sqrt(G2)*sig2, eta*np.sqrt(G1*G2)*sig1*sig2
                elif beta1==2:
                    c_op, H = np.sqrt(G1)*sig1-eta*1j*np.sqrt(G2)*sig2, eta*np.sqrt(G1*G2)*sig1*sig2
                psi, xi = unitsol(psi, DeltaT, H, c_op, (G1+G2)*DeltaT)
                xi_eta_List[i-1] = (xi, eta)
                
        ##Update values
        for l, k in enumerate(indList2):
            Spsi[l] = qt.expect(pauliPlaqList[l],psi)
            #Spsi[l] = qt.expect(plaqS([int(k/(4**j)%4) for j in range(Nqb)]),psi)     #for N>15 to save memory
        
        if i%Nst == 0:
            if Nqb < 5:
                psi_List[int(i/Nst)] = qt.ket2dm(psi)
            phList[int(i/Nst)] = np.angle((state1).overlap(psi)*psi.overlap(state2))
            
            qfi[int(i/Nst)] = (SO[0:1+3*Nqb]@SO[0:1+3*Nqb]+2*sum([np.sum(np.kron(SO[1+3*k:1+3*(k+1)],SO[1+3*j:1+3*(j+1)])*Spsi[int(1+3*Nqb+(2*Nqb-j-1)*j/2)+k-j-1::int(Nqb*(Nqb-1)/2)]) for j in range(Nqb-1) for k in range(j+1,Nqb)])-(SO[0:1+3*Nqb]@Spsi[0:1+3*Nqb])**2)/2**(2*(Nqb-1))       ##starting point for 2 corr: (Nqb-1)+(Nqb-2)+...+(Nqb-j)
            
            ##entanglement entropy for 2 qubits:
            if Nqb == 2:
                S_list[int(i/Nst)] = qt.entropy_vn(psi.ptrace((0)))
    
    if Nqb == 2:
        return k_List, xi_eta_List, psi_List, qfi, phList, S_list
    elif Nqb < 5:
        return k_List, xi_eta_List, psi_List, qfi, phList
    else:
        return k_List, xi_eta_List, qfi, phList

##Time evolution solver    
##Entanglement swapping for beta1=beta2=x/y
def ent_swap_sol(psi, deltaT, c_op):
    '''return normalized state conditioned on measurement outcome
    Bell state measurement
    '''
    #random number generator
    rng = np.random.default_rng() 
    #probabilities
    P = np.zeros(4)
    P[0:2] = qt.expect([c.dag()*c for c in c_op], psi)*deltaT
    P[2:4] = (1-2*P[0:2])/2
    #stochastic measurement outcome
    while 1==1:
        k = int(4*rng.random())
        if rng.random() < P[k]:
            break
    ##time evolution
    if k < 2:
        return (c_op[k]*psi).unit(), (1,(-1)**k)
    else:
        return ((1-deltaT*c_op[k-2].dag()*c_op[k-2])*psi).unit(), (0,(-1)**k)
        
##Unitary dynamics for (beta1,beta2)=(x,y) or (x/y,z) and vice-versa
def unitsol(psi, deltaT, H, c_op, P):
    '''return normalized state conditioned on measurement outcome
    for single, unitary jump operator
    '''
    #random number generator
    rng = np.random.default_rng()
    if rng.random() >= P:
        return ((1-1j*deltaT*H-P/2)*psi).unit(), 0
    else:
        return (c_op*psi).unit(), 1

def schroesol(psi, deltaT, H):
    '''Single step Hamiltonian evolution
    Parameters
    ----------

    H: :class:`qutip.Qobj`
        Hamiltonian

    psi: :class:`qutip.Qobj`
        initial state vector (ket)
        
    deltaT: (float)
        time step length
        
    Returns
    -------

    result: :class:`qutip.Qobj`

        Normalized time-evolved state (ket)
    '''
    
    return ((1-1j*deltaT*H)*psi).unit()
    
###plaquette operators
def plaqS(ind):
    pauliLs = [s[i] for i in ind]
    return qt.tensor(pauliLs)
    
def plaqSlist(Nqb):
    return  [plaqS([int(i/(4**j)%4) for j in range(Nqb)]) for i in range(4**Nqb)]

##sparse implementation
def expQFIchgSparse(S, SO, J, Gamma, deltaT, nA, nB, Nqb, K):
    JA = J[nA]
    JB = J[nB]
    GA = Gamma[nA]
    GB = Gamma[nB]
    
    ##coupling list
    if K == 3:          ##beta=x; alpha=x,y,z; s=+
        slist = [1,1,1]
        aList = [1,2,3]
        bList = [1,1,1]
    elif K == 4:        ##beta=x; alpha=0,x,y,z; s=+
        slist = [1,1,1,1]
        aList = [0,1,2,3]
        bList = [1,1,1,1]
    elif K == 6:        ##beta=x,z; alpha=x,y,z; s=+
        slist = [1,1,1,1,1,1]
        aList = [1,2,3,1,2,3]
        bList = [1,1,1,3,3,3]
        
        ##beta=x,y; alpha=x,y,z; s=+
        #slist = [1,1,1,1,1,1]
        #aList = [1,2,3,1,2,3]
        #bList = [1,1,1,2,2,2]
    elif K == 8:        ##beta=x,y; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1]
        aList = [0,1,2,3,0,1,2,3]
        bList = [1,1,1,1,2,2,2,2]
    elif K == 9:        ##beta=x,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1]
        
        ##beta=x,y,z; alpha=x,y,z; s=+
        #slist = [1,1,1,1,1,1,1,1,1]
        #aList = [1,2,3,1,2,3,1,2,3]
        #bList = [1,1,1,2,2,2,3,3,3]
    elif K == 11:        ##beta=x,y,z; alpha=0,x,y,z; s=+
        slist = [1,1,1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,1,1,1,1,2,2,2,2]
    elif K == 10:        ##beta=x,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1]
    elif K == 12:        ##beta=x,y,z; alpha=x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,1,2,3,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,2,2,2]
    elif K == 14:        ##beta=x,y,z; alpha=0,x,y,z; s=+/-
        slist = [1,1,1,-1,-1,-1,1,1,1,1,1,1,1,1]
        aList = [1,2,3,1,2,3,0,1,2,3,0,1,2,3]
        bList = [3,3,3,3,3,3,1,1,1,1,2,2,2,2]
        
    ##indexlist
    indList2 = [0]
    indList2.extend([a*b for b in [4**i for i in range(Nqb)] for a in [1,2,3]])
    indList2.extend([alpha*4**j+beta*4**k for alpha in range(1,4) for beta in range(1,4) for j in range(Nqb-1) for k in range(j+1,Nqb)]) 
   
    ##expected qfi change
    dqfi=np.zeros(K**2)
    
    for j in range(K**2):        
        sA = slist[j%K]
        sB = slist[int(j/K)]
        aA = aList[j%K]
        aB = aList[int(j/K)]
        bA = bList[j%K]
        bB = bList[int(j/K)]
        
        ##correlator
        Q = Ssparse(S,aA*4**nA+aB*4**nB, indList2)
        
        ##F tensor
        F = np.zeros(1+3*Nqb)
        
        if aA==aB and aA==0:
            F = S[:1+3*Nqb]
        else:
            for l, k in enumerate(indList2[:1+3*Nqb]):
                muA = int(k/(4**nA)%4)
                muB = int(k/(4**nB)%4)
                if muA == 0 and muB == 0:
                    F[l] = Ssparse(S,k+aA*4**nA+aB*4**nB, indList2)
                elif muA == 0 and muB == aB:
                    F[l] = Ssparse(S,k+aA*4**nA-aB*4**nB, indList2)
                elif muA == aA and muB == 0:
                    F[l] = Ssparse(S,k-aA*4**nA+aB*4**nB, indList2)
                elif muA == aA and muB == aB:
                    F[l] = Ssparse(S,k-aA*4**nA-aB*4**nB, indList2)
                
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA!=0 and aB!=0:
                    rtm1 = 0
                    for k1 in range(1,4):
                        if k1 != aA and k1 != muA:
                            for k2 in range(1,4):
                                if k2 != aB and k2 != muB:
                                    rtm1 += LeviCivita(aA,muA,k1)*LeviCivita(aB,muB,k2)*Ssparse(S,k+(k1-muA)*4**nA+(k2-muB)*4**nB, indList2)
                    F[l] = rtm1
                    
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA==0 and aB!=0:
                    rtm1 = 0
                    for k2 in range(1,4):
                        if k2 != aB and k2 != muB:
                            rtm1 += LeviCivita(aB,muB,k2)*Ssparse(S,k+(k2-muB)*4**nB, indList2)
                    F[l] = rtm1
                elif muA != 0 and muA != aA and muB != 0 and muB != aB and aA!=0 and aB==0:
                    rtm1 = 0
                    for k1 in range(1,4):
                        if k1 != aA and k1 != muA:
                            rtm1 += LeviCivita(aA,muA,k1)*Ssparse(S,k+(k1-muA)*4**nA, indList2)
                    F[l] = rtm1
            
        ##<c_eta^\dagger c_eta>
        avcp = (bA != 3)*GA+(bB != 3)*GB
        rtm1 = (bA == bB)*(bA != 3)*2*np.sqrt(GA*GB)*Q
        avcm = avcp-rtm1
        avcp += rtm1
        
        dR = np.zeros(int(1+3*Nqb+9/2*Nqb*(Nqb-1)))
        Gplus, Gmin = np.zeros(1+3*Nqb), np.zeros(1+3*Nqb)
        
        ##calculate dR, Gplus/min
        for l, k in enumerate(indList2):
            ##<<dR>> terms
            muA = int(k/(4**nA)%4)
            muB = int(k/(4**nB)%4)
            rtm1 = 0
            rtm2 = 0
            ##A terms
            if muA != aA and muA != 0 and aA!=0:
                if bA !=3:
                    rtm1 -= GA*S[l]
                    rtm2 -= GA*S[l]
                else:
                    rtm4 = 0
                    for k1 in range(1,4):
                        if k1 != muA and k1 != aA:
                            rtm4 += int(LeviCivita(aA,k1,muA))*Ssparse(S,k+(k1-muA)*4**nA,indList2)
                    rtm1 += sA*JA*rtm4
            
            ##B terms
            if muB != aB and muB !=0 and aB!=0:
                if bB !=3:
                    rtm1 -= GB*S[l]
                    rtm2 -= GB*S[l]
                else:
                    rtm4 = 0
                    for k2 in range(1,4):
                        if k2 != muB and k2 != aB:
                            rtm4 += int(LeviCivita(aB,k2,muB))*Ssparse(S,k+(k2-muB)*4**nB,indList2)
                    rtm1 += sB*JB*rtm4
                    
            dR[l] = 2*deltaT*rtm1
        
            ##G terms
            if l < 3*Nqb+1 and rtm2 !=0:
                rtm3 = (bA == bB)*np.sqrt(GA*GB)*(F[l]-Q*S[l])
                Gplus[l] = rtm2+rtm3
                Gmin[l] = rtm2-rtm3
        
        ##assemble change to qfi
        if avcp == 0 and avcm == 0:
            dqfi[j] = (2*sum([np.sum(np.kron(SO[1+3*k:1+3*(k+1)],SO[1+3*i:1+3*(i+1)])*dR[int(1+3*Nqb+(2*Nqb-i-1)*i/2)+k-i-1::int(Nqb*(Nqb-1)/2)]) for i in range(Nqb-1) for k in range(i+1,Nqb)])-2*(SO[0:1+3*Nqb]@dR[0:1+3*Nqb])*(SO[0:1+3*Nqb]@S[0:1+3*Nqb]))/2**(2*(Nqb-1))
        elif avcp == 0:
            dqfi[j] = (2*sum([np.sum(np.kron(SO[1+3*k:1+3*(k+1)],SO[1+3*i:1+3*(i+1)])*dR[int(1+3*Nqb+(2*Nqb-i-1)*i/2)+k-i-1::int(Nqb*(Nqb-1)/2)]) for i in range(Nqb-1) for k in range(i+1,Nqb)])-2*(SO[0:1+3*Nqb]@dR[0:1+3*Nqb])*(SO[0:1+3*Nqb]@S[0:1+3*Nqb])-2*deltaT*(SO[0:1+3*Nqb]@Gmin)**2/avcm)/2**(2*(Nqb-1))
        elif avcm == 0:
            dqfi[j] = (2*sum([np.sum(np.kron(SO[1+3*k:1+3*(k+1)],SO[1+3*i:1+3*(i+1)])*dR[int(1+3*Nqb+(2*Nqb-i-1)*i/2)+k-i-1::int(Nqb*(Nqb-1)/2)]) for i in range(Nqb-1) for k in range(i+1,Nqb)])-2*(SO[0:1+3*Nqb]@dR[0:1+3*Nqb])*(SO[0:1+3*Nqb]@S[0:1+3*Nqb])-2*deltaT*(SO[0:1+3*Nqb]@Gplus)**2/avcp)/2**(2*(Nqb-1))
        else:
            dqfi[j] = (2*sum([np.sum(np.kron(SO[1+3*k:1+3*(k+1)],SO[1+3*i:1+3*(i+1)])*dR[int(1+3*Nqb+(2*Nqb-i-1)*i/2)+k-i-1::int(Nqb*(Nqb-1)/2)]) for i in range(Nqb-1) for k in range(i+1,Nqb)])-2*(SO[0:1+3*Nqb]@dR[0:1+3*Nqb])*(SO[0:1+3*Nqb]@S[0:1+3*Nqb])-2*deltaT*((SO[0:1+3*Nqb]@Gplus)**2/avcp+(SO[0:1+3*Nqb]@Gmin)**2/avcm))/2**(2*(Nqb-1))
                       
    return dqfi

##sparse implementation
def Ssparse(S,k,indList2):
    if k in indList2:
        return S[np.where(np.array(indList2)==k)[0][0]]
    else:
        return 0

##Multicore compatibility in windows
if __name__ == '__main__':
    main()