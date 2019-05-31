'''
Serial implementation of the Inferior Olive network model used in:

THE GENERATION OF PHASE DIFFERENCES AND FREQUENCY CHANGES IN A NETWORK MODEL OF INFERIOR OLIVE SUBTHRESHOLD OSCILLATIONS

by Ben Torben-Nielsen, Idan Segev and Yosi Yarom

This network model is a slight modification of the original model as to make it run in serial mode, rather than in parallel as in the original implementation. The implemented network is the reference "4 clusters x 12 cells" network. This code will reproduce figure 1A, and Figure 2 A(right panel), B and C.

Implemented by Ben Torben-Nielsen, Hebrew University of Jerusalem, Israel
btorbennielsen@gmail.com
'''

import cPickle as pickle
import sys, time

import neuron
from neuron import h

from pylab import *
import numpy as np
import matplotlib.pyplot as plt

from cells import ManorCell,ManorCellHHS # load the cell templates

rnd_seed=21051982 # seed for the random generators.
mechanisms = []

def figure1_A() :
    plt.figure(1)
    x = loadtxt('gl_gcal.dat')
    cp = contourf(linspace(0,0.45,50),linspace(0,1.5,50),x,range(3,13))
    cb = plt.colorbar(cp)
    cb.set_label('Frequency (Hz)',fontsize=20)
    grid(1)
    np.random.seed(rnd_seed)
    clusters = []
    cluster1,specs1 =  makeCluster(12,gl_loc=0.18,gl_scale=0.005,gt_loc=0.4,gt_scale=0.1,test=1)
    cluster2,specs2 =  makeCluster(12,gl_loc=0.4,gl_scale=0.005,gt_loc=1.22,gt_scale=0.1,test=1)
    cluster3,specs3 =  makeCluster(12,gl_loc=0.35,gl_scale=0.005,gt_loc=0.7,gt_scale=0.1,test=1)
    cluster4,specs4 =  makeCluster(12,gl_loc=0.25,gl_scale=0.005,gt_loc=0.72,gt_scale=0.1,test=1)
    clusters.append((cluster1,specs1))
    clusters.append((cluster2,specs2))
    clusters.append((cluster3,specs3))
    clusters.append((cluster4,specs4))
    plt.xlabel('$g_l$ ($mS/cm^2$)',fontsize=20)
    plt.ylabel('$g_{Ca}$ ($mS/cm^2$)',fontsize=20)

def figure2_ABC() :
    print 'starting network simulation'
    clusters = run_network(t_stop=2000)
    
    peaks={}
    vms={}
    plt.figure(2,figsize=(6,10))

    # get the data and plot the raster
    plt.subplot(311)
    t = clusters[0][0][0].getRecordedVm()['t']
    y = 0
    for i in range(len(clusters)) :
        tCluster = clusters[i][0]
        for j in range(len(tCluster)) :
            tCell = tCluster[j]
            tVm = tCell.getRecordedVm()['vm']
            Ps,As = getPeaksFromNoiselessModel(t,tVm,minP=0.1)
            peaks[i,j] = (Ps,As)
            for peak in peaks[i,j] :
                plt.subplot(311)
                plt.plot([peak,peak],[y,y+1],'k')
                y += 1
            vms[i,j] = tVm
            if(i == 0) :
                plt.subplot(313)
                plt.plot(t,vms[0,j],'b')
        plt.subplot(312)
        plt.plot(t,vms[i,0])

    plt.subplot(311)
    plt.axis([1000,2000,0,49])
    plt.xlabel('time (ms)')
    plt.ylabel('cell number (12 per cluster)')
    plt.subplot(312)
    plt.axis([1000,2000,-61,-51])
    plt.xlabel('time (ms)')
    plt.ylabel('Vm (mV), one per cluster')
    plt.subplot(313)
    plt.axis([1000,2000,-61,-51])
    plt.xlabel('time (ms)')
    plt.ylabel('Vm (mV), one cluster')
    
def run_network(t_stop=3000,test=False) :
    if(test) :
        plt.figure(-1)
        x = loadtxt('gl_gcal.dat')
        contour(linspace(0,0.45,50),linspace(0,1.5,50),x,range(3,13))
        grid(1)
    np.random.seed(rnd_seed)
    clusters = []
    cluster1,specs1 =  makeCluster(12,gl_loc=0.18,gl_scale=0.005,gt_loc=0.4,gt_scale=0.1)
    cluster2,specs2 =  makeCluster(12,gl_loc=0.4,gl_scale=0.005,gt_loc=1.22,gt_scale=0.1)
    cluster3,specs3 =  makeCluster(12,gl_loc=0.35,gl_scale=0.005,gt_loc=0.7,gt_scale=0.1)
    cluster4,specs4 =  makeCluster(12,gl_loc=0.25,gl_scale=0.005,gt_loc=0.72,gt_scale=0.1)
    clusters.append((cluster1,specs1))
    clusters.append((cluster2,specs2))
    clusters.append((cluster3,specs3))
    clusters.append((cluster4,specs4))

    intra_cluster_connect_N(cluster1,coupling=(0.05,0.7),N=4)
    intra_cluster_connect_N(cluster2,coupling=(0.2,1.1),N=4)
    intra_cluster_connect_N(cluster3,coupling=(0.05,0.7),t1=1500,t2=3000,F=1,N=4)
    intra_cluster_connect_N(cluster4,coupling=(0.05,0.7),t1=1500,t2=3000,F=1,N=4)

    inter_cluster_connect_asym(cluster1,cluster3,coupling=(0.5,0.1),t1=10000,t2=10000,F1=1,F2=1)
    inter_cluster_connect(cluster2,cluster4,coupling=(0.4,1),t1=10000,t2=10000,F=1,P=1)
    inter_cluster_connect(cluster2,cluster3,coupling=(0.4,1),t1=10000,t2=10000,F=1,P=1)
    inter_cluster_connect(cluster3,cluster4,coupling=(0.5,1),t1=10000,t2=10000,F=1,P=1)
    inter_cluster_connect_asym(cluster1,cluster4,coupling=(0.5,0.1),t1=10000,t2=10000,F1=1,F2=1)

    t = simulate(t_stop=t_stop,dt=0.1)
    
    saveDataFN(clusters,t,base='network',peak=True,vm=True)

    return clusters

def makeHeteroCellManorDist(gl_loc=0.19,gl_scale=0.045,gt_loc=0.4,gt_scale=0.1,test=True) :
    g_l = np.random.normal(loc=gl_loc,scale=gl_scale)
    g_cal= np.random.normal(loc=gt_loc,scale=gt_scale)
    g_l = g_l if(g_l> 0.025) else 0.025 # otherwise the g_cal can blow up the Vm
    g_cal = g_cal if(g_cal >0) else 0
    if(test) :
        plt.plot(g_l,g_cal,'rs')
        plt.grid(1)
    return ManorCellHHS(g_l=g_l,g_cal=g_cal,g_kdr=0,g_na=0),(g_l,g_cal)

def makeCluster(N,gl_loc=0.19,gl_scale=0.045,gt_loc=0.4,gt_scale=0.1,test=False) :
    cluster = []
    specs = []
    np.random.seed(rnd_seed)
    for i in range(N) :
        r1,r2 = makeHeteroCellManorDist(gl_loc=gl_loc,gl_scale=gl_scale,gt_loc=gt_loc,gt_scale=gt_scale,test=test)
        cluster.append( r1 )
        specs.append( r2 )
    return cluster,specs

def intra_cluster_connect_N(cluster,coupling=(1,1),N=4,t1=100000,t2=100000,F=1.0) :
    '''
    Connecte the neurons inside a cluster
    '''
    global mechanisms,allCCs,interCCs,intraCCs
    np.random.seed(rnd_seed)
    for i in range(len(cluster)) :
        n = 0
        while(n < N) :
            j = np.random.random_integers(0,len(cluster)-1)
            if(i != j) :
                # print 'connecting: ', i, ' -> ', j
                g1,g2= _GJConnect(cluster[i],cluster[j],coupled=coupling,t1=t1,t2=t2,F=F)
                mechanisms.append(g1);mechanisms.append(g2)
                n +=1 

def inter_cluster_connect_asym(cluster1,cluster2,coupling=(1,1),t1=100000,t2=100000,F1=1,F2=1) :
    '''
    Connect the neurons belonging to different clusters. This method has an asymmetric coupling conductance. In nature asymmetry can result from differences in input resistances between cells.
    Asymmetry is described in Devor & Yarom, J Neurophysiol 87:3048-3058, 2002. 
    '''
    global mechanisms
    np.random.seed(rnd_seed)
    for i in range(len(cluster1)) :
        j = np.random.random_integers(0,len(cluster2)-1)
        #print 'inter_connect ', i, ' -> ', j
        g1,g2 = _GJConnectAsym(cluster1[i],cluster2[j],coupled=coupling,t1=t1,t2=t2,F1=F1,F2=F2)
        mechanisms.append(g1);mechanisms.append(g2)

def inter_cluster_connect(cluster1,cluster2,coupling=(1,1),t1=100000,t2=100000,F=1.0,P=1) :
    '''
    Connect neurons belonging to different cluster. This method inserts symmetric gap-junction conductances
    '''
    global mechanisms
    np.random.seed(rnd_seed)
    for i in range(len(cluster1)) :
        if(np.random.rand() <= P) :
            j = np.random.random_integers(0,len(cluster2)-1)
            #print 'inter_connect ', i, ' -> ', j
            g1,g2 = _GJConnect(cluster1[i],cluster2[j],coupled=coupling,t1=t1,t2=t2,F=F)
            mechanisms.append(g1);mechanisms.append(g2)

def _GJConnect(cell1,cell2,coupled=(0.5,0.5),t1=100000,t2=100000,F=1.0):
    gap1 = h.gap2(cell1.soma(0.5))
    gap1.t1=t1
    gap1.t2=t2
    gap2 = h.gap2(cell2.soma(0.5))
    gap2.t1=t1
    gap2.t2=t2
    h.setpointer(cell1.soma(0.5)._ref_v,'vgap',gap2)
    h.setpointer(cell2.soma(0.5)._ref_v,'vgap',gap1)
    c1 = np.random.uniform(coupled[0],coupled[1])
    c2 = c1
    gap1.g1=c1
    gap1.g2=c1*F
    gap2.g1=c2
    gap2.g2=c2*F
    return gap1,gap2

def _GJConnectAsym(cell1,cell2,coupled=(0.5,0.5),t1=100000,t2=100000,F1=1,F2=1):
    gap1 = h.gap2(cell1.soma(0.5))
    gap1.t1=t1
    gap1.t2=t2
    gap2 = h.gap2(cell2.soma(0.5))
    gap2.t1=t1
    gap2.t2=t2
    h.setpointer(cell1.soma(0.5)._ref_v,'vgap',gap2)
    h.setpointer(cell2.soma(0.5)._ref_v,'vgap',gap1)
    gap1.g1=coupled[0]
    gap1.g2=coupled[0]*F1
    gap2.g1=coupled[1]
    gap2.g2=coupled[1]*F2
    return gap1,gap2

def getPeaksFromNoiselessModel(t,vm,minP=0) :
    peaks = []
    amps = []
    damp = []
    for i in xrange(1,len(vm)-1) :
        if((vm[i] > vm[i-1]) and (vm[i] > vm[i+1])):
            #peak found
            amp =np.sqrt( (vm[i]-np.mean(vm))**2 )
            if(amp > minP) :
                peaks.append( t[i] )
                amps.append( amp )  
    return peaks,amps

def saveDataFN(clusters,t,base='xxx',peak=True,vm=False) :
    peaks={}
    vms={}
    for i in range(len(clusters)) :
        tCluster = clusters[i][0]
        for j in range(len(tCluster)) :
            tCell = tCluster[j]
            tVm = tCell.getRecordedVm()['vm']
            # tVm = retrieveVm(tCell)['vm']
            if(peak) :
                Ps,As = getPeaksFromNoiselessModel(t,tVm,minP=0.1)
                peaks[i,j] = (Ps,As)
            if(vm) : vms[i,j] = tVm
    if(peak) : pickle.dump( peaks, open(base+'_peaks_raw.pkl','w') )
    if(vm) : pickle.dump( vms, open(base+'_vms.pkl','w') )
    print 'saveDataFN finished writing'

def simulate(v_init=-55,dt=0.1,t_stop=3000):
    startT = time.time()
    h.finitialize(v_init)
    cvode = h.CVode()
    cvode.active(0)
    h.dt=0.1
    neuron.run(t_stop)
    print 'Simulation took: ', time.time()-startT, 's'
    return np.linspace(0,t_stop,t_stop/dt)

if __name__ == '__main__' :
    figure1_A()
    figure2_ABC()
    plt.show()
