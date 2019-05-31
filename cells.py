'''
Template for the model developed by Manor et al 1997.

This template is used in the network model by Torben-Nielsen, Segev and Yarom.

Implemented by Ben Torben-Nielsen, Hebrew University of Jerusalem, Israel
btorbennielsen@gmail.com
'''

import neuron
from neuron import h

import numpy as np

class ManorCell(object) :
    def __init__(self,g_l=0.15,g_cal=0.4) :
        soma = h.Section()
        soma.diam=25
        soma.L=25
        soma.nseg = 1
        soma.insert('leak')
        soma.insert('stoca')
        for seg in soma :
            seg.leak.gbar = g_l
            seg.leak.el=-63
            seg.stoca.gbar=g_cal
        self.soma = soma
        self.mechanisms = []

        self._initRecordingVm()
        self._initRecordingSpikes()

    def clearMechanisms(self) :
        self.mechanisms = []

    def insertHGap(self,**kwargs) :
        hgap = h.gap(self.soma(0.5))
        self.hgap = hgap

    def insertAlphaSynapse(self,**kwargs) :
        syn = h.AlphaSynapse(self.soma(0.5))
        syn.onset = kwargs['onset'] if('onset' in kwargs)  else 5
        syn.tau = kwargs['tau'] if('tau' in kwargs)  else 0.1
        syn.gmax = kwargs['gmax'] if('gmax' in kwargs)  else 5 # muS
        syn.e = kwargs['e'] if('e' in kwargs)  else 0 # muS
        self.mechanisms.append(syn)

    def insertExp2Synapse(self,**kwargs) :
        syn = h.Exp2Syn(self.soma(0.5))
        syn.tau1=kwargs['tau1'] if('tau1' in kwargs)  else 0.1
        syn.tau2= kwargs['tau2'] if('tau2' in kwargs)  else 0.1
        ns = h.NetStim()
        ns.start = kwargs['onset'] if('onset' in kwargs)  else 5
        ns.noise=0
        ns.number=1
        S= kwargs['gmax'] if('gmax' in kwargs)  else 0.15
        nc = h.NetCon(ns,syn,0,0,S)
        self.mechanisms.append(ns)
        self.mechanisms.append(nc)
        self.mechanisms.append(syn)

    def insertIClamp(self,**kwargs) :
        ic = h.IClamp(self.soma(0.5))
        ic.delay = kwargs['delay'] if('delay' in kwargs)  else 5
        ic.amp = kwargs['amp'] if('amp' in kwargs)  else 1
        ic.dur = kwargs['dur'] if('dur' in kwargs)  else 10
        self.mechanisms.append(ic)
        
    def getRecordedVm(self) :
	'''
	HOC-object cannot be pickled: convert to np.array
	'''
	ret = {}
	ret['t'] = np.array(self.VmRec['t'])
	ret['vm'] = np.array(self.VmRec['vm'])
	return ret
		
    def getRecordedSpikes(self) :
	return np.array(self.SpikesRec)

    def getRecordedCurrents(self) :
        ret = {}
        ret['leak'] = np.array(self.currents['leak'])
        ret['kdr'] = np.array(self.currents['kdr'])
        ret['na'] = np.array(self.currents['na'])
        return ret

    def _initRecordingVm(self) :
	self.VmRec= {}
	for var in 't', 'vm' :
            self.VmRec[var] = h.Vector()
	self.VmRec['vm'].record(self.soma(0.5)._ref_v)
	self.VmRec['t'].record(h._ref_t)
		
    def _initRecordingSpikes(self):
        self.SpikesRec = h.Vector()
        self.nc = h.NetCon(self.soma(0.5)._ref_v,None,sec=self.soma)
        self.nc.record(self.SpikesRec)

    def _initRecordingCurrents(self) :
        self.currents = {}
        for cur in ['leak','kdr','na'] :
            self.currents[cur] =h.Vector()
        self.currents['leak'].record(self.soma(0.5)._ref_i_leak)
        self.currents['kdr'].record(self.soma(0.5)._ref_i_iokdr)
        self.currents['na'].record(self.soma(0.5)._ref_i_iona)

    def simulate(self,v=-55,t=1000) :
        h.finitialize(v)
        neuron.run(t)

class ManorCellHHS(ManorCell) :
    def __init__(self,el=-63,g_l=0.15,g_cal=0.4,g_kdr=0,g_na=0) :
        #print 'ManorCellHHS'
        soma = h.Section()
        soma.L=25
        soma.diam=25
        soma.nseg = 5
        self.g_l=g_l
        self.g_cal=g_cal
        soma.insert('leak')
        soma.insert('stoca')
        soma.insert('iona')
        soma.insert('iokdr')
        for seg in soma :
            seg.leak.el=el
            seg.leak.gbar = g_l
            seg.stoca.gbar=g_cal
            seg.iokdr.gbar =g_kdr
            seg.iona.gbar=g_na
        self.soma = soma    
        self.mechanisms = []
        self._initRecordingVm()
        self._initRecordingSpikes()
        self._initRecordingCurrents()

