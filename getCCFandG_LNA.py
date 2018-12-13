import threading 
import numpy as np

class getCCFandG_LNA(threading.Thread):
    CCF = []
    G_LNA = []
    lock = threading.Lock()
    
    def __init__(self,CCF, G_lna):
        threading.Thread.__init__(self)
        self.CCFdata = CCF
        self.G_LNAdata = G_lna
        getCCFandG_LNA.lock.acquire()
        getCCFandG_LNA.CCF = []
        getCCFandG_LNA.lock.release()
        getCCFandG_LNA.lock.acquire()
        getCCFandG_LNA.G_LNA = []
        getCCFandG_LNA.lock.release()
        
        
    def get_CCF(self, Start_freq, Stop_freq, scaling_factor):
        # CCFtemp = 0
        #CCFtemp = [self.CCFdata[i,1] for i in enumerate(len(self.CCFdata[:,0])) if self.CCFdata[i,0] >= Start_freq and self.CCFdata[i,0] <= Stop_freq]
        range_fact = int(len(self.CCFdata[1,:])/scaling_factor)
        getCCFandG_LNA.lock.acquire()
        CCF = [self.CCFdata[1,i*scaling_factor] for i in range(range_fact)]
        getCCFandG_LNA.CCF = np.array(CCF, dtype='float32')
        getCCFandG_LNA.lock.release()
        
    def get_G_LNA(self, Start_freq, Stop_freq, scaling_factor):
        #Gaintemp = [self.G_LNAdata[1][cnt] for cnt in range(len(self.G_LNAdata[0][:])) if self.G_LNAdata[0][cnt] >= Start_freq and self.G_LNAdata[0][cnt] <= Stop_freq]
        print(self.G_LNAdata)
        range_fact = int(len(self.G_LNAdata[1][:])/scaling_factor)
        
        getCCFandG_LNA.lock.acquire()
        G_LNA = [self.G_LNAdata[1][i*scaling_factor] for i in range(range_fact)]
        getCCFandG_LNA.G_LNA = np.array(G_LNA, dtype='float32')
        getCCFandG_LNA.lock.release()
        
    