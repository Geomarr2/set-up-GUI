import threading 
import numpy as np

class ReduceData(threading.Thread):
    FreqMaxMinValues = {}
    #MinValues = {}
    lock = threading.Lock()
    
    def __init__(self,original_data,Cfreq, plot_num, scaling_factor, bw):
        threading.Thread.__init__(self)
        self.original_data = original_data
        self.Cfreq = Cfreq   
        self.plot_num = plot_num
        self.scaling_factor = scaling_factor
        self.bw = bw
        ReduceData.lock.acquire()
        ReduceData.FreqMaxMinValues[Cfreq] = []
        ReduceData.lock.release()

    def read_reduce_Data(self):
        # read in the whole BW in one array
# set the display sample size depending on the display bandwidth and resolution  
     #   t0 = time.time()
        
        spec = self.dBuV_M2V_M(self.original_data)
        #freq = spectrum[0]
        
        x = int(len(self.original_data)/self.scaling_factor)
        spec_min = np.array([], dtype=np.float32)
        spec_max = np.array([], dtype=np.float32)
        freq = np.array([], dtype=np.float32)
        Start_freq = self.Cfreq -self.bw/2
        Stop_freq = self.Cfreq +self.bw/2
        spec_max = [np.max(spec[(i*x):(x*i+x)]) for i in range (self.scaling_factor)]
        spec_min = [np.min(spec[(i*x):(x*i+x)]) for i in range (self.scaling_factor)]
        spec_max = self.V_M2dBuV_M(spec_max)
        spec_min = self.V_M2dBuV_M(spec_min)
        print(self.Cfreq)
        freq = np.linspace(Start_freq,Stop_freq,len(spec_max)) 
        temp = freq,spec_max,spec_min
        ReduceData.lock.acquire()
        ReduceData.FreqMaxMinValues[self.Cfreq] = temp
        ReduceData.lock.release()
    
    def dBuV_M2V_M(self,spec):
        VperM = pow(10,(spec-120)/20)
        return VperM    
    
    def V_M2dBuV_M(self,spec):
        dBuV_M = 20*np.log10(spec)+120
        return dBuV_M  
    
class plotInfo():
    def __init__(self):
        self.newWindow = Tk()
        self.newWindow.mainloop()
    def createWindow(self, txt):
        scroll_txt = scrolledtext.ScrolledText(self.newWindow,width=40,height=10).pack()
        print(txt)
        scroll_txt.insert(INSERT, txt)
    def deleteWindow(self):
        self.newWindow.destroy()