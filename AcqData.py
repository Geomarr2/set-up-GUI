# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:50:10 2017

@author: Geomarr
"""
import os
import ctypes as c
import numpy as np
import math as math
import matplotlib.pyplot as plt
import time
import logging
from threading import Timer,Event,Lock
#import spectra
import tektronix_errors

pathCaldata = "D:/RFIData/Effelsberg/Webcam-DCS7513/20171128/01_KalibrationData_background/" #Path to save the calibration data spectra
DataPath = "D:/Geomarr/Spectrum/FullSpectrum"       #Path to save the spectra
integrationTime = 1          #integration time in sec
startFreqMHz = 1000                        #start frequency in MHz
stopFreqMHz  = 1040            #stop frequency in MHz
displayResolution = 1            # times 107Hz
title = "Power Spetrum"
usecase = 0  # 0 = plain data taking; 1 = calibrate data; 2 = acquire calibration data; 3 = start RFI data; 4 = acquire background data
GPU_integration_time = 3
color = "g"
DataPath_gLNA = 20
DataPath_Lcable = -1
DataPath_Eff_ant = 0.75

#### Programm please do not change


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
_log = logging.getLogger("spectrometer")

def load_library(name,path):
    _log.info("loading library %s\\%s"%(path,name))
    cwd = os.getcwd()
    os.chdir(path)
    lib = c.cdll.LoadLibrary(name)
    os.chdir(cwd)
    return lib

rsa306 = load_library("RSA_API.dll","C:\\Tektronix\\RSA_API\\lib\\x64")

class IQSTRMIQINFO(c.Structure):
        _fields_ = [("timestamp", c.c_uint64),
                    ("triggerCount", c.c_int),
                    ("triggerIndices", c.POINTER(c.c_int)),
                    ("scaleFactor", c.c_double),
                    ("acqStatus", c.c_uint64)]

class Cplx32(c.Structure):
    _fields_ =[("i", c.c_float),
               ("q", c.c_float)]

class LogException(Exception):
    def __init__(self,message):
        _log.error(message)
        super(LogException,self).__init__(message)
               
def safe_call(func,*args):
    retval = func(*args)
    if retval != 0:
        errorstr = tektronix_errors.codes.get(retval,"Unknown")
        if args:
            argstr = ", ".join([repr(arg) for arg in args])
            message = "Error on call to %s with args (%s)"%(func.__name__,argstr)
        else:
            message = "Error on call to %s"%(func.__name__)
        raise LogException(message+" (Type %d: %s)"%(retval,errorstr))               
        
def init_device():
    count = c.c_int(0) 
    idx = (c.c_int * 10)()
    temp = " "*25
    serial = c.c_char_p(temp.encode('utf-8'))
    dtype = c.c_char_p(temp.encode('utf-8'))
    _log.info("Searching for device...")
    safe_call(rsa306.DEVICE_Search,c.byref(count), idx, serial, dtype)
    if count.value < 1:
        raise LogException('No Tektronix devices found.')
    elif count.value > 1:
        raise LogException('Too many Tektronix devices found (%d).'%(count.value))
    _log.info('Device type: {}'.format(dtype.value))
    _log.info('Device serial number: {}'.format(serial.value))
    _log.info('Connecting to device...')
    safe_call(rsa306.DEVICE_Connect,idx[0])
    _log.info('Loading preset configuration...')
    safe_call(rsa306.CONFIG_Preset)
    _log.info('Initialization complete.')           
        
 
class TektronixDevice(object):
    def __init__(self):
        self.cfreq = 0
        self.bw = 0
        self.true_bw = 0
        self.buffer_size = 0
        self.running = Event()
        self.connected = Event()
        self.init()
        
    def init(self):
        init_device()
        self.set_cfreq()
        self.set_bandwidth()  
        self.set_buffer_size()
        self.connected.set()
        
    def set_cfreq(self, cfreq=100e6):
        tmp = c.c_double(0)
        _log.info("Setting centre frequency to %.3f MHz"%(cfreq/1e6))
        safe_call(rsa306.CONFIG_SetCenterFreq,c.c_double(cfreq))
        safe_call(rsa306.CONFIG_GetCenterFreq,c.byref(tmp))
        self.cfreq = tmp.value
        if not np.isclose(cfreq,self.cfreq):
            _log.warning("Centre frequency set to %.3f MHz"%(self.cfreq/1e6))
    
    def set_bandwidth(self, bandwidth=40e6):
        tmp0,tmp1 = c.c_double(0),c.c_double(0)
        _log.info("Setting bandwidth to %.3f MHz"%(bandwidth/1e6))
        rsa306.IQSTREAM_ClearAcqStatus()
        safe_call(rsa306.IQSTREAM_SetAcqBandwidth,c.c_double(bandwidth))
        safe_call(rsa306.IQSTREAM_GetAcqParameters, c.byref(tmp0), c.byref(tmp1))
        self.bw = tmp0.value
        self.true_bw = tmp1.value
        if not np.isclose(bandwidth,self.bw):
            _log.warning("Bandwidth set to %.3f MHz"%(self.bw/1e6))
        
    def set_buffer_size(self, size=523392):  
        tmp = c.c_int(0)
        _log.info("Setting buffer size to %d samples"%(size))
        safe_call(rsa306.IQSTREAM_SetOutputConfiguration,c.c_int(0), c.c_int(0))
        safe_call(rsa306.IQSTREAM_SetIQDataBufferSize,c.c_int(size))
        safe_call(rsa306.IQSTREAM_GetIQDataBufferSize,c.byref(tmp))
        self.buffer_size = tmp.value
        if self.buffer_size != size:
            _log.warning("Buffer size set to %d"%(self.buffer_size))
             
    def run(self):
        _log.info("Running device...")
        self.running.set()
        safe_call(rsa306.DEVICE_Run)
    
    def get_stream(self):
        assert self.running.is_set(), "Device must be running to enable stream"
        return Stream(self)
            
    def stop(self):
        self.running.clear()
        safe_call(rsa306.DEVICE_Stop)
        _log.info("Device stopped.")
                        
    def disconnect(self):
        self.running.clear()
        safe_call(rsa306.DEVICE_Disconnect)
        self.connected.clear()
        _log.info("Disconnected from device.")
        
        
class Stream(object):
    def __init__(self,device):
        self.device = device
        self.running = Event()
        self.lock = Lock()
        self.timer = None
        self._iqlen = c.c_int(0)
        self._iqinfo = IQSTRMIQINFO()
        _log.info("Stream prepared.")
        
    def start(self,duration=None):
        if self.is_running():
            self.stop()
        _log.info("Starting IQ stream...")
        safe_call(rsa306.IQSTREAM_Start)
        self.running.set()
        if duration is not None:
            _log.info("Scheduling stream end in %.2f seconds."%duration)
            self.timer = Timer(duration,self.stop)
            self.timer.start()
    
    def stop(self):
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
            _log.info("Stopping IQ stream...")
            safe_call(rsa306.IQSTREAM_Stop)
        self.running.clear()
    
    def read(self,buffer):
        with self.lock:
            _log.debug("Reading from IQ stream...")
            safe_call(rsa306.IQSTREAM_GetIQData, buffer, c.byref(self._iqlen), c.byref(self._iqinfo))
        return self._iqlen.value

    def is_running(self):
        return self.running.is_set()

        
def process_stream(device,fft_engine,duration):
    nsamp = fft_engine.batch * fft_engine.size
    batch_duration = nsamp/device.true_bw
    stream = device.get_stream()
    stream.start(duration)
    while stream.is_running():
        batch_count = 0
        for batch_idx in range(fft_engine.batch):
            while stream.is_running():
                time.sleep(0.001)   
                nread = stream.read(fft_engine.input_ptr(batch_idx))
                if nread != 0:
                    _log.debug("Read %d samples from stream"%nread)
                    batch_count += 1
                    break
                else:
                    _log.debug("Empty stream read, trying again...")
        _log.info("Forming spectrum...")
        _log.info("%i batches have been used to form a spectra."%(batch_count))
        start_time = time.time()
        fft_engine.execute(batch_count)
        exec_time = time.time()-start_time
        _log.info("Took %.3f seconds to process %.3f seconds of data."%(exec_time,batch_duration))

# genertes a list of standart center frequencies        
def generate_CFlist(startfreq, stopfreq):
    lowerlimit = 80e6 #  Hz Defines the lower frequency limit
    upperlimit = 6000e6 # Hz Defines the upper freqzency limit
    if startfreq > stopfreq: 
        temp = startfreq
        startfreq = stopfreq
        stopfreq = temp
    if lowerlimit >= startfreq and startfreq <= upperlimit: startfreq = lowerlimit
    if lowerlimit >= stopfreq and stopfreq <= upperlimit: stopfreq = upperlimit
    if startfreq > stopfreq: 
        temp = startfreq
        startfreq = stopfreq
        stopfreq = temp
        
    cfreq_tabelle = list(range(int(round((stopfreq/1e6-80)/40+0.499999))-int((startfreq/1e6-80)/40)))
    
    for i in range(len(cfreq_tabelle)):
        cfreq_tabelle[i]= ((i+int((startfreq/1e6-80)/40))*40+100)*1e6
    return cfreq_tabelle

def to_decibels(x):
    calfactor = 1000/50/(523392/2)*(523392/2)                    
    return 10*np.log10(x*x*calfactor)
    
def convertPower2ElecFieldStrength(spectrum): # returns in [dBuV/m]
    Z = 119.9169832 * np.pi  # Impedance of freespace
    gLNA= 10.0 #dB
    gCable = -1.0  #dB
    antennaEfficiency = 0.75 
    r = 1.0  # Distance DUT to Antenna
    CCF =-20.0  # Chamber Calibration Factor in dBm
    temp = -gLNA - gCable - (10.0 * np.log10(antennaEfficiency)) - CCF + (10.0 * np.log10(Z / (4.0 * np.pi * (r *r)))) + 90.0  
    return spectrum[0], spectrum[1]+temp

def change_freq_channel(spectrum, factor):
    outputChannel = int(len(spectrum[0])/factor)
    outputfreqlist = np.zeros(outputChannel)
    outputspeclist = np.zeros(outputChannel)
    for i in range(outputChannel):
        outputfreqlist[i] = np.mean(spectrum[0][i*factor:(i*factor+factor)])
        outputspeclist[i] = np.mean(spectrum[1][i*factor:(i*factor+factor)])
    return outputfreqlist, outputspeclist

def trim_fft(device,fft,f):
    _log.info("Usable Bandwidth: %.2f MHz"%(device.bw/1e6))
    final_sample = int(math.ceil(device.buffer_size/(device.true_bw/device.bw)))
    _log.info("Usable number of Samples: %i "%(final_sample))
    starttrim = int((fft.size-final_sample)/2) 
    stoptrim = int(fft.size-(fft.size-final_sample)/2)
    return fft[starttrim:stoptrim],f[starttrim:stoptrim]

def trim_spectrum(spectrum):
    final_sample= 373852
    specsize=len(spectrum[0])
    _log.info("Usable number of Samples: %i "%(final_sample))
    starttrim = int((specsize-final_sample)/2) 
    stoptrim = int(specsize-(specsize-final_sample)/2)
    freq = np.array(spectrum[0])
    fft= np.array(spectrum[1])
    return freq[starttrim:stoptrim],fft[starttrim:stoptrim]

def mean_batches(fftarray):
    meanfft =0
    for x in range(len(fftarray)):
        meanfft = fftarray[x] + meanfft
    return meanfft/len(fftarray)
        
def plot_spectrumold(device,fft):
    bw = device.true_bw
    resolution = bw/device.buffer_size
    lower = device.cfreq-bw/2
    upper = device.cfreq+bw/2
    f = np.linspace(lower,upper,device.buffer_size)/1e6
    #max_specs = to_decibels(np.fft.fftshift(np.array(fft.max_spectra)))
    mean_specs = to_decibels(np.fft.fftshift(np.array(mean_batches(fft.mean_spectra))))
    #trimmed_max_spec,ftrim = trim_spectrum(device,max_specs.max(axis=0),f)
    trimmed_mean_spec,ftrim = trim_fft(device,mean_specs,f)
    
    #write_spec_to_file(ftrim, trimmed_mean_spec,trimmed_mean_spec, "D:/RFIData/filetestnew.txt")
    
    _log.info("Generating plots (%.2f to %.2f MHz)"%(lower/1e6,upper/1e6))
    #plt.plot(f,max_specs.max(axis=0),c="r")
    #plt.plot(f,mean_specs.max(axis=0),c="b")
    #plt.plot(ftrim,trimmed_max_spec, c="r")
    plt.plot(ftrim,trimmed_mean_spec, c="r")
    #plt.ylim(-80,0)
    plt.ylabel("Power (dB)")
    plt.xlabel("Frequency (MHz) (resolution %.2f Hz)"%resolution)

def plot_stiched_spectrum(spectrum, color, resfact = 1):
    
    trimspec = np.array(change_freq_channel(trim_spectrum(spectrum),resfact))
    #trimspec = np.array(trim_spectrum(spectrum))
    
    spec = to_decibels(trimspec[1])
    resolution =  0.1069943751528491*resfact
    
    #_log.info("Generating plots (%.2f to %.2f MHz)"%(lower/1e6,upper/1e6))
    plt.plot(trimspec[0]/1e6,spec, c=color)
    #plt.ylim(-80,0)
    plt.ylabel("Power (dB)")
    plt.xlabel("Frequency (MHz) (resolution %.3f kHz)"%resolution)
    
def freq_scale(device):
    bw = device.true_bw
    resolution = bw/device.buffer_size
    lower = device.cfreq-bw/2
    upper = device.cfreq+bw/2
    return np.linspace(lower,upper,device.buffer_size)
    
def spectra_linear(device,fft): 
    #max_specs = to_decibels(np.fft.fftshift(np.array(fft.max_spectra)))
    
    return freq_scale(device), np.fft.fftshift(np.array(mean_batches(fft.mean_spectra)))

    
def generate_blanklist(spectrum):
    temp = np.array(spectrum[1])
    cutoff = temp.mean()*2
    for i in range(len(temp)):
        if temp[i] > cutoff: 
            temp[i]=0
            temp[i-1]=0
            temp[i+1]=0 
        else: temp[i]=1
    return spectrum[0], temp

    
def save_spectrum2file(spectrum, path,filename):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(path+filename,  spectrum)
    
def load_spectrum(path, filename):
    return np.load(path+filename)

def save_fft2file(device, fft, path):
    bw = device.true_bw
    resolution = bw/device.buffer_size
    lower = device.cfreq-bw/2
    upper = device.cfreq+bw/2
    f = np.linspace(lower,upper,device.buffer_size)/1e6
    mean_specs = to_decibels(np.fft.fftshift(np.array(mean_batches(fft.mean_spectra))))
    trimmed_mean_spec,ftrim = trim_fft(device,mean_specs,f)
    outputData = np.array([ftrim,trimmed_mean_spec])
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(path+"Spectrum_"+str(device.cfreq/1e6)+ "MHz",  outputData.T)
    
def calibrateData(spectrum, centerFrequency, path):
    blanklist = np.load(path+"BlanckFile"+str(int(centerFrequency/1e6))+ "MHz.npy")
    noisespec = np.load(path +"50OhmRefSpectrum"+str(int(centerFrequency/1e6))+ "MHz.npy")
    if np.array_equiv(noisespec[0],blanklist[0]) and np.array_equiv(noisespec[0],spectrum[0]):
         _log.info("Calibration successfully data loaded.")
    else: 
        _log.info("Calibration data different frequency scale.")
        return        
    return spectrum[0], (spectrum[1]-noisespec[1])*blanklist[1]
    
    
        
def main(min_freq,max_freq,integration_time,path):
    # integration_time = GPU integration time
    device = TektronixDevice()
    batch = int(integration_time*device.bw/device.buffer_size)
    dump_time = batch*device.buffer_size/device.bw
    
    _log.info("Setting dump time to %.3f seconds."%(dump_time))
    fft = spectra.FftDetect(device.buffer_size,batch)

    """
    ################################################################
    Start IQ streaming 
    ################################################################
    """
    device.run()
    steps = math.ceil((max_freq-min_freq)/device.bw)
    act_steps =1
    
    #for freq in np.arange(min_freq,max_freq,device.bw):
    for freq in range(int(min_freq+device.bw/2), int(max_freq+device.bw/2), int(device.bw)):
        device.set_cfreq(freq)
        _log.info("Frequency step %i of %i."%(act_steps,steps))
        process_stream(device,fft,0.04)
        plot_stiched_spectrum(device,fft)
        if path != "":
            save_fft2file(device,fft, path)
        _log.info("Size of fft mean_spectra: "+ str(np.size(fft)))
        fft.clear()
        act_steps +=1
        
    device.stop()
    plt.tight_layout()
    plt.show()
    fft.destroy()
    device.disconnect()
        
def acquire_data(min_freq,max_freq,integration_time,path, pathCaldata, displayResolution=1, title ="", usecase = 0):
    # usecase:  0 = plain data taking
    #           1 = calibrate data
    #           2 = acquire calibration data
    #           3 = start RFI data
    #           4 = acquire background data
    GPU_integration_time = 2.0
    device = TektronixDevice()
    batch = int(GPU_integration_time*device.bw/device.buffer_size)
    dump_time = batch*device.buffer_size/device.bw
    _log.info("Setting dump time to %.3f seconds."%(dump_time))
    fft = spectra.FftDetect(device.buffer_size,batch)
    """
    ################################################################
    Start IQ streaming 
    ################################################################
    """
    device.run()
    Cfreqlist = generate_CFlist(min_freq,max_freq)
    act_steps =1
    
    for i in range(len(Cfreqlist)):
        device.set_cfreq(int(Cfreqlist[i]))
        _log.info("Frequency step %i of %i."%(act_steps,len(Cfreqlist)))
        process_stream(device,fft,integration_time)
        spectrum = spectra_linear(device,fft)
        if usecase == 0:
            plot_stiched_spectrum(spectrum, "k")
            save_spectrum2file(spectrum, path,"RFISpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
        if usecase == 1:
            save_spectrum2file(spectrum, path,"RFISpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
            plot_stiched_spectrum(calibrateData(spectrum, Cfreqlist[i], path),"r",displayResolution)
        if path != "" and usecase == 2:
            plot_stiched_spectrum(spectrum, "k")
            save_spectrum2file(generate_blanklist(spectrum), path,"BlanckFile"+str(int(device.cfreq/1e6))+ "MHz" )
            save_spectrum2file(spectrum, path,"50OhmRefSpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
        if path != "" and usecase == 3:
            plot_stiched_spectrum(spectrum, "r", displayResolution)
            save_spectrum2file(spectrum, path,"IntTime_RFISpectrum"+str(int(device.cfreq/1e6))+ "MHz" )
            # Load Background spectra and plot
            filename = "Background"+str(int(device.cfreq/1e6))+ "MHz.npy"
            plot_stiched_spectrum(load_spectrum(pathCaldata, filename), "b",displayResolution)            
        if path != "" and usecase == 4: #Generates and saves the background spectra
            plot_stiched_spectrum(spectrum, "c",displayResolution)
            save_spectrum2file(spectrum, pathCaldata,"Background"+str(int(device.cfreq/1e6))+ "MHz" )
        _log.info("Size of fft mean_spectra: "+ str(np.size(fft)))
        fft.clear()
        act_steps +=1
    
    device.stop()
    plt.tight_layout()
    plt.title(title)
    
    plt.show()
    fft.destroy()
    device.disconnect()
    
if __name__ == "__main__":
#    _log.setLevel(logging.INFO) 
    #main(300e6,320e6,2.00,path)
    
    #for i in range(0, 3):
    #    DataPath = DataPath + str(i)
    acquire_data(startFreqMHz*1e6,stopFreqMHz*1e6,integrationTime,DataPath,DataPath_gLNA,DataPath_Lcable,DataPath_Eff_ant,GPU_integration_time, displayResolution, title, usecase, color)
        