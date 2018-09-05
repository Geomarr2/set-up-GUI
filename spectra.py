import numpy as np
import ctypes as c
as_c = np.ctypeslib.as_ctypes

class Complex64(c.Structure):
    _fields_ = [
        ("i",c.c_float),
        ("q",c.c_float)
    ]

class Plan(c.Structure):
    _fields_ = [
        ("plan",c.c_void_p),
        ("input_d",c.c_void_p),
        ("output_d",c.c_void_p),
        ("spectrum_mean_d",c.c_void_p),
        ("spectrum_max_d",c.c_void_p),
        ("size",c.c_int32),
        ("batch",c.c_int32),
        ("input_bytes",c.c_int32),
        ("output_bytes",c.c_int32)
    ]


lib = c.WinDLL("./spectra.dll")    
lib.init.restype = c.c_void_p
lib.deinit.restype = c.c_void_p
lib.execute.restype = c.c_void_p
lib.init.argtypes = [c.POINTER(Plan)]
lib.deinit.argtypes = [c.POINTER(Plan)]
lib.execute.argtypes = [c.POINTER(Plan), c.POINTER(Complex64), 
                        c.POINTER(c.c_float), c.POINTER(c.c_float),
                        c.c_int]

                        
class FftDetect(object):
    def __init__(self,size,batch):
        self.input = np.empty(size*batch, dtype="complex64")
        self.output_mean = np.empty(size, dtype="float32")
        self.output_max = np.empty(size, dtype="float32")
        self.mean_spectra = []
        self.max_spectra = []
        self.size = size
        self.batch = batch
        self.plan = Plan()
        self.plan.size = size
        self.plan.batch = batch
        lib.init(c.byref(self.plan))
        
    def destroy(self):
        lib.deinit(c.byref(self.plan))

    def input_ptr(self,batch_offset=0):
        assert self.input.size == (self.size*self.batch), "Input is wrong size"
        f32 = self.input.view("float32")
        f32_c = as_c(f32)
        byte_offset = batch_offset*self.size*c.sizeof(Complex64)
        return c.cast(c.byref(f32_c,byte_offset),c.POINTER(Complex64))

    def output_mean_ptr(self):
        return as_c(self.output_mean)
        
    def output_max_ptr(self):
        return as_c(self.output_max)
        
    def execute(self,batch=None):
        if batch is None:
            batch = self.batch
        lib.execute(c.byref(self.plan), self.input_ptr(),
                    self.output_mean_ptr(), self.output_max_ptr(),
                    c.c_int(batch))
        self.mean_spectra.append(self.output_mean)
        self.max_spectra.append(self.output_max)
        self.output_mean = np.empty(self.size, dtype="float32")
        self.output_max = np.empty(self.size, dtype="float32")
        self.input[:] = 0+0j 
        # Note even with this we are still real time
        # but thus memset should really go to the device
        
    def clear(self):
        self.mean_spectra = []
        self.max_spectra = []


def main():
    size = 1<<19
    batch = 2
    x = FftDetect(size,batch)
    norm = np.random.normal
    r = (norm(0,1,size*batch) + norm(0,1,size*batch)*1j).astype("complex64")
    x.input[:] = r
    x.execute()
    powers = (abs(np.fft.fft(r.reshape(batch,size)))**2) 
    means = np.array(x.mean_spectra)
    if not np.allclose(means,powers.mean(axis=0),rtol=1e-4):
        raise Exception("Mean FFT results differ")
    else:
        print ("Numpy FFT+detection mean matches CUDA FFT+detection ")
        
    if not np.allclose(x.max_spectra[-1],powers.max(axis=0),rtol=1e-4):
        raise Exception("Max FFT results differ")
    else:
        print ("Numpy FFT+detection max matches CUDA FFT+detection ")
        
    x.destroy()
        
if __name__ == "__main__":
    main()
