from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse
import logging
import scipy
import cv2

LOGGING_FORMAT = "[%(asctime)s] (%(levelname)s) %(message)s"

def get_gaussian_kernel(sigma=1):
    logging.debug(f"Computing gaussian kernel with sigma={sigma}")
    number_of_coeffs = 3
    number_of_zeros = 0
    while number_of_zeros < 2 :
        delta = np.zeros(number_of_coeffs)
        delta[delta.size//2] = 1
        coeffs = scipy.ndimage.gaussian_filter1d(delta, sigma=sigma)
        number_of_zeros = coeffs.size - np.count_nonzero(coeffs)
        number_of_coeffs += 1
    logging.debug("Kernel computed")
    return coeffs[1:-1]

class cFlowdenoiseMPrunner():

    def __init__(self, **kwargs):
        #Processes all the inputs to variables self.<var>=input_var
        #Does not check

        #All these attributes will be the same amongst other processes.
        for k, v in kwargs.items():
            setattr(self, k, v)
            #print(k,":",v)

        
        self._dshape = tuple(self.dshape_list)

        self.updateKernels()

        #Set verbosity
        if self.verbosity == 2:
            logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG)
            logging.info("Verbosity level = 2")
        elif self.verbosity == 1:
            logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
            logging.info("Verbosity level = 1")        
        else:
            logging.basicConfig(format=LOGGING_FORMAT, level=logging.CRITICAL)
    

    def updateKernels(self):
        #Note that
        self._kernels = [None]*3
        self._kernels[0] = get_gaussian_kernel(self.ksigma_list[0])
        self._kernels[1] = get_gaussian_kernel(self.ksigma_list[1])
        self._kernels[2] = get_gaussian_kernel(self.ksigma_list[2])
        logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*self._kernels]]}")

    def run(self):
        logging.debug("subprocess_module: run()")
        # print("subprocess_module: __name__:", str(__name__))
        # print("subprocess_module: __file__:", str(__file__))

        #Launch multiprocesses based in the information

        for i0 in range(3): #Filter along the 3 axis
            self.axis_i= i0
            logging.info(f"Axis {self.axis_i} starting")
            axis_dim = self._dshape[self.axis_i]
            logging.info(f"axis_dim:{axis_dim}")

            logging.info(f"self.number_of_processes:{self.number_of_processes}")

            chunk_size = axis_dim//(self.number_of_processes)
            n_remain_slices = axis_dim % self.number_of_processes
            logging.info(f"n_remain_slices:{n_remain_slices}")

            #Arguments for PoolExecutor
            chunk_start_indexes = [i*chunk_size for i in range(self.number_of_processes)]
            chunk_sizes = [chunk_size]*(self.number_of_processes)
            if n_remain_slices>0: #last slices
                chunk_start_indexes.append(self.number_of_processes*chunk_size )
                chunk_sizes.append(n_remain_slices)
    
            nprocesses = len(chunk_start_indexes)
            with ProcessPoolExecutor(max_workers=self.number_of_processes) as executor:
                tasks_range = range(nprocesses)
                for result in executor.map(self.chunk_worker,tasks_range,chunk_start_indexes,chunk_sizes):
                    logging.info(f"Process {result} finished")
            
            logging.info(f"Axis {self.axis_i} completed")

            #Copies contents of filtered volume to new datavol, readying for next axis
            in_sh = shared_memory.SharedMemory(name=self.in_data_sh_name)
            out_sh = shared_memory.SharedMemory(name=self.out_data_sh_name)
            data_vol0 = np.ndarray(self._dshape, dtype=self.dtype_name, buffer=in_sh.buf)
            filtered_vol0 = np.ndarray(self._dshape, dtype=self.dtype_name, buffer=out_sh.buf)
            data_vol0[...] = filtered_vol0[...]  
            in_sh.close()
            out_sh.close()

        logging.info("All processes completed")


    def chunk_worker(self, task_i, chunk_start_idx, chunk_size):
        # gets shared memory
        in_sh = shared_memory.SharedMemory(name=self.in_data_sh_name)
        out_sh = shared_memory.SharedMemory(name=self.out_data_sh_name)
        progr_sh = shared_memory.SharedMemory(name=self.progress_sh_name)
        
        try:
            data_vol0 = np.ndarray(self._dshape, dtype=self.dtype_name, buffer=in_sh.buf)
            filtered_vol0 = np.ndarray(self._dshape, dtype=self.dtype_name, buffer=out_sh.buf)
            progress0 = np.ndarray((3), dtype=np.uint32, buffer=progr_sh.buf)

            for i in range(chunk_size):
                #Work slice-by-slice
                self.filter_along_axis_slice(chunk_start_idx + i , data_vol0, filtered_vol0)    

                #Update progress
                progress0[self.axis_i]+=1

            return task_i

        except Exception as e:
            logging.error("ERROR: The following error occurred:")
            logging.error(str(e))
        finally:
            in_sh.close()
            out_sh.close()
            progr_sh.close()
            #Close but don't destroy

        return task_i
    
    def filter_along_axis_slice(self,islice, data_vol0, filtered_vol0):
        axis_i = self.axis_i
        logging.debug(f"filter_along_axis_slice() with islice:{islice},  axis_i:{axis_i}")
 
        kernel= self._kernels[axis_i]

        assert kernel.size % 2 != 0, "Error kernel.size must be odd"

        ks2 = kernel.size//2
        assert kernel.size== 2*ks2+1 , "Error"
        
        #logging.info("Transposing if needed")
        if axis_i==0:
            data_vol_transp=data_vol0 #Potentially GIL locking making slow execution
            # data_vol_transp= data_vol0.copy() #Makes no difference in linux
        elif axis_i==1:
            data_vol_transp=np.transpose(data_vol0,axes=(1,0,2))
        elif axis_i==2:
            data_vol_transp=np.transpose(data_vol0,axes=(2,0,1))

        data_islice = data_vol_transp[islice,:,:]
        #data_islice = data_roi[ks2,:,:]
        tmp_slice = np.zeros_like(data_islice).astype(np.float32)
        slice_shape= tmp_slice.shape

        if self.useOF:
            prev_flow = np.zeros(shape=(*slice_shape, 2), dtype=np.float32)

            # self.slice_filter_method=0
            data_roi = np.roll(data_vol_transp, shift= -(islice- ks2), axis=0)[:kernel.size,:,:] #Gets ROI by rolling and then cropping
            #i_values= []
            #Down-side
            for i in range(ks2-1,-1,-1): #data slices from middle-1 to down boundary
                ref = data_roi[i,:,:]
                flow = self.get_flow(ref, data_islice, prev_flow)
                prev_flow = flow
                OF_compensated_slice = self.warp_slice(ref, flow)
                tmp_slice += OF_compensated_slice * kernel[i]
                #i_values.append(i)  

            tmp_slice += data_islice * kernel[ks2] #Middle slice, no OF needed, just kernel convolve
            #i_values.append(ks2)

            prev_flow = np.zeros(shape=(*slice_shape, 2), dtype=np.float32)
            #Up-side
            for i in range(ks2+1,kernel.size, +1):
                ref = data_roi[i,:,:]
                flow = self.get_flow(ref, data_islice, prev_flow)
                prev_flow = flow
                OF_compensated_slice = self.warp_slice(ref, flow)
                tmp_slice += OF_compensated_slice * kernel[i]
                #i_values.append(i)         

        else:
            #No OF
            #Simple 2D convolution (2D*2D) and sum along z axis? with circularity
            #Can maybe accelerated using scipy.ndimage.convolve() with 'wrap' setting
            for i in range(kernel.size):
                tmp_slice += data_vol_transp[(islice + i - ks2) % data_vol_transp.shape[0], :, :]*kernel[i]


        #logging.info("Restoring orientation")
        if axis_i==0:
            filtered_vol0[islice, :, :] = tmp_slice
        elif axis_i==1:
            filtered_vol0[:, islice, :] = tmp_slice
        elif axis_i==2:
            filtered_vol0[:, :, islice] = tmp_slice


    def warp_slice(self,reference, flow):
        height, width = flow.shape[:2]
        map_x = np.tile(np.arange(width), (height, 1))
        map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
        map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
        warped_slice = cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped_slice

    def get_flow(self, reference, target, prev_flow=None):

        if self.bComputeFlowWithPreviousFlow:
            flags0 = cv2.OPTFLOW_USE_INITIAL_FLOW
            flow0 = prev_flow   
        else:
            flags0 = 0
            flow0 = None    

#        flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=flow0, pyr_scale=0.5, levels=int(self.OF_LEVELS), winsize=int(self.OF_WINDOW_SIZE), iterations=int(self.OF_ITERS), poly_n=self.OF_POLY_N, poly_sigma=self.OF_POLY_SIGMA, flags=flags0)
        flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=flow0, pyr_scale=0.5, levels=self.levels, winsize=self.winsize, iterations=self.iters, poly_n=5, poly_sigma=1.2, flags=flags0)

        return flow


def parseArgs():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--in_data_sh_name",
                        help="Input data shared memory name",
                        type=str)
    parser.add_argument("--out_data_sh_name",
                        help="Output data shared memory name",
                        type=str)
    parser.add_argument("--dtype_name", type=str)
    parser.add_argument("--dshape_list", nargs="+", type=int)
    parser.add_argument("--levels", type=int)
    parser.add_argument("--winsize", type=int)
    parser.add_argument("--bComputeFlowWithPreviousFlow", action="store_true")
    parser.add_argument("--useOF", action="store_true")
    parser.add_argument("--ksigma_list", type=float, nargs="+", help="kernel size")
    parser.add_argument("--iters", type=int)
    parser.add_argument("--number_of_processes", type=int, help="number of processes")
    parser.add_argument("--verbosity", type=int, help="Verbosity level", default=0)
    parser.add_argument("--progress_sh_name",
                        help="Progress shared memory name",
                        type=str)

    return parser

if __name__=="__main__":
    logging.info(f"_flowdenoising_subprocess.py, running __main__")
    parser = parseArgs()
    logging.info("arguments parsed")

    args_dict = vars(parser.parse_args()) #vars converts namespace to dict
    logging.info("Args:", str(args_dict))

    myrunner= cFlowdenoiseMPrunner(**args_dict)
    myrunner.run()
    logging.info("End  of _flowdenoising_subprocess __main__")