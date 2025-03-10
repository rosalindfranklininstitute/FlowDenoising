#!/usr/bin/env python
'''3D Gaussian filtering controlled by the optical flow.
'''

# "flowdenoising.py" is part of "https://github.com/microscopy-processing/FlowDenoising", authored by:
#
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).
#
# Please, refer to the LICENSE.txt to know the terms of usage of this software.

"""
2023 Significantly modified version by Luis Perdigao @RFI UK

Code put in a class so that it be run in a notebook or integrated in other programs

Can be ran as module.

There are two ways it can do parallel calculation: threaded or multiprocess
This is more of a problem in Linux. In windows the multi-threading seems to work fine


"""

import logging
import os
import numpy as np
import cv2
import scipy.ndimage
import time
import skimage.io
import mrcfile
import argparse
import threading

import multiprocessing

import random
import concurrent.futures

import pathlib

LOGGING_FORMAT = "[%(asctime)s] (%(levelname)s) %(message)s"

# __fdn_progress_sm_value__ = Value('f', 0)

FDN_LIZZIE="Is great"

stopEv=threading.Event() #Global

class cFlowDenoiser():
    OFCA_EXTENSION_MODE = cv2.BORDER_REPLICATE
    OF_LEVELS = 3
    OF_WINDOW_SIZE = 5
    OF_ITERS = 3
    OF_POLY_N = 5
    OF_POLY_SIGMA = 1.2
    #SIGMA = 2.0

    _data_vol=None
    _filtered_vol=None

    #__percent__ = Value('f', 0)
    __percent__=0

    def __init__(self, *,
        sigma=(2.0,2.0,2.0),
        levels=3 ,
        winsize=5 ,
        max_number_of_processes=None,
        bComputeFlowWithPreviousFlow=True,
        timeout_mins = 30,
        use_OF=True,
        process_mode='threaded', #choices=['threaded', 'sequential','multiproc'],
        verbosity=0
        ):

        #print(f"levels:{levels}")
        #print(f"winsize:{winsize}")
        self.SIGMA= sigma
        self.OF_LEVELS= levels
        self.OF_WINDOW_SIZE= winsize
        
        number_of_PUs = multiprocessing.cpu_count()
        logging.info(f"Number of processing units: {number_of_PUs}")

        self.max_number_of_processes=max_number_of_processes
        if max_number_of_processes is None:

            self.max_number_of_processes = number_of_PUs
            
        self.bComputeFlowWithPreviousFlow = bComputeFlowWithPreviousFlow
        self.timeout_mins = timeout_mins

        self.data_shape=None
        self.data_type=None

        self.use_OF=use_OF
        self.calculation_interrupt=False

        self.sm_name_suffix = str(random.randint(1000,9999))

        self.process_mode=process_mode

        self.slice_filter_method=0 #Two methods currently being tested

        # setup kernels here
        self.updateKernels()

        self.verbosity=verbosity

        self.mp_progress_np=None #To share progress in subprocess parallel execution

    def updateKernels(self):
        #Note that
        self._kernels = [None]*3
        self._kernels[0] = self.get_gaussian_kernel(self.SIGMA[0])
        self._kernels[1] = self.get_gaussian_kernel(self.SIGMA[1])
        self._kernels[2] = self.get_gaussian_kernel(self.SIGMA[2])
        logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*self._kernels]]}")

    @staticmethod
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


    def warp_slice(self,reference, flow):
        height, width = flow.shape[:2]
        map_x = np.tile(np.arange(width), (height, 1))
        map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
        map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
        warped_slice = cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=self.OFCA_EXTENSION_MODE)
        return warped_slice

    def get_flow(self, reference, target, prev_flow=None):

        if __debug__:
            time_0 = time.perf_counter()

        if self.bComputeFlowWithPreviousFlow:
            flags0 = cv2.OPTFLOW_USE_INITIAL_FLOW
            flow0 = prev_flow   
        else:
            flags0 = 0
            flow0 = None    

        flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=flow0, pyr_scale=0.5, levels=int(self.OF_LEVELS), winsize=int(self.OF_WINDOW_SIZE), iterations=int(self.OF_ITERS), poly_n=self.OF_POLY_N, poly_sigma=self.OF_POLY_SIGMA, flags=flags0)

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
        return flow

    def do_filter(self):

        self.__percent__=0

        self.calculation_interrupt=False
 
        stopEv.clear()
        feddback_thread = threading.Thread(target=self.feedback_periodic, args=(stopEv,))
        feddback_thread.daemon = True # To obey CTRL+C interruption.
        #This also means that the thread is killed when the program exits
        feddback_thread.start()

        if not 'multiproc' in self.process_mode:
            logging.info("Filtering along Z")
            self.filter_along_axis(self._kernels[0],  axis_i=0)
            self._data_vol[...] = self._filtered_vol[...]

            logging.info("Filtering along Y")
            self.filter_along_axis(self._kernels[1], axis_i=1)
            self._data_vol[...] = self._filtered_vol[...]

            logging.info("Filtering along X")
            self.filter_along_axis(self._kernels[2], axis_i=2)
            self._data_vol[...] = self._filtered_vol[...]
        else:
            #Run in multiprocess mode with subprocess
            self.run_OF_MP()
            
            #Collect result
            self._data_vol[...] = self._filtered_vol[...]


        #When filtering is complete it continues here
        #stop feedback_thread.
        # There no stop() function to do that, so Event() is used
        logging.info("Setting stopEv to stop feedback thread")
        stopEv.set()

    def filter_along_axis_slice(self,islice, kernel, axis_i):
        logging.debug(f"filter_along_axis_slice() with islice:{islice},  axis_i:{axis_i}")
        #print (f"filter_along_axis_slice() with islice:{islice},  axis_i:{axis_i}, kernel.size:{kernel.size}")
        
        data_vol0=self._data_vol
        filtered_vol0=self._filtered_vol

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

        if self.use_OF:
            prev_flow = np.zeros(shape=(*slice_shape, 2), dtype=np.float32)

            if self.slice_filter_method==0:
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
                #print(f"i_values down:{i_values}")

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
                #print(f"i_values down-up:{i_values}")           

            else:
                #Note that the mod (%) is used here to introduce circularity in picking adjacent slices
                for i in range(ks2 - 1, -1, -1):
                    flow = self.get_flow(data_vol_transp[(islice + i - ks2) % data_vol_transp.shape[0], :, :],
                                    data_vol_transp[islice, :, :], prev_flow)
                    #indices of reference slice go from islice+ks2-1-ks2 = islice-1
                    # decreasing to islice-ks2 (including)

                    prev_flow = flow
                    OF_compensated_slice = self.warp_slice(data_vol_transp[(islice + i - ks2) % data_vol_transp.shape[0], :, :], flow)
                    tmp_slice += OF_compensated_slice * kernel[i]
                tmp_slice += data_vol_transp[islice, :, :] * kernel[ks2]
                prev_flow = np.zeros(shape=(*slice_shape, 2), dtype=np.float32)
                
                for i in range(ks2 + 1, kernel.size):
                    flow = self.get_flow(data_vol_transp[(islice + i - ks2) % data_vol_transp.shape[0], :, :],
                                    data_vol_transp[islice, :, :], prev_flow)
                    #indices of reference slice go from islice+ks2+1-ks2 = islice+1
                    #increasing to islice+kernel.size-1-ks2 (including)
                    prev_flow = flow
                    OF_compensated_slice = self.warp_slice(data_vol_transp[(islice + i - ks2) % data_vol_transp.shape[0], :, :], flow)
                    tmp_slice += OF_compensated_slice * kernel[i]
        else:
            #No OF
            #Simple 2D convolution (2D*2D) and sum along z axis? with circularity
            #Can maybe accelerated using scipy.ndimage.convolve() with 'wrap' setting
            for i in range(kernel.size):
                tmp_slice += data_vol_transp[(islice + i - ks2) % data_vol_transp.shape[0], :, :]*kernel[i]


        #logging.info("Restoring orieantion")
        if axis_i==0:
            filtered_vol0[islice, :, :] = tmp_slice
        elif axis_i==1:
            filtered_vol0[:, islice, :] = tmp_slice
        elif axis_i==2:
            filtered_vol0[:, :, islice] = tmp_slice

        # self.filtered_vol0[islice, :, :] = tmp_slice



    def filter_along_axis_chunk_worker(self, chunk_start_idx, chunk_size, kernel, axis_i):
        logging.debug(f"filter_along_axis_chunk_worker() with chunk_start_idx:{chunk_start_idx}, chunk_size:{chunk_size}, axis_i:{axis_i}")

        #logging.info(f"Collected shared arrays")

        for i in range(chunk_size):
            if self.calculation_interrupt:
                logging.info(f"Interrupting task with chunk_start_idx {chunk_start_idx}")
                break
        
            #Work slice-by-slice
            self.filter_along_axis_slice(chunk_start_idx + i , kernel, axis_i)
            self.__percent__ += 1

        return chunk_start_idx


    def filter_along_axis(self, kernel, axis_i):
        if not(axis_i==0 or axis_i==1 or axis_i==2):
            raise ValueError(f"Axis {axis_i} not valid")
        
        #global __percent__
        logging.debug(f"Filtering along axis {axis_i} with l={self.OF_LEVELS}, w={self.OF_WINDOW_SIZE}, and kernel length={kernel.size}")

        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000

        if 'threaded' in self.process_mode:
            #Use the parallel processing algorithm

            #Don't need to create shared memories as
            # self._data_vol and self._filtered_vol are already shared
            # with names "data_vol_sm" and "filtered_vol_sm"
            
            axis_dim = self._data_vol.shape[axis_i]
            logging.info(f"axis_dim:{axis_dim}")
            logging.info(f"self.max_number_of_processes:{self.max_number_of_processes}")

            #chunk_size = axis_dim//self.max_number_of_processes
            # last slices, leave last process for last processing
            chunk_size = axis_dim//(self.max_number_of_processes)
            n_remain_slices = axis_dim % self.max_number_of_processes
            logging.info(f"n_remain_slices:{n_remain_slices}")

            #Arguments for PoolExecutor
            chunk_start_indexes = [i*chunk_size for i in range(self.max_number_of_processes)]
            chunk_sizes = [chunk_size]*(self.max_number_of_processes)
            kernels = [kernel]*self.max_number_of_processes
            axis_i_s = [axis_i]*self.max_number_of_processes
            if n_remain_slices>0: #last slices
                chunk_start_indexes.append(self.max_number_of_processes*chunk_size )
                chunk_sizes.append(n_remain_slices)
                kernels.append(kernel)
                axis_i_s.append(axis_i)

            logging.info(f"chunk_indexes:{chunk_start_indexes}")
            logging.info(f"chunk_sizes:{chunk_sizes}")
            #logging.info(f"kernels:{kernels}")
            logging.info(f"axis_i_s:{axis_i_s}")

            logging.info("Starting threads")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_number_of_processes) as executor:
                
                
                # futures = [executor.submit(self.filter_along_axis_chunk_worker,
                #                             chunk_indexes,
                #                             chunk_sizes,
                #                             chunk_offsets,
                #                             kernels,
                #                             axis_i_s) for i in range(self.max_number_of_processes)]

                futures=[]
                for args0 in zip(chunk_start_indexes,chunk_sizes,kernels,axis_i_s):
                    exec0=executor.submit(self.filter_along_axis_chunk_worker, *args0)
                    futures.append(exec0)

                done, not_done = concurrent.futures.wait(futures, timeout=0) #returns result immediately
                try:
                    while not_done:
                        freshly_done, not_done= concurrent.futures.wait(futures, timeout=3)
                        done |= freshly_done #This line is probably not needed
                except KeyboardInterrupt:
                    logging.error("**** EXCEPTION KeyboardInterrupt****. Cancelling other tasks.")
                    self.calculation_interrupt=True
                    # only futures that are not done will prevent exiting
                    for future in not_done:
                        # cancel() returns False if it's already done or currently running,
                        # and True if was able to cancel it; we don't need that return value
                        _ = future.cancel()
                    # wait for running futures that the above for loop couldn't cancel (note timeout)
                    _ = concurrent.futures.wait(not_done, timeout=None)

        elif 'sequential' in self.process_mode:
            #Sequential
            logging.info("Running sequentially")
            axis_dim = self._data_vol.shape[axis_i]
            chunk_size = axis_dim

            self.filter_along_axis_chunk_worker(chunk_start_idx=0, chunk_size=chunk_size,
                                                                 kernel=kernel, axis_i=axis_i)
        else:
            raise ValueError(f"Invaid self.process_mode:{ self.process_mode}")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
            logging.debug(f"Min OF val: {min_OF}")
            logging.debug(f"Max OF val: {max_OF}")

    def run_OF_MP(self):
        logging.debug("run_OF_MP()")

        from multiprocessing import shared_memory
        from multiprocessing import resource_tracker
        import subprocess

        #Run in multiprocessing mode
        #Along all axis

        in_arr_name = "sm_in_42"
        out_arr_name = "sm_out_42"
        progress_arr_name = "sm_progress_42"

        try:
            sm_in = shared_memory.SharedMemory(
                create=True,
                size=self._data_vol.nbytes,
                name= in_arr_name)
            logging.debug("sm_in created")

            in_array_sm= np.ndarray(shape=self.data_shape,
                dtype=self.data_type,
                buffer=sm_in.buf)
            in_array_sm[...] = self._data_vol[...]

            sm_out = shared_memory.SharedMemory(
                create=True,
                size=self._data_vol.nbytes, #same size as inarray
                name=out_arr_name)
            logging.debug("sm_out created")
            outarray_sm=np.ndarray(shape=self.data_shape,
                dtype=self.data_type,
                buffer=sm_out.buf)
            outarray_sm.fill(0)

            #Create shared memory for checking progress in parallel process(es)
            #Progress in each axis
            self.mp_progress_np=np.zeros(3, dtype=np.uint32)
            sm_progress = shared_memory.SharedMemory(
                create=True,
                size=self.mp_progress_np.nbytes,
                name= progress_arr_name)
            self.mp_progress_np = np.ndarray( (3), dtype=np.uint32, buffer=sm_progress.buf)
            logging.debug("sm_progress created")

            logging.debug("Shared memories created")

            try:
                resource_tracker.unregister(sm_in._name,"shared_memory")
                resource_tracker.unregister(sm_out._name,"shared_memory")
                resource_tracker.unregister(sm_progress._name,"shared_memory")
                #Need to register back before unlinking to delete remains
                logging.debug("Shared memories unregistered from resource tracker")
            except:
                logging.info("Failed to unregister shared memories. Maybe this is windows")
            
            #Continue

            logging.debug("Setting up _flowdenoising_subprocessMP.py")
            thisdir = pathlib.Path(__file__).parent
            py_torun = thisdir / "_flowdenoising_subprocessMP.py"
            logging.debug(f"torun:{str(py_torun)}")
            #define commands and parameters here
            cmds = ["python", py_torun,
                    "--in_data_sh_name", in_arr_name,
                    "--out_data_sh_name", out_arr_name,
                    "--dtype_name", str(self.data_type),
                    "--dshape_list", *[str(i) for i in self.data_shape], #unfolds data_shape, TODO: check it works
                    "--levels", str(self.OF_LEVELS),
                    "--winsize", str(self.OF_WINDOW_SIZE),
                    "--ksigma_list", *[str(i) for i in self.SIGMA],
                    "--iters", str(self.OF_ITERS),
                    "--number_of_processes", str(self.max_number_of_processes),
                    "--verbosity", str(self.verbosity),
                    "--progress_sh_name", progress_arr_name,
                    ]

            if self.use_OF:
                cmds.append("--useOF")

            if self.bComputeFlowWithPreviousFlow:
                cmds.append("--bComputeFlowWithPreviousFlow")

            #launchSubProcessAndPrintToConsole(cmds)
            #RUN CALCULATION IN A SEPaRATE PROCESS
            process = subprocess.run(args=cmds, capture_output=True, text=True)
            logging.debug("subprocess result stdout: ")
            logging.debug(str(process.stdout))

            logging.debug("subprocess result stderr:")
            logging.debug(str(process.stderr))

            # print("After subprocess, collect result from shared memory")
            # print(str(outarray_sm))

            self._filtered_vol[...]=outarray_sm[...] #Copy result
            #self.__percent__ += self.data_shape[axis_i]

        except Exception as e:
            logging.error("Some error occurred")
            logging.error(str(e))
        finally:
            try:
                sm_in.close()
                sm_out.close()
                sm_progress.close()

                try:
                    resource_tracker.register(sm_in._name,"shared_memory")
                    resource_tracker.register(sm_out._name,"shared_memory")
                    resource_tracker.register(sm_progress._name,"shared_memory")
                except:
                    pass

                logging.debug("sm_in.unlink()")
                sm_in.unlink()
                logging.debug("sm_out.unlink()")
                sm_out.unlink()
                logging.debug("sm_progress.unlink()")
                sm_progress.unlink()
            except Exception as e:
                logging.error("Error occured when trying to close and unlink shared memories")
                logging.error(str(e))


    def feedback_periodic(self,stopEv: threading.Event):
        #Can use this thread to cancel calculation in case of keyboard interrrupt
        time_0 = time.perf_counter()
        n_iterations = int(np.sum(np.array(self._data_vol.shape)))

        logging.debug(f"self.calculation_interrupt:{self.calculation_interrupt}")
        logging.debug(f"stopEv.is_set():{stopEv.is_set()}")
        while not stopEv.is_set() or not self.calculation_interrupt:
            current_time = time.perf_counter()
            if self.timeout_mins > 0:
                if (current_time - time_0) > (60 * self.timeout_mins):
                    logging.debug("Timeout to complete, stopping calculation")
                    stopEv.set()
                    self.calculation_interrupt=True

            if not self.mp_progress_np is None:
                self.__percent__= np.sum(self.mp_progress_np)

            logging.info(f"{self.__percent__}/{n_iterations} completed")

            time.sleep(1)
        logging.debug("feedback_periodic thread stopped.")


    def runOpticalFlow(self, data_vol:np.ndarray):

        """
        runs opticalFlow filter on the numpy volume provided
        """

        vol_size_bytes = data_vol.dtype.itemsize * data_vol.size
        logging.info(f"shape of the input volume (Z, Y, X) = {data_vol.shape}")
        self.data_shape=data_vol.shape

        logging.info(f"type of the volume = {data_vol.dtype}")
        self.data_type= data_vol.dtype
        
        logging.info(f"vol requires {vol_size_bytes/(1024*1024):.1f} MB")
        logging.info(f"data max = {data_vol.max()}")
        logging.info(f"data min = {data_vol.min()}")

        self.vol_mean = data_vol.mean()
        logging.info(f"Input vol average = {self.vol_mean}")

        # if self.bComputeFlowWithPreviousFlow:
        #     self.get_flow = self.get_flow_without_prev_flow
        #     logging.info("Not reusing adjacent OF fields as predictions")
        # else:
        #     self.get_flow = self.get_flow_with_prev_flow
        #     logging.info("Using adjacent OF fields as predictions")

        # sigma=self.SIGMA
        # logging.info(f"sigma={tuple(sigma)}")

        self.updateKernels()
        # kernels = [None]*3
        # kernels[0] = self.get_gaussian_kernel(sigma[0])
        # kernels[1] = self.get_gaussian_kernel(sigma[1])
        # kernels[2] = self.get_gaussian_kernel(sigma[2])
        # logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*kernels]]}")


        self._data_vol=np.array(data_vol) #Simply copies

        self._filtered_vol = np.zeros_like(data_vol)

        if __debug__:
            logging.info(f"Filtering ...")
            time_0 = time.perf_counter()
        try:
            #RUN THE FILTER
            #self.do_filter(self._kernels)
            self.do_filter()

            if __debug__:
                #time_1 = time.perf_counter()        
                time_1 = time.perf_counter()        
                logging.info(f"Volume filtered in {time_1 - time_0} seconds")

            result = np.array(self._filtered_vol) #makes a copy before returning
            return result
        
        except Exception as e:
            logging.error("Something wrong happened.", str(e))
        finally:
            self.calculation_interrupt=True
            stopEv.set()

# def launchSubProcessAndPrintToConsole(cmds):
#     #Start module
#     print("Starting subprocess_module using Popen")
#     import subprocess
#     #From https://stacktuts.com/how-to-suppress-or-capture-the-output-of-subprocess-run-in-python
#     process = subprocess.Popen(cmds,
#                                 stdout=subprocess.PIPE,
#                                 text=True)
#     while True:
#         output = process.stdout.readline()
#         if output == '' and process.poll() is not None:
#             break
#         if output:
#             print(output.strip())


# def show_memory_usage(msg=''):
#     logging.info(f"{psutil.Process(os.getpid()).memory_info().rss/(1024*1024):.1f} MB used in process {os.getpid()} {msg}")

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

def parseArgs():
    OF_LEVELS = 3
    OF_WINDOW_SIZE = 5
    SIGMA = 2.0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input", type=int_or_str,
                        help="Input a MRC-file or a multi-image TIFF-file",
                        default="./volume.mrc")
    parser.add_argument("-o", "--output", type=int_or_str,
                        help="Output a MRC-file or a multi-image TIFF-file",
                        default="./denoised_volume.mrc")
    parser.add_argument("-s", "--sigma", nargs="+",
                        help="Gaussian sigma for each dimension in the order (Z, Y, X)",
                        default=(SIGMA, SIGMA, SIGMA))
    parser.add_argument("-l", "--levels", type=int_or_str,
                        help="Number of levels of the Gaussian pyramid used by the optical flow estimator",
                        default=OF_LEVELS)
    parser.add_argument("-w", "--winsize", type=int_or_str,
                        help="Size of the window used by the optical flow estimator",
                        default=OF_WINDOW_SIZE)
    parser.add_argument("-v", "--verbosity", type=int_or_str,
                        help="Verbosity level", default=0)
    parser.add_argument("-n", "--no_OF", action="store_true", help="Disable optical flow compensation")
    
    number_of_PUs = multiprocessing.cpu_count()
    parser.add_argument("-p", "--number_of_processes", type=int_or_str,
                        help="Maximum number of processes",
                        default=number_of_PUs)
    parser.add_argument("--recompute_flow", action="store_true", help="Disable the use of adjacent optical flow fields")
    parser.add_argument("--timeout", type=int, help="Timeout after x mins. Set to -1 for no timeout. Default 30 mins", default=30)

    parser.add_argument("--procmode", choices=['threaded', 'sequential','multiproc'],
                        default='threaded')

    return parser

def main():
    parser = parseArgs()
    parser.description = __doc__
    args = parser.parse_args()

    if args.verbosity == 2:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.DEBUG)
        logging.info("Verbosity level = 2")
    elif args.verbosity == 1:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.INFO)
        logging.info("Verbosity level = 1")        
    else:
        logging.basicConfig(format=LOGGING_FORMAT, level=logging.CRITICAL)
    
    if __debug__:
        logging.info(f"reading \"{args.input}\"")
        time_0 = time.perf_counter()

    logging.debug(f"input = {args.input}")

    if "mrc" in args.input.split('.')[-1].lower():
        logging.info(f"Input file is MRC")
        # if args.memory_map:
        #     logging.info(f"Using memory mapping")
        #     vol_MRC = rc = mrcfile.mmap(args.input, mode='r+')
        #     data_vol = vol_MRC.data
        # else:
        with  mrcfile.open(args.input, mode="r") as vol_MRC:     
            data_vol = vol_MRC.data
    else:
        #data_vol = skimage.io.imread(args.input, plugin="tifffile").astype(np.float32)
        data_vol = skimage.io.imread(args.input)

    if __debug__:
        time_1 = time.perf_counter()
        logging.info(f"read \"{args.input}\" in {time_1 - time_0} seconds")
    
    logging.info(f"args : {args}")
    sigma = [float(i) for i in args.sigma]
    
    #Setup filter
    filter0 = cFlowDenoiser(
        sigma=sigma,
        levels=args.levels,
        winsize=args.winsize,
        max_number_of_processes=args.number_of_processes,
        bComputeFlowWithPreviousFlow= not args.recompute_flow,
        timeout_mins = args.timeout,
        use_OF= not args.no_OF,
        process_mode=args.procmode,
        verbosity=args.verbosity
        )
    #filter0 = cFlowDenoiser()
    
    # *** run the filter
    filtered_vol0= filter0.runOpticalFlow(data_vol)

    #print results statistics
    logging.info(f"{args.output} type = {filtered_vol0.dtype}")
    logging.info(f"{args.output} max = {filtered_vol0.max()}")
    logging.info(f"{args.output} min = {filtered_vol0.min()}")
    logging.info(f"{args.output} average = {filtered_vol0.mean()}")

    if __debug__:
        logging.info(f"writting \"{args.output}\"")
        time_0 = time.perf_counter()

    logging.debug(f"output = {args.output}")

    if "mrc" in args.output.split('.')[-1].lower():
        logging.info(f"Writting MRC file")
        with mrcfile.new(args.output, overwrite=True) as mrc:
            #mrc.set_data(_filtered_vol.astype(np.float32))
            mrc.set_data(filtered_vol0)
            #mrc.data
    else:
        logging.debug(f"Writting TIFF file")
        #skimage.io.imsave(args.output, _filtered_vol.astype(np.float32), plugin="tifffile")
        skimage.io.imsave(args.output, filtered_vol0, plugin="tifffile")

    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")

if __name__ == "__main__":
    main()
