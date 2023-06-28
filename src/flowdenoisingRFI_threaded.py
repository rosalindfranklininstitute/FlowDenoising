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
2023 Significantly modified version by Luis Perdigao @RFI

Code put in a class so that it potentially run in a notebook or integrated in other programs


"""

import logging
import os
import numpy as np
import cv2
import scipy.ndimage
import time
# import imageio
# import tifffile
import skimage.io
import mrcfile
import argparse
import threading

# import concurrent
import multiprocessing
from multiprocessing import shared_memory, Value
#from concurrent.futures.process import ProcessPoolExecutor
import random
#import _thread
#import sys
import concurrent.futures

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
        disable_OF_compensation=True,
        enable_mem_map = True,
        max_number_of_processes=None,
        bComputeFlowWithPreviousFlow=True,
        timeout_mins = 30,
        use_OF=True,
        do_sequentially=False,
        ):

        print(f"levels:{levels}")
        print(f"winsize:{winsize}")
        self.SIGMA= sigma
        self.OF_LEVELS= levels
        self.OF_WINDOW_SIZE= winsize
        self.disable_OF_compensation=disable_OF_compensation
        self.enable_mem_map= enable_mem_map
        
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

        self.do_sequentially=do_sequentially
        


    @staticmethod
    def get_gaussian_kernel(sigma=1):
        logging.info(f"Computing gaussian kernel with sigma={sigma}")
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

    def do_filter(self,kernels):

        self.__percent__=0

        self.calculation_interrupt=False
 
        stopEv.clear()
        feddback_thread = threading.Thread(target=self.feedback_periodic, args=(stopEv,))
        feddback_thread.daemon = True # To obey CTRL+C interruption.
        #This also means that the thread is killed when the program exits
        feddback_thread.start()

        logging.info("Filtering along Z")
        self.filter_along_axis(kernels[0],  axis_i=0)
        self._data_vol[...] = self._filtered_vol[...]

        logging.info("Filtering along Y")
        self.filter_along_axis(kernels[0], axis_i=1)
        self._data_vol[...] = self._filtered_vol[...]

        logging.info("Filtering along X")
        self.filter_along_axis(kernels[0], axis_i=2)
        self._data_vol[...] = self._filtered_vol[...]

        #When filtering is complete it continues here
        #stop feedback_thread.
        # There no stop() function to do that, so Event() is used
        stopEv.set()

    def filter_along_axis_slice(self,data_vol0,filtered_vol0, islice, kernel, axis_i):
        logging.debug(f"filter_along_axis_slice() with islice:{islice},  axis_i:{axis_i}")

        assert kernel.size % 2 != 0 # kernel.size must be odd

        ks2 = kernel.size//2

        #logging.info("Transposing if needed")
        if axis_i==0:
            data_vol_transp=data_vol0
        elif axis_i==1:
            data_vol_transp=np.transpose(data_vol0,axes=(1,0,2))
        elif axis_i==2:
            data_vol_transp=np.transpose(data_vol0,axes=(2,0,1))
                            
        tmp_slice = np.zeros_like(data_vol_transp[islice, :, :]).astype(np.float32)

        if self.use_OF:
            prev_flow = np.zeros(shape=(data_vol_transp.shape[1], data_vol_transp.shape[2], 2), dtype=np.float32)

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
            prev_flow = np.zeros(shape=(data_vol_transp.shape[1], data_vol_transp.shape[2], 2), dtype=np.float32)
            
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


        #logging.info("Restoring orienation")
        if axis_i==0:
            filtered_vol0[islice, :, :] = tmp_slice
        elif axis_i==1:
            filtered_vol0[:, islice, :] = tmp_slice
        elif axis_i==2:
            filtered_vol0[:, :, islice] = tmp_slice

        # self.filtered_vol0[islice, :, :] = tmp_slice
        self.__percent__ += 1

    def filter_along_axis_chunk_worker(self, chunk_start_idx, chunk_size, kernel, axis_i):
        logging.info(f"filter_along_axis_chunk_worker() with chunk_start_idx:{chunk_start_idx}, chunk_size:{chunk_size}, axis_i:{axis_i}")

        logging.info(f"Collected shared arrays")

        for i in range(chunk_size):
            if self.calculation_interrupt:
                logging.info(f"Interrupting task with chunk_start_idx {chunk_start_idx}")
                break
        
            #Work slice-by-slice
            self.filter_along_axis_slice(self._data_vol,self._filtered_vol, chunk_start_idx + i , kernel, axis_i)

        return chunk_start_idx


    def filter_along_axis(self, kernel, axis_i):
        if not(axis_i==0 or axis_i==1 or axis_i==2):
            raise ValueError(f"Axis {axis_i} not valid")
        
        #global __percent__
        logging.info(f"Filtering along axis {axis_i} with l={self.OF_LEVELS}, w={self.OF_WINDOW_SIZE}, and kernel length={kernel.size}")

        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000

        if not self.do_sequentially:
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
                    print("**** EXCEPTION KeyboardInterrupt****. Cancelling other tasks.")
                    self.calculation_interrupt=True
                    # only futures that are not done will prevent exiting
                    for future in not_done:
                        # cancel() returns False if it's already done or currently running,
                        # and True if was able to cancel it; we don't need that return value
                        _ = future.cancel()
                    # wait for running futures that the above for loop couldn't cancel (note timeout)
                    _ = concurrent.futures.wait(not_done, timeout=None)

        else:
            #Sequential
            print("Running sequentially")
            axis_dim = self._data_vol.shape[axis_i]
            chunk_size = axis_dim

            self.filter_along_axis_chunk_worker(chunk_start_idx=0, chunk_size=chunk_size,
                                                                 kernel=kernel, axis_i=axis_i)
        

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
            logging.debug(f"Min OF val: {min_OF}")
            logging.debug(f"Max OF val: {max_OF}")


    def feedback_periodic(self,stopEv: threading.Event):
        #Can use this thread to cancel calculation in case of keyboard interrrupt
        time_0 = time.perf_counter()
        n_iterations = int(np.sum(np.array(self._data_vol.shape)))

        print(f"self.calculation_interrupt:{self.calculation_interrupt}")
        print(f"stopEv.is_set():{stopEv.is_set()}")
        while not stopEv.is_set() or not self.calculation_interrupt:
            current_time = time.perf_counter()
            if self.timeout_mins > 0:
                if (current_time - time_0) > (60 * self.timeout_mins):
                    logging.info("Timeout to complete, stopping calcualtion")
                    stopEv.set()
                    self.calculation_interrupt=True

            logging.info(f"{self.__percent__}/{n_iterations} completed")
            time.sleep(1)
        logging.info("feedback_periodic thread stopped.")


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

        sigma=self.SIGMA
        logging.info(f"sigma={tuple(sigma)}")

        kernels = [None]*3
        kernels[0] = self.get_gaussian_kernel(sigma[0])
        kernels[1] = self.get_gaussian_kernel(sigma[1])
        kernels[2] = self.get_gaussian_kernel(sigma[2])
        logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*kernels]]}")


        self._data_vol=np.array(data_vol) #Simply copies

        self._filtered_vol = np.zeros_like(data_vol)

        if __debug__:
            logging.info(f"Filtering ...")
            time_0 = time.perf_counter()
        try:
            #RUN THE FILTER
            self.do_filter(kernels)

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
    #parser.add_argument("-t", "--transpose", nargs="+",
    #                    help="Transpose pattern (see https://numpy.org/doc/stable/reference/generated/numpy.transpose.html, by default the 3D volume in not transposed)",
    #                    default=(0, 1, 2))
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
    parser.add_argument("-m", "--memory_map",
                        action="store_true",
                        help="Enable memory-mapping (see https://mrcfile.readthedocs.io/en/stable/usage_guide.html#dealing-with-large-files, only for MRC files)")
    
    number_of_PUs = multiprocessing.cpu_count()
    parser.add_argument("-p", "--number_of_processes", type=int_or_str,
                        help="Maximum number of processes",
                        default=number_of_PUs)
    parser.add_argument("--recompute_flow", action="store_true", help="Disable the use of adjacent optical flow fields")
    parser.add_argument("--timeout", type=int, help="Timeout after x mins. Set to -1 for no timeout. Default 30 mins", default=30)

    #TODO: add do_sequentially argument

    return parser

if __name__ == "__main__":

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
    
    sigma = [float(i) for i in args.sigma]

    if __debug__:
        logging.info(f"reading \"{args.input}\"")
        time_0 = time.perf_counter()

    logging.debug(f"input = {args.input}")

    is_MRC_input = ( args.input.split('.')[-1] == "MRC" or args.input.split('.')[-1] == "mrc" )
    if is_MRC_input:
        if args.memory_map:
            logging.info(f"Using memory mapping")
            vol_MRC = rc = mrcfile.mmap(args.input, mode='r+')
        else:
            vol_MRC = mrcfile.open(args.input, mode="r+")
        data_vol = vol_MRC.data
    else:
        data_vol = skimage.io.imread(args.input, plugin="tifffile").astype(np.float32)
    vol_size = data_vol.dtype.itemsize * data_vol.size
    
    # logging.info(f"shape of the input volume (Z, Y, X) = {data_vol.shape}")
    # logging.info(f"type of the volume = {data_vol.dtype}")
    # logging.info(f"vol requires {vol_size/(1024*1024):.1f} MB")
    # logging.info(f"{args.input} max = {data_vol.max()}")
    # logging.info(f"{args.input} min = {data_vol.min()}")
    # vol_mean = data_vol.mean()
    # logging.info(f"Input vol average = {vol_mean}")

    if __debug__:
        time_1 = time.perf_counter()
        logging.info(f"read \"{args.input}\" in {time_1 - time_0} seconds")

    #Setup and run filter
    filter0 = cFlowDenoiser(
        sigma=sigma,
        levels=args.levels,
        winsize=args.winsize,
        disable_OF_compensation=args.no_OF,
        enable_mem_map = args.memory_map,
        max_number_of_processes=args.number_of_processes,
        bComputeFlowWithPreviousFlow=args.recompute_flow,
        timeout_mins = args.timeout,
        )
    #TODO: add do_sequentially argument
    
    #run the filter
    _filtered_vol= filter0.runOpticalFlow(data_vol)

    logging.info(f"{args.output} type = {_filtered_vol.dtype}")
    logging.info(f"{args.output} max = {_filtered_vol.max()}")
    logging.info(f"{args.output} min = {_filtered_vol.min()}")
    logging.info(f"{args.output} average = {_filtered_vol.mean()}")

    # #Starts filtering
    # if args.no_OF:
    #     no_OF_filter(kernels) #Filters without optical flow
    # else:
    #     OF_filter(kernels, levels, winsize) #Filters with optical flow

    #When filtering is complete it continues here
    #stop feedback_thread.
    # There no stop() function to do that, so Event() is used
    # stopEv.set()

    # if __debug__:
    #     #time_1 = time.perf_counter()        
    #     time_1 = time.perf_counter()        
    #     logging.info(f"Volume filtered in {time_1 - time_0} seconds")

    #filtered_vol = np.transpose(filtered_vol, transpose_pattern)
    logging.info(f"{args.output} type = {_filtered_vol.dtype}")
    logging.info(f"{args.output} max = {_filtered_vol.max()}")
    logging.info(f"{args.output} min = {_filtered_vol.min()}")
    logging.info(f"{args.output} average = {_filtered_vol.mean()}")
    
    if __debug__:
        logging.info(f"writting \"{args.output}\"")
        time_0 = time.perf_counter()

    logging.debug(f"output = {args.output}")

    MRC_output = ( args.output.split('.')[-1] == "MRC" or args.output.split('.')[-1] == "mrc" )

    if MRC_output:
        logging.debug(f"Writting MRC file")
        with mrcfile.new(args.output, overwrite=True) as mrc:
            mrc.set_data(_filtered_vol.astype(np.float32))
            #mrc.data
    else:
        logging.debug(f"Writting TIFF file")
        skimage.io.imsave(args.output, _filtered_vol.astype(np.float32), plugin="tifffile")

    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")


