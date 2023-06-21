#!/usr/bin/env python
'''3D Gaussian filtering controlled by the optical flow.
'''

# "flowdenoising.py" is part of "https://github.com/microscopy-processing/FlowDenoising", authored by:
#
# * J.J. Fernández (CSIC).
# * V. González-Ruiz (UAL).
#
# Please, refer to the LICENSE.txt to know the terms of usage of this software.

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
from concurrent.futures.process import ProcessPoolExecutor
import random
import _thread
import sys

LOGGING_FORMAT = "[%(asctime)s] (%(levelname)s) %(message)s"

__percent__ = Value('f', 0)

class cFlowDenoiser():
    OFCA_EXTENSION_MODE = cv2.BORDER_REPLICATE
    OF_LEVELS = 3
    OF_WINDOW_SIZE = 5
    OF_ITERS = 3
    OF_POLY_N = 5
    OF_POLY_SIGMA = 1.2
    #SIGMA = 2.0

    data_vol=None
    _filtered_vol=None

    def __init__(self, *,
        sigma=(2.0,2.0,2.0),
        levels=3,
        winsize=5,
        disable_OF_compensation=True,
        enable_mem_map = True,
        max_number_of_processes=None,
        bComputeFlowWithPreviousFlow=True,
        timeout_mins = 30,
        ):

        self.sigma= sigma
        self.OF_LEVELS= levels,
        self.OF_WINDOW_SIZE= winsize,
        self.disable_OF_compensation=disable_OF_compensation,
        self.enable_mem_map= enable_mem_map
        
        number_of_PUs = multiprocessing.cpu_count()
        logging.info(f"Number of processing units: {number_of_PUs}")

        self.max_number_of_processes=max_number_of_processes
        if max_number_of_processes is None:

            self.max_number_of_processes = number_of_PUs
            
        self.bComputeFlowWithPreviousFlow = bComputeFlowWithPreviousFlow
        self.timeout_mins = timeout_mins


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

    def get_flow_with_prev_flow(self, reference, target, l, w, prev_flow=None):
        if l is None:
            l= self.OF_LEVELS
        if w is None:
            w=self.OF_WINDOW_SIZE

        if __debug__:
            time_0 = time.perf_counter()
        flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=l, winsize=w, iterations=self.OF_ITERS, poly_n=self.OF_POLY_N, poly_sigma=self.OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        #flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=0)
        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
        return flow
    
    def get_flow_without_prev_flow(self,reference, target, l=OF_LEVELS, w=OF_WINDOW_SIZE, prev_flow=None):
        if __debug__:
            time_0 = time.perf_counter()
        #flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=prev_flow, pyr_scale=0.5, levels=l, winsize=w, iterations=OF_ITERS, poly_n=OF_POLY_N, poly_sigma=OF_POLY_SIGMA, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        flow = cv2.calcOpticalFlowFarneback(prev=target, next=reference, flow=None, pyr_scale=0.5, levels=l, winsize=w, iterations=self.OF_ITERS, poly_n=self.OF_POLY_N, poly_sigma=self.OF_POLY_SIGMA, flags=0)
        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"OF computed in {1000*(time_1 - time_0):4.3f} ms, max_X={np.max(flow[0]):+3.2f}, min_X={np.min(flow[0]):+3.2f}, max_Y={np.max(flow[1]):+3.2f}, min_Y={np.min(flow[1]):+3.2f}")
        return flow

    def OF_filter_along_Z_slice(self,z, kernel):
        data_vol=self.data_vol
        ks2 = kernel.size//2
        tmp_slice = np.zeros_like(data_vol[z, :, :]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(data_vol.shape[1], data_vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 - 1, -1, -1):
            flow = self.get_flow(data_vol[(z + i - ks2) % data_vol.shape[0], :, :],
                            data_vol[z, :, :], self.levels, self.winsize, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(data_vol[(z + i - ks2) % data_vol.shape[0], :, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += data_vol[z, :, :] * kernel[ks2]
        prev_flow = np.zeros(shape=(data_vol.shape[1], data_vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = self.get_flow(data_vol[(z + i - ks2) % data_vol.shape[0], :, :],
                            data_vol[z, :, :], self.levels, self.winsize, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(data_vol[(z + i - ks2) % data_vol.shape[0], :, :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        self._filtered_vol[z, :, :] = tmp_slice
        __percent__.value += 1

    def OF_filter_along_Y_slice(self,y, kernel):
        data_vol=self.data_vol
        ks2 = kernel.size//2
        tmp_slice = np.zeros_like(data_vol[:, y, :]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(data_vol.shape[0], data_vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 - 1, -1, -1):
            flow = self.get_flow(data_vol[:, (y + i - ks2) % data_vol.shape[1], :],
                            data_vol[:, y, :], self.levels, self.winsize, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(data_vol[:, (y + i - ks2) % data_vol.shape[1], :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += data_vol[:, y, :] * kernel[ks2]
        prev_flow = np.zeros(shape=(data_vol.shape[0], data_vol.shape[2], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = self.get_flow(data_vol[:, (y + i - ks2) % data_vol.shape[1], :],
                            data_vol[:, y, :], self.levels, self.winsize, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(data_vol[:, (y + i - ks2) % data_vol.shape[1], :], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        self._filtered_vol[:, y, :] = tmp_slice
        __percent__.value += 1

    def OF_filter_along_X_slice(self,x, kernel):
        data_vol=self.data_vol
        ks2 = kernel.size//2
        tmp_slice = np.zeros_like(data_vol[:, :, x]).astype(np.float32)
        assert kernel.size % 2 != 0 # kernel.size must be odd
        prev_flow = np.zeros(shape=(data_vol.shape[0], data_vol.shape[1], 2), dtype=np.float32)
        for i in range(ks2 - 1, -1, -1):
            flow = self.get_flow(data_vol[:, :, (x + i - ks2) % data_vol.shape[2]],
                            data_vol[:, :, x], self.OF_LEVELS, self.OF_WINDOW_SIZE, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(data_vol[:, :, (x + i - ks2) % data_vol.shape[2]], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        tmp_slice += data_vol[:, :, x] * kernel[ks2]
        prev_flow = np.zeros(shape=(data_vol.shape[0], data_vol.shape[1], 2), dtype=np.float32)
        for i in range(ks2 + 1, kernel.size):
            flow = self.get_flow(data_vol[:, :, (x + i - ks2) % data_vol.shape[2]],
                            data_vol[:, :, x], self.OF_LEVELS, self.OF_WINDOW_SIZE, prev_flow)
            prev_flow = flow
            OF_compensated_slice = self.warp_slice(data_vol[:, :, (x + i - ks2) % data_vol.shape[2]], flow)
            tmp_slice += OF_compensated_slice * kernel[i]
        self._filtered_vol[:, :, x] = tmp_slice
        __percent__.value += 1

    def OF_filter_along_Z_chunk(self,chunk_index, chunk_size, chunk_offset, kernel):
        for z in range(chunk_size):
            self.OF_filter_along_Z_slice(chunk_index*chunk_size + z + chunk_offset, kernel)
        return chunk_index

    def OF_filter_along_Y_chunk(self,chunk_index, chunk_size, chunk_offset, kernel):
        for y in range(chunk_size):
            self.OF_filter_along_Y_slice(chunk_index*chunk_size + y + chunk_offset, kernel)
        return chunk_index

    def OF_filter_along_X_chunk(self,chunk_index, chunk_size, chunk_offset, kernel):
        for x in range(chunk_size):
            self.OF_filter_along_X_slice(chunk_index*chunk_size + x + chunk_offset, kernel)
        return chunk_index
        
    def OF_filter_along_Z(self,kernel, l, w):
        data_vol=self.data_vol
        global __percent__
        logging.info(f"Filtering along Z with l={l}, w={w}, and kernel length={kernel.size}")

        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000

        Z_dim = data_vol.shape[0]
        chunk_size = Z_dim//self.max_number_of_processes
        #for i in range(number_of_processes):
        #    OF_filter_along_Z_chunk(i, padded_vol, kernel)
        chunk_indexes = [i for i in range(self.max_number_of_processes)]
        chunk_sizes = [chunk_size]*self.max_number_of_processes
        chunk_offsets = [0]*self.max_number_of_processes
        kernels = [kernel]*self.max_number_of_processes
        with ProcessPoolExecutor(max_workers=self.max_number_of_processes) as executor:
            for _ in executor.map(self.OF_filter_along_Z_chunk,
                                chunk_indexes,
                                chunk_sizes,
                                chunk_offsets,
                                kernels):
                logging.debug(f"PE #{_} has finished")
        remainding_slices = Z_dim % self.max_number_of_processes
        if remainding_slices > 0:
            chunk_indexes = [i for i in range(remainding_slices)]
            chunk_sizes = [1]*remainding_slices
            chunk_offsets = [chunk_size*self.max_number_of_processes]*remainding_slices
            kernels = [kernel]*remainding_slices
            with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
                for _ in executor.map(self.OF_filter_along_Z_chunk,
                                    chunk_indexes,
                                    chunk_sizes,
                                    chunk_offsets,
                                    kernels):
                    logging.debug(f"PU #{_} finished")
        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")
            logging.debug(f"Min OF val: {min_OF}")
            logging.debug(f"Max OF val: {max_OF}")

    def OF_filter_along_Y(self,kernel, l, w):
        data_vol=self.data_vol
        global __percent__
        logging.info(f"Filtering along Y with l={l}, w={w}, and kernel length={kernel.size}")
        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000

        Y_dim = data_vol.shape[1]
        chunk_size = Y_dim//self.max_number_of_processes
        #for i in range(number_of_processes):
        #    OF_filter_along_Y_chunk(i, padded_vol, kernel)
        chunk_indexes = [i for i in range(self.max_number_of_processes)]
        chunk_sizes = [chunk_size]*self.max_number_of_processes
        chunk_offsets = [0]*self.max_number_of_processes
        kernels = [kernel]*self.max_number_of_processes
        with ProcessPoolExecutor(max_workers=self.max_number_of_processes) as executor:
            for _ in executor.map(self.OF_filter_along_Y_chunk,
                                chunk_indexes,
                                chunk_sizes,
                                chunk_offsets,
                                kernels):
                logging.debug(f"PE #{_} has finished")
        remainding_slices = Y_dim % self.max_number_of_processes
        if remainding_slices > 0:
            chunk_indexes = [i for i in range(remainding_slices)]
            chunk_sizes = [1]*remainding_slices
            chunk_offsets = [chunk_size*self.max_number_of_processes]*remainding_slices
            kernels = [kernel]*remainding_slices
            with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
                for _ in executor.map(self.OF_filter_along_Y_chunk,
                                    chunk_indexes,
                                    chunk_sizes,
                                    chunk_offsets,
                                    kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")
            logging.debug(f"Min OF val: {min_OF}")
            logging.debug(f"Max OF val: {max_OF}")

    def OF_filter_along_X(self,kernel, l, w):
        data_vol=self.data_vol
        global __percent__
        logging.info(f"Filtering along X with l={l}, w={w}, and kernel length={kernel.size}")
        if __debug__:
            time_0 = time.perf_counter()
            min_OF = 1000
            max_OF = -1000

        X_dim = data_vol.shape[2]
        chunk_size = X_dim//self.max_number_of_processes
        #for i in range(number_of_processes):
        #    OF_filter_along_X_chunk(i, padded_vol, kernel)
        chunk_indexes = [i for i in range(self.max_number_of_processes)]
        chunk_sizes = [chunk_size]*self.max_number_of_processes
        chunk_offsets = [0]*self.max_number_of_processes
        kernels = [kernel]*self.max_number_of_processes
        with ProcessPoolExecutor(max_workers=self.max_number_of_processes) as executor:
            for _ in executor.map(self.OF_filter_along_X_chunk,
                                chunk_indexes,
                                chunk_sizes,
                                chunk_offsets,
                                kernels):
                logging.debug(f"PE #{_} has finished")
        remainding_slices = X_dim % self.max_number_of_processes
        if remainding_slices > 0:
            chunk_indexes = [i for i in range(remainding_slices)]
            chunk_sizes = [1]*remainding_slices
            chunk_offsets = [chunk_size*self.max_number_of_processes]*remainding_slices
            kernels = [kernel]*remainding_slices
            with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
                for _ in executor.map(self.OF_filter_along_X_chunk,
                                    chunk_indexes,
                                    chunk_sizes,
                                    chunk_offsets,
                                    kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")

    def OF_filter(self,kernels, l, w):
        data_vol=self.data_vol
        self.OF_filter_along_Z(kernels[0], l, w)
        data_vol[...] = self._filtered_vol[...]
        self.OF_filter_along_Y(kernels[1], l, w)
        data_vol[...] = self._filtered_vol[...]
        self.OF_filter_along_X(kernels[2], l, w)

    def no_OF_filter_along_Z_slice(self,z, kernel):
        data_vol=self.data_vol
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(data_vol.shape[1], data_vol.shape[2]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += data_vol[(z + i - ks2) % data_vol.shape[0], :, :]*kernel[i]
        self._filtered_vol[z, :, :] = tmp_slice
        __percent__.value += 1

    def no_OF_filter_along_Y_slice(self,y, kernel):
        data_vol=self.data_vol
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(data_vol.shape[0], data_vol.shape[2]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += data_vol[:, (y + i - ks2) % data_vol.shape[1], :]*kernel[i]
        self._filtered_vol[:, y, :] = tmp_slice
        __percent__.value += 1

    def no_OF_filter_along_X_slice(self,x, kernel):
        data_vol=self.data_vol
        ks2 = kernel.size//2
        tmp_slice = np.zeros(shape=(data_vol.shape[0], data_vol.shape[1]), dtype=np.float32)
        for i in range(kernel.size):
            tmp_slice += data_vol[:, :, (x + i - ks2) % data_vol.shape[2]]*kernel[i]
        self._filtered_vol[:, :, x] = tmp_slice
        __percent__.value += 1

    def no_OF_filter_along_Z_chunk(self,chunk_index, chunk_size, chunk_offset, kernel):
        for z in range(chunk_size):
            self.no_OF_filter_along_Z_slice(chunk_index*chunk_size + z + chunk_offset, kernel)
        return chunk_index

    def no_OF_filter_along_Y_chunk(self,chunk_index, chunk_size, chunk_offset, kernel):
        for y in range(chunk_size):
            self.no_OF_filter_along_Y_slice(chunk_index*chunk_size + y + chunk_offset, kernel)
        return chunk_index

    def no_OF_filter_along_X_chunk(self,chunk_index, chunk_size, chunk_offset, kernel):
        for x in range(chunk_size):
            self.no_OF_filter_along_X_slice(chunk_index*chunk_size + x + chunk_offset, kernel)
        return chunk_index

    def no_OF_filter_along_Z(self,kernel):
        data_vol=self.data_vol
        logging.info(f"Filtering along Z with kernel length={kernel.size}")

        if __debug__:
            time_0 = time.perf_counter()

        Z_dim = data_vol.shape[0]
        chunk_size = Z_dim//self.max_number_of_processes
        #for i in range(number_of_processes):
        #    no_OF_filter_along_Z_chunk(i, kernel)
        chunk_indexes = [i for i in range(self.max_number_of_processes)]
        chunk_sizes = [chunk_size]*self.max_number_of_processes
        chunk_offsets = [0]*self.max_number_of_processes
        kernels = [kernel]*self.max_number_of_processes
        with ProcessPoolExecutor(max_workers=self.max_number_of_processes) as executor:
            for _ in executor.map(self.no_OF_filter_along_Z_chunk,
                                chunk_indexes,
                                chunk_sizes,
                                chunk_offsets,
                                kernels):
                logging.debug(f"PU #{_} finished")
        remainding_slices = Z_dim % self.max_number_of_processes
        if remainding_slices > 0:
            chunk_indexes = [i for i in range(remainding_slices)]
            chunk_sizes = [1]*remainding_slices
            chunk_offsets = [chunk_size*self.max_number_of_processes]*remainding_slices
            kernels = [kernel]*remainding_slices
            with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
                for _ in executor.map(self.no_OF_filter_along_Z_chunk,
                                    chunk_indexes,
                                    chunk_sizes,
                                    chunk_offsets,
                                    kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Z spent {time_1 - time_0} seconds")

    def no_OF_filter_along_Y(self,kernel):
        data_vol=self.data_vol
        logging.info(f"Filtering along Y with kernel length={kernel.size}")

        if __debug__:
            time_0 = time.perf_counter()

        Y_dim = data_vol.shape[1]
        chunk_size = Y_dim//self.max_number_of_processes
        #for i in range(number_of_processes):
        #    no_OF_filter_along_Y_chunk(i, kernel)
        chunk_indexes = [i for i in range(self.max_number_of_processes)]
        chunk_sizes = [chunk_size]*self.max_number_of_processes
        chunk_offsets = [0]*self.max_number_of_processes
        kernels = [kernel]*self.max_number_of_processes
        with ProcessPoolExecutor(max_workers=self.max_number_of_processes) as executor:
            for _ in executor.map(self.no_OF_filter_along_Y_chunk,
                                chunk_indexes,
                                chunk_sizes,
                                chunk_offsets,
                                kernels):
                logging.debug(f"PU #{_} finished")
        remainding_slices = Y_dim % self.max_number_of_processes
        if remainding_slices > 0:
            chunk_indexes = [i for i in range(remainding_slices)]
            chunk_sizes = [1]*remainding_slices
            chunk_offsets = [chunk_size*self.max_number_of_processes]*remainding_slices
            kernels = [kernel]*remainding_slices
            with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
                for _ in executor.map(self.no_OF_filter_along_Y_chunk,
                                    chunk_indexes,
                                    chunk_sizes,
                                    chunk_offsets,
                                    kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along Y spent {time_1 - time_0} seconds")

    def no_OF_filter_along_X(self,kernel):
        data_vol=self.data_vol
        logging.info(f"Filtering along X with kernel length={kernel.size}")
        if __debug__:
            time_0 = time.perf_counter()

        X_dim = data_vol.shape[2]
        chunk_size = X_dim//self.max_number_of_processes
        #for i in range(number_of_processes):
        #    no_OF_filter_along_X_chunk(i, kernel)
        chunk_indexes = [i for i in range(self.max_number_of_processes)]
        chunk_sizes = [chunk_size]*self.max_number_of_processes
        chunk_offsets = [0]*self.max_number_of_processes
        kernels = [kernel]*self.max_number_of_processes
        with ProcessPoolExecutor(max_workers=self.max_number_of_processes) as executor:
            for _ in executor.map(self.no_OF_filter_along_X_chunk,
                                chunk_indexes,
                                chunk_sizes,
                                chunk_offsets,
                                kernels):
                logging.debug(f"PU #{_} finished")
        remainding_slices = X_dim % self.max_number_of_processes
        if remainding_slices > 0:
            chunk_indexes = [i for i in range(remainding_slices)]
            chunk_sizes = [1]*remainding_slices
            chunk_offsets = [chunk_size*self.max_number_of_processes]*remainding_slices
            kernels = [kernel]*remainding_slices
            with ProcessPoolExecutor(max_workers=remainding_slices) as executor:
                for _ in executor.map(self.no_OF_filter_along_X_chunk,
                                    chunk_indexes,
                                    chunk_sizes,
                                    chunk_offsets,
                                    kernels):
                    logging.debug(f"PU #{_} finished")

        if __debug__:
            time_1 = time.perf_counter()
            logging.debug(f"Filtering along X spent {time_1 - time_0} seconds")

    def no_OF_filter(self,kernels):
        data_vol=self.data_vol
        self.no_OF_filter_along_Z(kernels[0])
        data_vol[...] = self._filtered_vol[...]
        self.no_OF_filter_along_Y(kernels[1])
        data_vol[...] = self._filtered_vol[...]
        self.no_OF_filter_along_X(kernels[2])

    def feedback_periodic(self,stopEv: threading.Event):
        time_0 = time.perf_counter()
        while not stopEv.is_set():
            current_time = time.perf_counter()
            if self.timeout_mins > 0:
                if (current_time - time_0) > (60 * self.timeout_mins):
                    pass
                    #_thread.interrupt_main() #may not work inside a class                    raise Exception(f"Timeout reached {self.timeout_mins} mins elapsed. Terminating now.")
            #logging.info(f"{100*__percent__.value/np.sum(data_vol.shape):3.2f} % completed")
            logging.info(f"{__percent__.value} completed")
            time.sleep(1)
        logging.info("feedback_periodic thread stopped.")


    def runOpticalFlow(self, data_vol:np.ndarray):

        """
        runs opticalFlow filter on the numpy volume provided
        """

        vol_size = data_vol.dtype.itemsize * data_vol.size
        logging.info(f"shape of the input volume (Z, Y, X) = {data_vol.shape}")
        logging.info(f"type of the volume = {data_vol.dtype}")
        logging.info(f"vol requires {vol_size/(1024*1024):.1f} MB")
        logging.info(f"data max = {data_vol.max()}")
        logging.info(f"data min = {data_vol.min()}")
        vol_mean = data_vol.mean()
        logging.info(f"Input vol average = {vol_mean}")

        if self.bComputeFlowWithPreviousFlow:
            get_flow = self.get_flow_without_prev_flow
            logging.info("No reusing adjacent OF fields as predictions")
        else:
            get_flow = self.get_flow_with_prev_flow
            logging.info("Using adjacent OF fields as predictions")

        sigma=self.sigma
        logging.info(f"sigma={tuple(sigma)}")

        kernels = [None]*3
        kernels[0] = self.get_gaussian_kernel(sigma[0])
        kernels[1] = self.get_gaussian_kernel(sigma[1])
        kernels[2] = self.get_gaussian_kernel(sigma[2])
        logging.info(f"length of each filter (Z, Y, X) = {[len(i) for i in [*kernels]]}")

        # Copy the volume to shared memory
        randint0 = random.randint(0,999999) #Randomize the shared memory name

        SM_vol=None
        SM_filtered_vol=None

        try:
                
            try:
                SM_vol = shared_memory.SharedMemory(
                    create=True,
                    size=vol_size,
                    name="vol"+str(randint0)) # See /dev/shm/
            except Exception as sm_vol_e:
                raise MemoryError("Error creating shared memory SM_vol: {sm_vol_e}")
                #sys.exit("Error creating shared memory SM_vol: {sm_vol_e}")

            # _vol = np.ndarray(
            #     shape=vol.shape,
            #     dtype=vol.dtype,
            #     buffer=SM_vol.buf)
            # _vol[...] = vol[...]
            # vol = _vol
            self.data_vol = np.ndarray(
                shape=data_vol.shape,
                dtype=data_vol.dtype,
                buffer=SM_vol.buf)
            try:
                SM_filtered_vol = shared_memory.SharedMemory(
                    create=True,
                    size=vol_size,
                    name="filtered_vol"+str(randint0)) # See /dev/shm
            except Exception as sm_filtered_vol_e:
                #logging.error(f"Error: {sm_filtered_vol_e}")
                SM_vol.close()
                SM_vol.unlink()
                raise MemoryError("Error creating shared memory SM_filtered_vol: {sm_vol_e}")
                #sys.exit("Error creating shared memory SM_filtered_vol: {sm_vol_e}")

            #_filtered_vol is a global variable
            #Consider putting this in a class
            self._filtered_vol = np.ndarray(
                shape=data_vol.shape,
                dtype=data_vol.dtype,
                buffer=SM_filtered_vol.buf)
            self._filtered_vol.fill(0)

            #logging.info(f"Number of concurrent processes: {number_of_PUs}")

            #feddback_thread = threading.Thread(target=feedback_periodic)
            stopEv=threading.Event()
            feddback_thread = threading.Thread(target=self.feedback_periodic, args=(stopEv,))
            feddback_thread.daemon = True # To obey CTRL+C interruption.
            #This also means that the thread is killed when the program exits
            feddback_thread.start()

            if __debug__:
                logging.info(f"Filtering ...")
                time_0 = time.perf_counter()
            #Starts filtering
            if self.disable_OF_compensation:
                self.no_OF_filter(kernels) #Filters without optical flow
            else:
                self.OF_filter(kernels, self.OF_LEVELS, self.OF_WINDOW_SIZE) #Filters with optical flow

            #When filtering is complete it continues here
            #stop feedback_thread.
            # There no stop() function to do that, so Event() is used
            stopEv.set()

            if __debug__:
                #time_1 = time.perf_counter()        
                time_1 = time.perf_counter()        
                logging.info(f"Volume filtered in {time_1 - time_0} seconds")

            result = np.array(self._filtered_vol) #makes a copy, to a non-shared memory space

            return result
        
        except Exception as e:
            logging.error(f"Some error occured: {str(e)}")
        
        finally:
            logging.info("Closing and unlinking shared memory")
            SM_vol.close()
            SM_vol.unlink()
            SM_filtered_vol.close()
            SM_filtered_vol.unlink()



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

    return parser

if __name__ == "__main__":

    parser = parseArgs
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
        enable_mem_map = args.mamory_map,
        max_number_of_processes=args.number_of_processes,
        bComputeFlowWithPreviousFlow=args.recompute_flow,
        timeout_mins = args.timeout,
        )
    
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
            mrc.data
    else:
        logging.debug(f"Writting TIFF file")
        skimage.io.imsave(args.output, _filtered_vol.astype(np.float32), plugin="tifffile")

    if __debug__:
        time_1 = time.perf_counter()        
        logging.info(f"written \"{args.output}\" in {time_1 - time_0} seconds")


