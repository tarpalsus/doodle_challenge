# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def AILabUtil(pathlist = [r'C:\tools\cuda\bin',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin']):
    import GPUtil
    
    import os
    os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)
    import tensorflow as tf
    import keras
    
    available_device = None
    GPUavailability = GPUtil.getAvailability(GPUtil.getGPUs(), maxLoad = 0.4, maxMemory = 0.4, includeNan=False, excludeID=[], excludeUUID=[])
    for i, device in enumerate(GPUavailability):
        if device == 1:
            available_device = i
            break
    if available_device == None:
        print('No available GPU devices')
    return available_device
        
