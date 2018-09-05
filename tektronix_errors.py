# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:21:55 2017

@author: ebarr
"""

def error_str_to_dict(errorstr,counter):
    out = {}
    errorstr = errorstr.strip()
    errors = [error.strip(",") for error in errorstr.splitlines()]
    for error in errors:
        out[error] = counter
        counter += 1
    return out   
    
errors = {}
errors["noError"] = 0

estr = """
errorNotConnected,
errorIncompatibleFirmware,
errorBootLoaderNotRunning,
errorTooManyBootLoadersConnected,
errorRebootFailure,
errorGNSSNotInstalled,
errorGNSSNotEnabled
"""
errors.update(error_str_to_dict(estr,101))  

estr="""
errorPOSTFailureFPGALoad,
errorPOSTFailureHiPower,
errorPOSTFailureI2C,
errorPOSTFailureGPIF,
errorPOSTFailureUsbSpeed,
errorPOSTDiagFailure,
errorPOSTFailure3P3VSense
"""
errors.update(error_str_to_dict(estr,201))

estr = """
errorBufferAllocFailed,
errorParameter,
errorDataNotReady
"""
errors.update(error_str_to_dict(estr,301))
estr = """
errorParameterTraceLength,
errorMeasurementNotEnabled,
errorSpanIsLessThanRBW,
errorFrequencyOutOfRange,
""" 
errors.update(error_str_to_dict(estr,1101))
estr = """
errorStreamADCToDiskFileOpen,
errorStreamADCToDiskAlreadyStreaming,
errorStreamADCToDiskBadPath,
errorStreamADCToDiskThreadFailure,
errorStreamedFileInvalidHeader,
errorStreamedFileOpenFailure,
errorStreamingOperationNotSupported,
errorStreamingFastForwardTimeInvalid,
errorStreamingInvalidParameters,
errorStreamingEOF,
"""
errors.update(error_str_to_dict(estr,1201))
estr = """
errorIQStreamInvalidFileDataType,
errorIQStreamFileOpenFailed,
errorIQStreamBandwidthOutOfRange,
 """
errors.update(error_str_to_dict(estr,1301))
estr = """
errorTimeout,
errorTransfer,
errorFileOpen,
errorFailed,
errorCRC,
errorChangeToFlashMode,
errorChangeToRunMode,
errorDSPLError,
errorLOLockFailure,
errorExternalReferenceNotEnabled,
errorLogFailure,
errorRegisterIO,
errorFileRead,
errorConsumerNotActive,
"""
errors.update(error_str_to_dict(estr,3001))
estr = """
errorDisconnectedDeviceRemoved,
errorDisconnectedDeviceNodeChangedAndRemoved,
errorDisconnectedTimeoutWaitingForADcData,
errorDisconnectedIOBeginTransfer,
errorOperationNotSupportedInSimMode,
"""
errors.update(error_str_to_dict(estr,3101))
estr = """
errorFPGAConfigureFailure,
errorCalCWNormFailure,
errorSystemAppDataDirectory,
errorFileCreateMRU,
errorDeleteUnsuitableCachePath,
errorUnableToSetFilePermissions,
errorCreateCachePath,
errorCreateCachePathBoost,
errorCreateCachePathStd,
errorCreateCachePathGen,
errorBufferLengthTooSmall,
errorRemoveCachePath,
errorGetCachingDirectoryBoost,
errorGetCachingDirectoryStd,
errorGetCachingDirectoryGen,
errorInconsistentFileSystem,
"""
errors.update(error_str_to_dict(estr,3201))
estr = """
errorWriteCalConfigHeader,
errorWriteCalConfigData,
errorReadCalConfigHeader,
errorReadCalConfigData,
errorEraseCalConfig,
errorCalConfigFileSize,
errorInvalidCalibConstantFileFormat,
errorMismatchCalibConstantsSize,
errorCalConfigInvalid,
"""
errors.update(error_str_to_dict(estr,3301))
  
estr = """
errorFlashFileSystemUnexpectedSize,
errorFlashFileSystemNotMounted,
errorFlashFileSystemOutOfRange,
errorFlashFileSystemIndexNotFound,
errorFlashFileSystemReadErrorCRC,
errorFlashFileSystemReadFileMissing,
errorFlashFileSystemCreateCacheIndex,
errorFlashFileSystemCreateCachedDataFile,
errorFlashFileSystemUnsupportedFileSize,
errorFlashFileSystemInsufficentSpace,
errorFlashFileSystemInconsistentState,
errorFlashFileSystemTooManyFiles,
errorFlashFileSystemImportFileNotFound,
errorFlashFileSystemImportFileReadError,
errorFlashFileSystemImportFileError,
errorFlashFileSystemFileNotFoundError,
errorFlashFileSystemReadBufferTooSmall,
errorFlashWriteFailure,
errorFlashReadFailure,
errorFlashFileSystemBadArgument,
errorFlashFileSystemCreateFile,
"""
errors.update(error_str_to_dict(estr,3401))
        
estr = """
errorMonitoringNotSupported,
errorAuxDataNotAvailable,
"""
errors.update(error_str_to_dict(estr,3501))
        
estr = """
errorBatteryCommFailure
errorBatteryChargerCommFailure
errorBatteryNotPresent
"""
errors.update(error_str_to_dict(estr,3601))

estr = """
errorESTOutputPathFile
errorESTPathNotDirectory,
errorESTPathDoesntExist,
errorESTUnableToOpenLog,
errorESTUnableToOpenLimits,
"""
errors.update(error_str_to_dict(estr,3701))

errors["errorRevisionDataNotFound"] = 3801
		
estr = """
error112MHzAlignmentSignalLevelTooLow,
error10MHzAlignmentSignalLevelTooLow,
errorInvalidCalConstant,
errorNormalizationCacheInvalid,
errorInvalidAlignmentCache,
errorLockExtRefAfterAlignment,
"""
errors.update(error_str_to_dict(estr,3901))

errors["errorTriggerSystem"] = 4000
errors["errorVNAUnsupportedConfiguration"] = 4100
errors["errorADCOverrange"] = 9000
errors["errorOscUnlock"] = 9001          
errors["errorNotSupported"] = 9901
errors["errorPlaceholder"] = 9999
errors["notImplemented"] = -1

codes = dict([(val,key) for key,val in errors.items()])

