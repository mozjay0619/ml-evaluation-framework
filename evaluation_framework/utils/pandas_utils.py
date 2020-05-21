import pandas as pd
import numpy as np

def is_date_str_type(ser):
    
    try:
        pd.to_datetime(ser, format='%Y-%m-%d', errors='raise')
        return True
    except:
        return False
    
def is_not_date_str_type(ser):
    
    try:
        pd.to_datetime(ser.iloc[0:10], format='%Y-%m-%d', errors='raise')
        return False
    except:
        return True
        
def is_numeric_type(ser):
    
    if ser.dtype.kind in 'biufc':
        return True
    else:
        return False

def is_datetime_type(ser):
    
    if ser.dtype.kind in 'Mm':
        return True
    else:
        return False
    
def is_float32_type(ser):
    
    if ser.dtype=='float32':
        return True
    else:
        return False

def cast_datetime2int64(ser):
    
    return ser.astype(np.int64)

def cast_int64_2datetime(ser):
    
    return pd.to_datetime(ser)

def cast_numeric2float32(ser):
    
    return ser.astype(np.float32)

def encode_str2bytes(ser):
    
    lens = ser.str.len()
    maxlen = lens.max().astype(int)
    dtype = 'S{}'.format(maxlen, 1)
    return ser.values.astype(dtype)

def decode_bytes2str(ndarray):

    return ndarray.astype(str)

def encode_date_sequence(ser):
    
    return (ser - ser.min()).dt.days




    