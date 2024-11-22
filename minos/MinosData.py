import numpy as np
import re
from minos.MinosTypeInfo import TypeInfo
from copy import deepcopy
import os
import struct
from minos.MinosUtil import str2int
import mmap

class MinosData():
    """ Class object for decoding binary files
        or txt produced by Minos.
    """
    def __init__(self, fpth, memmapOnly=False):
        self.Path = ""
        self.Type = 'Binary'
        self.Values = None
        self.Info = None
        self.Map = None
        self.initType()
        self.process(fpth, memmapOnly)
    
    def initType(self,):
        """ Initializing some default properties.
        """
        self.LogPattern = re.compile(r"\[(?P<timestamp>\d+)\] (?P<name>[\w\.]+) \(?(?P<type>.*?)\)? ?= (?P<value>[^\r\n]+)")
        self.MessagePattern = re.compile(r"\[(?P<timestamp>\d+)\] (?P<message>[^\r\n]+)")
        self.NetNumbers = ["System.SByte" "System.Byte" "System.Int16" "System.UInt16",
                           "System.Int32" "System.UInt32" "System.Int64" "System.UInt64",
                           "System.Single" "System.Double"]
    
    def str2array(self, string, func):
        """ Convert a string to an array based on certain function.
        """
        return [func(element) for element in string.split(',')]
    
    def getConvertFunc(self, type):
        """ Obtain the conversion function of a specific type.
        """
        # Default identity function
        func = lambda x: x        

        if type.endswith("[]"):
            func = lambda s: self.str2array(s, self.get_convert_func(type[:-2]))
        
        if type in self.NetNumbers:
            func = float
        elif type == 'System.Boolean':
            func = lambda s: s == "True"
        elif type == "UnityEngine.Color":
            func = lambda s: np.array([float(n) for n in s.replace("RGBA(", "").replace(")", "").split(",")])

        return func
    
    def process(self, fpth, memmapOnly):
        """ Main function for decoding the input file. If
        memmapOnly, only memmapfile will be loaded. 
        """
        if not os.path.exists(fpth):
            return
        self.Path = fpth
        
        # instead of using the full FileIO from Matlab, 
        # simplily get the basic file info used in this function
        file_size = os.path.getsize(fpth)
        _, file_ext = os.path.splitext(os.path.basename(fpth))

        # processing binary file
        if file_ext == '.bin':
            self.Type = 'Binary'

            # Open the file in binary read mode
            with open(fpth, 'rb') as fid:
                # Read the first 8 bytes
                s = fid.read(8)

            # Extract the version (bytes 5 and 6)
            ver = s[4:6]  # Python uses zero-based indexing

            # Calculate the header length (bytes 7 and 8) using struct.unpack to interpret as uint16
            headerLength = struct.unpack('H', s[6:8])[0]  # 'H' is the format character for unsigned short (uint16)

            # Calculate the offset
            offset = float(headerLength + 8)

            if ver == bytes([1, 0]):
                with open(fpth, 'rb') as fid:
                    # Seek to 8 bytes from the beginning of the file
                    fid.seek(8, os.SEEK_SET)
                    
                    # Read the header based on headerLength
                    header = fid.read(headerLength).decode('utf-8')
                    
                # Create TypeInfo object
                ti = TypeInfo(header)
                
                # Calculate the length
                len_value = int((file_size - offset) / float(ti.data.Bytes))
                
                # Check if len is an integer
                if abs(len_value - round(len_value)) > 1e-10:  # Using epsilon comparison
                    raise ValueError("Number of entries is not an integer")
                
                # Get format and info1 from TypeInfo object
                fmt, info1 = ti.getFormat()
                self.Info = ti.data

                if len_value == 0:
                    if not memmapOnly:
                        value = {k:fmt[k] for k in ['Type', 'Name']}
                        self.Values = value
                else:
                    if not memmapOnly:
                        # update the format information
                        fmt['Bytes'] = [cur.Bytes*cur.Length for cur in info1]
                        fmt['Offset'] = np.cumsum([0] + fmt['Bytes'][:-1])
                        n = len(info1)

                        # iteratively read the binary file based on the start position/dtype/strides of
                        # each type of data specified in TypeInfo object.
                        value = []
                        with open(fpth, 'rb') as fid:
                            with mmap.mmap(fid.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                                for i in range(n):
                                    start = int(offset + fmt['Offset'][i])
                                    wid = int(fmt['Size'][i][1])
                                    dtype = np.dtype(fmt['Type'][i])
                                    element_size = dtype.itemsize
                                    stride = ti.data.Bytes - fmt['Bytes'][i]

                                    # Calculate the number of bytes to read for the entire block (data + stride)
                                    bytes_per_element = element_size * wid
                                    total_bytes_per_entry = bytes_per_element+ stride
                                    total_bytes_to_read = total_bytes_per_entry * len_value

                                    # Read the entire block of data into raw_data
                                    mm.seek(start)
                                    raw_data = mm.read(total_bytes_to_read)

                                    # Create a NumPy array from the raw data
                                    # Convert the raw bytes to a NumPy array of the desired dtype
                                    full_array = np.frombuffer(raw_data, dtype=np.uint8) # not 100% sure about this int8 part

                                    # Extract only the data part (skip the stride)
                                    tmp = []
                                    counter = 0
                                    while counter<len(full_array):
                                        tmp.append(full_array[counter:counter+bytes_per_element].view(dtype))
                                        counter += total_bytes_per_entry
                                    tmp = np.array(tmp)
                                    if wid>1:
                                        tmp = np.array(tmp).reshape([len_value, wid]) 
                                    else:
                                        tmp = np.array(tmp).reshape(-1)

                                    # If the type is logical, convert to boolean array
                                    if info1[i].Type == 'logical':
                                        tmp = tmp.astype(bool)

                                    # Apply decoding if there are constants to process
                                    if len(info1[i].Constants['name'])!=0:
                                        tmp = ti.decode(info1[i], tmp)
                                    value.append(tmp)

                        
                        self.Values = {}
                        for idx, k in enumerate(fmt['Name']):
                            self.Values[k] = value[idx]
                    else:
                        raise NotImplementedError("memory map is used by default")
            else:
                raise ValueError("Unsupported version %d.%d" %(ver[0], ver[1]))
        elif file_ext == '.txt':
            self.Type = 'Log'

            # read the txt file and remove empty lines
            with open(fpth, 'r', encoding='utf-8') as file:
                txt = file.read()
                txt = [line for line in txt.splitlines() if line]

            n = len(txt)
            value = {k: [] for k in ['Timestamp', 'Name', 'Type', 'Value']}
            for line in txt:
                # the log file data should match with one of the predefined regex patterns
                match = re.match(self.LogPattern, line)
                if match is not None:
                    match = match.groupdict()
                    for k in match:
                        value[k[0].upper()+k[1:]].append(match[k])
                else:
                    match = re.match(self.MessagePattern, line).groupdict()
                    value['Timestamp'].append(match['timestamp'])
                    value['Value'].append(match['message'])
                    value['Type'].append("")
                    value['Name'].append("")
            
            value["Timestamp"] = str2int(value["Timestamp"], "int64")
            value["Name"] = [cur.replace(".", "") for cur in value["Name"]]
            self.Values = value
        else:
            raise NotImplementedError("Unsupported file extension %s" %file_ext)

















