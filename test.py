from MinosTypeInfo import TypeInfo
from MinosData import MinosData
import numpy as np
from PolyfaceUtil import MinosMatlabWrapper

# # fpth = "/mnt/Data/RawData/Ephys/Mowgli/240820_192824_Mowgli/Minos/Player.bin"
# # fpth = "/mnt/Data/RawData/Ephys/Fred/241112_195312_Fred/Minos/Eye.bin"
# fpth = "/mnt/Data/RawData/Ephys/Fred/241112_195312_Fred/Minos/Poly Face Navigator/Trials.bin"
# data = MinosData(fpth)
# # print(np.unique(data.Values['ConvergenceZ']))
# # print(data.Values["Timestamp"][:20])
# # # # print('-'*15)
# print(data.Values['Event'])

minos_dir = '/mnt/Data/RawData/Ephys/Fred/241112_195312_Fred/'
tmp_dir = '/mnt/c/Users/shichen/Desktop/Project/Temporarily_recording_data/Fred/241112_195312_Fred'
processed_data = MinosMatlabWrapper(minos_dir, tmp_dir)
print(processed_data['Paradigm']['PolyFaceNavigator'].keys())