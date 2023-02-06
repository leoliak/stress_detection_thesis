from biosppy import storage
from biosppy.signals import ecg
from biosppy.signals import eda
import numpy as np

# load raw ECG signal
lene = 20000

signal1, mdata = storage.load_txt('d11.txt')
signal1 = signal1[0:20000]

# signal2, mdata = storage.load_txt('./biosppy/examples/eda.txt')
# signal2 = signal2[0:lene]

# signal3, mdata = storage.load_txt('d13.txt')
# signal3 = signal3[0:lene]/10
# storage.store_txt('d13.txt', signal3, 2048)

# signal3 = np.round(signal3)
# signal3 = signal3.astype(int)

# # process it and plot
out = ecg.ecg(signal=signal1, sampling_rate=2048., show=True)
filtered = out["filtered"]
peaks = out["rpeaks"]

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(np.arange(filtered.shape[0]), filtered)
plt.show()
# out1 = eda.eda(signal=signal2, sampling_rate=1000., show=True)
# out2 = eda.eda(signal=signal3, sampling_rate=2048., show=True)