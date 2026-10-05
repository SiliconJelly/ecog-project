import scipy.io as sio
import matplotlib.pyplot as plt

y = sio.loadmat("ECoG_Handpose.mat")["y"]
fs = 1200
print("Shape:", y.shape)

t = y[0]
seg = (t >= 10) & (t <= 40)          # seconds 10-40

fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
ax[0].plot(t[seg], y[1, seg] - y[1, seg].mean())
ax[0].set_title("ECoG channel 1")
ax[1].plot(t[seg], y[61, seg])
ax[1].set_title("Cue: 0 rest, 1 fist, 2 peace, 3 open")
ax[2].plot(t[seg], y[62:67, seg].T)
ax[2].set_title("Data glove (5 fingers)")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
