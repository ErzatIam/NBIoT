from scipy.io import loadmat
import cupy as cp
import matplotlib.pyplot as plt
import csv
import codecs
import numpy as np

normalize = lambda x : x / cp.max(cp.abs(x))

def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(datas)

file_path = 'split/4.22.948.8_2.raw'

raw_data = cp.fromfile(file_path, dtype=cp.complex64)
raw_data = normalize(raw_data).flatten()

total = raw_data[:1920*30]

save_path = 'corrs.csv'
step = 20

corrs = cp.empty((int((len(total))/step),int(len(total)/step)))
count = 0
for i in range(int(len(total)/step)):
    offset1 = i * step
    if offset1 + 19200 >= len(total):
        break
    window_norm = normalize(total[offset1:offset1 + 1920]).conj()
    for j in range(i,int(len(total)/step)):
        offset2 = step * j + offset1 + 18000
        if offset2 + 1920 >= len(total):
            break
        slice_norm = normalize(total[offset2:offset2 + 1920])
        corr = cp.sum((window_norm * slice_norm).real)
        corrs[i][j] = corr
        count += 1
        print(count/(len(total)/step)/(int(18280/step)))

print(cp.unravel_index(cp.argmax(corrs),corrs.shape))


total = raw_data[:1920 * 50]


offset1 = step * cp.unravel_index(cp.argmax(corrs),corrs.shape)[0]
corrs = cp.empty((1, int((len(total)) / step)))

window_norm = total[offset1:offset1 + 1920].conj()
for i in range(int((len(total)) / step)):
    offset2 = step * i
    if offset2 + 1920 >= len(total):
        break
    slice_norm = total[offset2:offset2 + 1920]
    corr = cp.sum((window_norm * slice_norm).real)
    corrs[0, i] = corr
    print(i / int((len(total)) / step))

print(cp.unravel_index(cp.argmax(corrs), corrs.shape))
print(cp.max(corrs))
print(cp.mean(corrs))



x = np.arange(1, corrs.shape[1] + 1).flatten()/1920 * step / 10
y = cp.abs(corrs).get().flatten()
data_write_csv(save_path,y)  
plt.plot(x, y)
plt.savefig("corr.png")
plt.show()
