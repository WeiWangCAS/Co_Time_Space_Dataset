import numpy as np

from load_data import co_scaled
from quant_and_coding import HEVC_encoding
from hamming import hamming

qp = [0, 12, 22, 32, 42, 51]

dequant_data, compressed_bits, ori_bits, mse, psnr, comp_rate = HEVC_encoding(np.expand_dims(co_scaled, axis=0), 8, 51)

dequant_data = np.squeeze(dequant_data, axis=0)

print(f'The current comp_rate is {comp_rate}')
print(f'The current mse is {mse}')
print(f'The current psnr is {psnr}')

np.save('./reconstructed_data/qp51.npy', dequant_data)



