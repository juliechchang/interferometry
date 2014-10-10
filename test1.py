# FFT of a greyscale image

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

## shifted returns the IFT of an image that has been shifted in the Fourier domain
def shifted (filename, n):
	im = Image.open(filename)
	imarr = np.asarray(im)
	f = np.fft.fft2(imarr)
	fshift = np.fft.fftshift(f)
	f1 = np.roll(fshift, n, axis=0)
	f_ishift = np.fft.ifftshift(f1)
	img_back = np.fft.ifft2(f_ishift)
	return img_back 

## Reference Image

imref = Image.open("ref.bmp")
ref_back = shifted("ref.bmp",500)

## Sample Image

im = Image.open("retina.bmp")
imarr = np.asarray(im)
f = np.fft.fft2(imarr)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

f1 = np.roll(fshift, 500, axis=0)
circle_shift = 20*np.log(np.abs(f1))

f_ishift = np.fft.ifftshift(f1)
img_back = np.fft.ifft2(f_ishift)
#img_back = np.abs(img_back)

## Extract phase

out = np.divide(img_back,ref_back)
#norm = np.divide(out, np.amax(out))
phase = np.angle(out)

## Figure

plt.figure(1)
plt.subplot(231),plt.imshow(im, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(circle_shift, cmap = 'gray')
plt.title('Circle Shift'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(imref, cmap = 'gray')
plt.title('Reference Image'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(phase , cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(2)
plt.imshow(phase)
plt.colorbar()
plt.show()

