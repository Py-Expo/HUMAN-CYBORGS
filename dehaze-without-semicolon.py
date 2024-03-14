import cv2
import math
import numpy as np
import os
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission



def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 70
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res
def increase_contrast(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Convert back to BGR if necessary
    if len(image.shape) == 3:
        equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return equalized

# if __name__ == '__main__':
#     import sys
#     try:
#         fn = sys.argv[1]
#     except:
#         fn = 'C:\\Users\\praji\\image-dehazing\\my-test\\haze5.jpeg'

#     def nothing(*argv):
#         pass

#     src = cv2.imread(fn)
#     I = src.astype('float64') / 255
#     output_folder = 'C:\\Users\\praji\\image-dehazing\\output\\'
#     dark = DarkChannel(I,15)
#     A = AtmLight(I, dark)
#     te = TransmissionEstimate(I, A,15)
#     t = TransmissionRefine(src, te)
#     J = Recover(I, t, A, 0.1)
#     cv2.imshow("dark", dark)
#     cv2.imshow("t", t)
#     cv2.imshow('I', src)
#     cv2.imshow('J', J)
#     cv2.imwrite("./image/J.png", J * 255)
#     cv2.waitKey()
# if __name__ == '__main__':
#     input_folder = 'C:\\Users\\praji\\image-dehazing\\test\\hazy\\'  # Folder containing input images
#     output_folder = 'C:\\Users\\praji\\image-dehazing\\output\\'  # Folder to save output images
#     sz = 1  # Dark Channel parameter
#     A_sz = 15  # Size parameter for estimating atmospheric light
#     tx = 0.1  # Minimum transmission value

#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Process each image in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.jpeg') or filename.endswith('.png'):
#             # Read the image
#             src = cv2.imread(os.path.join(input_folder, filename))
#             srcd=increase_contrast(src)
#             Id=srcd.astype('float64')/255
#             I = src.astype('float64') / 255

#             # Apply Dark Channel Prior
#             dark = DarkChannel(Id, sz)

#             # Estimate Atmospheric Light
#             A = AtmLight(Id, dark)

#             # Estimate Transmission
#             te = TransmissionEstimate(I, A, A_sz)

#             # Refine Transmission
#             t = TransmissionRefine(srcd, te)

#             # Recover the scene radiance
#             J = Recover(I, t, A, tx)

#             # Save the output image
#             cv2.imwrite(os.path.join(output_folder, 'dehazed_' + filename), J * 255)

#     print("Processing completed. Output images saved in:", output_folder)
if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'C:\\Users\\praji\\image-dehazing\\my-test\\haze5.jpeg'

    src = cv2.imread(fn)
    I = src.astype('float64') / 255
    output_folder = 'C:\\Users\\praji\\image-dehazing\\output\\'

    dark = DarkChannel(I, 1)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)
    cv2.imwrite(output_folder+'original.png',src)
    cv2.imwrite(output_folder + "dark_channel.png", dark*255)
    cv2.imwrite(output_folder + "transmission_estimated.png", te * 255)
    cv2.imwrite(output_folder + "refined_transmission.png", t * 255)
    cv2.imwrite(output_folder + "dehazed_image.png", J * 255)