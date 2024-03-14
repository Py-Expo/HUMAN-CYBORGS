import cv2
import numpy as np
import requests 
import cv2 
import numpy as np 
import imutils 
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = max(int(imsz / 1000), 1)
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3], dtype=np.float64)
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
def LocalTransmissionEstimate(im, A, sz):
    h, w, _ = im.shape
    transmission = np.zeros((h, w))

    # Padding for the input image to handle border pixels
    pad_size = sz // 2
    padded_im = cv2.copyMakeBorder(im, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    for y in range(h):
        for x in range(w):
            # Define local region
            local_region = padded_im[y : y + sz, x : x + sz, :]

            # Calculate the dark channel of the local region
            local_dark = np.min(local_region, axis=(0, 1))

            # Estimate transmission based on the local dark channel
            local_transmission = 1 - 0.95 * local_dark / A

            # Take the maximum transmission value within the local region
            transmission[y, x] = np.max(local_transmission)

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
    r = 10
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

def dehaze_frame(frame):
    framed=increase_contrast(frame)
    hsv_frame = cv2.cvtColor(framed, cv2.COLOR_BGR2HSV)
    hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1] * 1.5, 0, 255)  # Increase saturation by a factor of 1.5
    saturated_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
    Id = saturated_frame.astype('float64') / 255
    I=frame.astype('float64')/255

    dark = DarkChannel(Id, 10)
    A = AtmLight(Id, dark)
    te = TransmissionEstimate(I, A, 20)
    t = TransmissionRefine(framed, te)
    J = Recover(I, t, A, 0.1)

    return J


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

def main():
    cap = cv2.VideoCapture(r"C:\Users\praji\image-dehazing\WhatsApp Video 2024-03-14 at 21.35.20_3ea865d5.mp4")  # Open default camera

    while True:
        ret, frame = cap.read()
        frame=cv2.resize(frame,(840,640))

        if not ret:
            print("Failed to capture frame")
            break


        dehazed_frame = dehaze_frame(frame)
        cv2.imshow('Original',cv2.resize(frame,(848,480)))
        # cv2.imshow('contrast',increase_contrast(frame))
        cv2.imshow('Dehazed',cv2.resize(dehazed_frame,(848,480)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
# import cv2
# from urllib.parse import urlparse

# def main():
#     # Replace the URL with your phone's IP camera address
#     url = "http://172.16.45.145:8080/shot.jpg"
#     while True: 
#         img_resp = requests.get(url) 
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
#         img = cv2.imdecode(img_arr, -1) 
#         frame = imutils.resize(img, width=640, height=480) 
#         dehazed_frame = dehaze_frame(frame)
#         cv2.imshow('Original', cv2.resize(frame, (640, 480)))
#         cv2.imshow('Dehazed', cv2.resize(dehazed_frame, (640, 480)))

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()

  
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last. 

  
