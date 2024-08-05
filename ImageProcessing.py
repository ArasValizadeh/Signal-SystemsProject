import numpy as np
import cv2

def apply_convolution(image, kernel, stride, padding):
    inpLen = len(image[0])
    kernelLen = len(kernel)
    outLen = int((inpLen-kernelLen)/stride + 1)

    output = np.zeros((outLen, outLen))

    for k in range(3):
        for i1 in range(outLen):
            if(i1*stride+kernelLen > inpLen):
                break
            for j1 in range(outLen):
                if(j1*stride+kernelLen > inpLen):
                    break
                sum = 0
                for i2 in range(kernelLen):
                    for j2 in range(kernelLen):
                        sum += float(image[k][i1*stride + i2][j1*stride + j2]) *  (kernel[i2][j2])
                if(sum<0):
                    sum = 0
                output[i1][j1] += sum

    return output

def max_pooling(image, size, stride):
    inpLen = len(image[0])
    outLen = int((inpLen-size)/stride + 1)
    output = np.zeros((2, outLen, outLen))
    print(len(output))

    for k in range(2):
        channel = np.zeros((outLen, outLen))
        for i1 in range(outLen):
            if(i1*stride+size > inpLen):
                break
            for j1 in range(outLen):
                if(j1*stride+size > inpLen):
                    break
                max = -10000
                for i2 in range(size):
                    for j2 in range(size):
                        if(image[k][i1*stride + i2][j1*stride + j2] > max):
                            max = image[k][i1*stride + i2][j1*stride + j2]
                channel[i1][j1] = max
        output[k] = channel

    return output


image = cv2.imread('yann_lecun.jpg', cv2.IMREAD_COLOR)
image_arr = cv2.resize(image, (600, 600), 
               interpolation = cv2.INTER_LINEAR)
image_array = np.array(image_arr)

(b_channel, g_channel, r_channel) = cv2.split(image_array)

mainImage = np.array([b_channel,g_channel,r_channel])




cv2.imwrite('image.jpg', image_array)


Edge_Enhancement = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
Horizontal_Edge_Dectection = [[1,1,1],[0,0,0],[-1,-1,-1]]
Vertical_Edge_Detection = [[1,0,-1],[1,0,-1],[1,0,-1]]

first_feature_map = []

Gaussian_Blur = [[1/273,4/273,7/273,4/273,1/273],[4/273,16/273,26/273,16/273,4/273],[7/273,26/273,41/273,26/273,7/273],[4/273,16/273,26/273,16/273,4/273],[1/273,4/273,7/273,4/273,1/273]]
Large_Edge_Enhancement = [[0,0,-1,0,0],[0,0,-1,0,0],[-1,-1,8,-1,-1],[0,0,-1,0,0],[0,0,-1,0,0]]

out_Edge_Enhancement = apply_convolution(mainImage, Edge_Enhancement, 1, 1)
out_Horizontal_Edge_Dectection = apply_convolution(mainImage, Horizontal_Edge_Dectection, 1, 1)
out_Vertical_Edge_Detection = apply_convolution(mainImage, Vertical_Edge_Detection, 1, 1)

first_feature_map.append(out_Edge_Enhancement)
first_feature_map.append(out_Horizontal_Edge_Dectection)
first_feature_map.append(out_Vertical_Edge_Detection)


out_Gaussian_Blur = apply_convolution(first_feature_map, Gaussian_Blur, 1, 1)
out_Large_Edge_Enhancement = apply_convolution(first_feature_map, Large_Edge_Enhancement, 1, 1)

second_feature_map = []

second_feature_map.append(out_Gaussian_Blur)
second_feature_map.append(out_Large_Edge_Enhancement)

final_output = max_pooling(second_feature_map,3,1)


cv2.imwrite('Edge_Enhancement.jpg', out_Edge_Enhancement)
cv2.imwrite('Horizontal_Edge_Dectection.jpg', out_Horizontal_Edge_Dectection)
cv2.imwrite('Vertical_Edge_Detection.jpg', out_Vertical_Edge_Detection)

cv2.imwrite('Gaussian_Blur.jpg', out_Gaussian_Blur)
cv2.imwrite('Large_Edge_Enhancement.jpg', out_Large_Edge_Enhancement)

cv2.imwrite('final1.jpg', final_output[0])
cv2.imwrite('final2.jpg', final_output[1])
