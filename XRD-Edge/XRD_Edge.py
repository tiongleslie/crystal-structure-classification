import numpy as np
import math


class XRD_Edge(object):
    def __init__(self, I=None):
        if I is not None:
            self.I = I

    @staticmethod
    def plot_edge(im, pixel=255):
        outImage = im
        outImage = outImage.astype(np.float32)
        tol = 1.7
        thre = 50.
        d_thre = 10.
        tolenrance_range = []

        for i in range(256):
            for j in range(256):
                if im[i, j] >= thre:
                    for m in range(256):
                        for n in range(256):
                            if im[m, n] >= thre:
                                if i != m and j != n:
                                    D = math.sqrt(np.sum(np.square(n - j) - 1) + np.sum(np.square(m - i) - 1))
                                    if D >= d_thre:
                                        tolenrance_range.append(D)

        for i in range(256):
            for j in range(256):
                if im[i, j] >= thre:
                    for m in range(256):
                        for n in range(256):
                            if im[m, n] >= thre:
                                if i != m and j != n:
                                    x = np.array([n, j])
                                    y = np.array([m, i])
                                    D = math.sqrt((math.pow(n - j, 2)) + (math.pow(m - i, 2)))

                                    if len(tolenrance_range) <= 100:
                                        nPoints = max(abs(np.diff(x)), abs(np.diff(y))) + 1
                                        rIndex = np.round(np.linspace(y[0], y[1], nPoints - 1)).astype(int)
                                        cIndex = np.round(np.linspace(x[0], x[1], nPoints - 1)).astype(int)
                                        outImage[rIndex, cIndex] = pixel
                                    else:
                                        if D <= tol * min(tolenrance_range):
                                            nPoints = max(abs(np.diff(x)), abs(np.diff(y))) + 1
                                            rIndex = np.round(np.linspace(y[0], y[1], nPoints - 1)).astype(int)
                                            cIndex = np.round(np.linspace(x[0], x[1], nPoints - 1)).astype(int)
                                            outImage[rIndex, cIndex] = pixel

        return outImage, tolenrance_range

    def create_descriptor(self, C):
        temp = self.I
        temp = temp.astype(np.float32)
        r_gray = self.I
        r_gray[116:(116+22), 116:(116+22)] = 0.
        r_gray = r_gray.astype(np.float32)

        if r_gray.max() == 0:
            if C == 1:
                return np.dstack((temp, temp, temp))
            elif C == 2:
                return np.dstack((temp, temp, temp))
            elif C == 3:
                return np.dstack((temp, temp, temp))
        else:
            im = (((r_gray - 0) / (r_gray.max() - 0)) * 256).astype(np.float32)
            im[116:(116+22), 116:(116+22)] = temp[116:(116+22), 116:(116+22)]

            test_r, tolenrance_range = self.plot_edge(im)

            for i in range(256):
                for j in range(256):
                    if im[i, j] > 5:
                        test_r[i, j] = im[i, j]

            if C == 1:
                return np.dstack((test_r, im, im))
            elif C == 2:
                return np.dstack((im, test_r, im))
            elif C == 3:
                return np.dstack((im, im, test_r))
