import numpy as np
import cv2 as cv
import colour # colour-science for python
import pickle

# Read an RGB image
def readImg(filename):
    image = cv.imread(filename, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image / 255

def readBinImg(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def writeBinImg(img, filename):
    img = img.astype(np.float32)
    pickle.dump(img, open(filename, "wb"))

# Camera spec db: http://www.gujinwei.org/research/camspec/db.html
def readCamData():
    data = {}
    with open("data/camspec_database.txt", "r") as f:
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            r = np.array(f.readline().split())
            g = np.array(f.readline().split())
            b = np.array(f.readline().split())
            r = r.astype(np.float)
            g = g.astype(np.float)
            b = b.astype(np.float)
            R = {}
            G = {}
            B = {}
            for i in range(0, 33):
                wavelength = i * 10 + 400 # 400nm to 720nm
                R[wavelength] = r[i]
                G[wavelength] = g[i]
                B[wavelength] = b[i]
            data[line] = np.array([R, G, B])
    return data

def spectral2rgb(s):
    sd = colour.SpectralDistribution(s)
    cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.ILLUMINANTS_SDS['D65']
    XYZ = colour.sd_to_XYZ(sd, cmfs, illuminant)
    RGB = colour.XYZ_to_sRGB(XYZ / 100)
    return RGB

bayer_cfa0 = np.array([
    ["R", "G"],
    ["G", "B"]
])

bayer_cfa1 = np.array([
    ["B", "G"],
    ["G", "R"]
])

bayer_cfa2 = np.array([
    ["G", "R"],
    ["B", "G"]
])

bayer_cfa3 = np.array([
    ["G", "B"],
    ["R", "G"]
])

quad_bayer_cfa = np.array([
    ["R", "R", "G", "G"],
    ["R", "R", "G", "G"],
    ["G", "G", "B", "B"],
    ["G", "G", "B", "B"]
])

xtrans_cfa = np.array([
    ["G", "B", "R", "G", "R", "B"],
    ["R", "G", "G", "B", "G", "G"],
    ["B", "G", "G", "R", "G", "G"],
    ["G", "R", "B", "G", "B", "R"],
    ["B", "G", "G", "R", "G", "G"],
    ["R", "G", "G", "B", "G", "G"]
])

camdb = readCamData()

def cfa2rgb(cfa, camera_model):
    cam_resp = camdb[camera_model]
    cam_rgb = {
        "R": spectral2rgb(cam_resp[0]),
        "G": spectral2rgb(cam_resp[1]),
        "B": spectral2rgb(cam_resp[2])
    }
    _cfa = cfa.reshape(cfa.shape[0] * cfa.shape[1])
    cfa_rgb = np.array([None] * cfa.shape[0] * cfa.shape[1])
    for i in range(0, cfa.shape[0] * cfa.shape[1]):
        cfa_rgb[i] = cam_rgb[_cfa[i]]
    cfa_rgb = cfa_rgb.reshape(cfa.shape)
    return cfa_rgb

def im2cfa(im, cfa_filter):
    cfa = im.copy()
    filtershape = cfa_filter.shape
    for x in range(0, filtershape[0]):
        for y in range(0, filtershape[1]):
            cfa[x::filtershape[0], y::filtershape[1]] = cfa_filter[x, y] * im[x::filtershape[0], y::filtershape[1]]
    return cfa

def cfa2grayscale(cfa):
    return np.clip(np.dot(cfa, [0.2126, 0.7152, 0.0722]), 0, 1)

def cfaAddNoise(cfa, intensity):
    return cfa + np.random.normal(0, 1, cfa.shape) * intensity