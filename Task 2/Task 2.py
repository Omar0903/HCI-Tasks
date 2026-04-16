import matplotlib.pyplot as plt
import numpy as np
import numpy as geek
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import pywt
import statsmodels.api as sm
import scipy.fftpack

def ButterBandpassFilter(data, lowCutoff, highCutoff, samplingRate, filterOrder):
    nyq = 0.5 * samplingRate
    low = lowCutoff / nyq
    high = highCutoff / nyq
    b, a = butter(filterOrder, [low, high], btype='band', analog=False, fs=None)
    filteredData = filtfilt(b, a, data)
    return filteredData

def ExtractDwtFeatures(segment):
    coeffs = pywt.wavedec(segment, 'db4', level=3)
    cA3 = coeffs[0]
    cD3 = coeffs[1]
    energyCa3 = np.sum(np.square(cA3))
    energyCd3 = np.sum(np.square(cD3))
    return energyCa3, energyCd3

def ExtractAcDctFeatures(segment):
    nLags = min(1000, len(segment) - 1)
    ac = sm.tsa.acf(segment, nlags=nLags)
    accSeg = ac[0:min(100, len(ac))]
    dctResult = scipy.fftpack.dct(accSeg, type=2)
    return dctResult

targetFiles = ['ECG_Ali.txt', 'ECG_Mohamed.txt']
globalFeatureMap = []
firstDctResults = []
fileNamesForDct = []

for fileName in targetFiles:
    with open(fileName) as f:
        lines = f.readlines()

    ecgX = []
    ecgY = []
    for i in range(len(lines) - 1):
        lineData = lines[i+1].split()
        ecgY.append(int(lineData[0]))

    ecgX = range(0, len(ecgY))[0:1000]
    ecgY = ecgY[0:1000]

    filteredSignal = ButterBandpassFilter(ecgY, 1, 40, 250, 2)

    arrX = geek.array(ecgX)
    arrY = geek.array(filteredSignal)
    dx = geek.diff(arrX)
    dyDiff = geek.diff(arrY)
    dy = np.zeros(len(arrY))
    dy[0:len(dy)-1] = dyDiff / dx

    result = [dy[i]**2 for i in range(len(dy))]

    winSize = round(0.03 * 250)
    sumVal = 0
    for j in range(winSize):
        sumVal += result[j] / winSize
        result[j] = sumVal

    for index in range(winSize, len(result)):
        sumVal += result[index] / winSize
        sumVal -= result[index - winSize] / winSize
        result[index] = sumVal

    allPeaks, _ = find_peaks(result)
    thresholdPeaks, _ = find_peaks(result, height=60000)

    fs = 250.0
    qPoints = []
    rPoints = []
    sPoints = []

    for idx, p in enumerate(thresholdPeaks):
        searchWindow = round(0.1 * fs)
        startR = max(0, p - searchWindow)
        endR = min(len(filteredSignal), p + searchWindow)
        trueR = startR + np.argmax(filteredSignal[startR:endR])
        rPoints.append(trueR)
        
        trueQ = trueR
        while trueQ > 0 and filteredSignal[trueQ - 1] < filteredSignal[trueQ]:
            trueQ -= 1
        qPoints.append(trueQ)
        
        trueS = trueR
        while trueS < len(filteredSignal) - 1 and filteredSignal[trueS + 1] < filteredSignal[trueS]:
            trueS += 1
        sPoints.append(trueS)
        
        if idx == 0:
            qrInterval = (trueR - trueQ) / fs 
            rsInterval = (trueS - trueR) / fs 
            dyQs = filteredSignal[trueS] - filteredSignal[trueQ]
            dxQs = (trueS - trueQ) / fs
            qsSlope = dyQs / dxQs if dxQs != 0 else 0
            
            segmentStart = max(0, trueR - round(0.2 * fs))
            segmentEnd = min(len(filteredSignal), trueR + round(0.2 * fs))
            beatSegment = filteredSignal[segmentStart:segmentEnd]
            
            dwtEnergyCa3, dwtEnergyCd3 = ExtractDwtFeatures(beatSegment)
            dctResult = ExtractAcDctFeatures(beatSegment)
            
            firstDctResults.append(dctResult)
            fileNamesForDct.append(fileName)
            
            dct1 = dctResult[0] if len(dctResult) > 0 else 0
            dct2 = dctResult[1] if len(dctResult) > 1 else 0
            dct3 = dctResult[2] if len(dctResult) > 2 else 0

            
            globalFeatureMap.append({
                'Signal_Name': fileName,
                'R_Peak_Index': trueR,
                'QR_Interval_s': qrInterval,
                'RS_Interval_s': rsInterval,
                'QS_Slope': qsSlope,
                'DWT_Energy_cA3': dwtEnergyCa3,
                'DWT_Energy_cD3': dwtEnergyCd3,
                'AC_DCT_1': dct1,
                'AC_DCT_2': dct2,
                'AC_DCT_3': dct3
            })

    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Analysis for {fileName}", fontsize=16, fontweight='bold')

    plt.subplot(3, 2, 1)
    plt.plot(ecgX, ecgY)
    plt.title("1: Raw ECG Signal")


    plt.subplot(3, 2, 2)
    plt.plot(np.arange(0, len(filteredSignal)), filteredSignal)
    plt.title("2: Filtered Signal")

    plt.subplot(3, 2, 3)
    plt.plot(np.arange(0, len(filteredSignal)), dy)
    plt.title("3: Derivative Signal")

    plt.subplot(3, 2, 4)
    plt.plot(np.arange(0, len(result)), result)
    plt.plot(allPeaks, [result[p] for p in allPeaks], "x", color='blue', label='All Peaks')
    plt.plot(thresholdPeaks, [result[p] for p in thresholdPeaks], "x", color='red', label='Threshold Peaks')
    plt.title("4: Squared & Integrated (All Peaks)")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(np.arange(0, len(result)), result)
    plt.plot(thresholdPeaks, [result[p] for p in thresholdPeaks], "x", color='red')
    plt.title("5: Integrated with Threshold Peaks")

    plt.subplot(3, 2, 6)
    plt.plot(np.arange(0, len(filteredSignal)), filteredSignal)
    if len(qPoints) > 0:
        plt.plot(qPoints, filteredSignal[qPoints], 'v', color='green', label='Q')
        plt.plot(rPoints, filteredSignal[rPoints], 'o', color='red', label='R')
        plt.plot(sPoints, filteredSignal[sPoints], '^', color='blue', label='S')
    plt.title("6: Fiducial Points (Q, R, S) on Filtered Signal")
   
    plt.legend()

    plt.tight_layout()

dfFeatures = pd.DataFrame(globalFeatureMap)
outFileName = 'ECG_Features_Data.xlsx'
dfFeatures.to_excel(outFileName, index=False)
print(dfFeatures.to_string())

if len(firstDctResults) >= 2:
    plt.figure(figsize=(24, 6))
    plt.suptitle("DCT Comparison for First QRS of Both Signals", fontsize=16, fontweight='bold')
    
    plt.subplot(121)
    plt.plot(np.arange(0, len(firstDctResults[0])), firstDctResults[0])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude(v)")
    plt.title(f"DCT of {fileNamesForDct[0]}")
    
    plt.subplot(122)
    plt.plot(np.arange(0, len(firstDctResults[1])), firstDctResults[1])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude(v)")
    plt.title(f"DCT of {fileNamesForDct[1]}")

plt.show()

