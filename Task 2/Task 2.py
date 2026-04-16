import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import pywt
import statsmodels.api as sm
import scipy.fftpack


def ButterBandpassFilter(data, lowCutoff, highCutoff, samplingRate, filterOrder):
    nyqFreq = 0.5 * samplingRate
    low = lowCutoff / nyqFreq
    high = highCutoff / nyqFreq
    b, a = butter(filterOrder, [low, high], btype='band')
    filteredData = filtfilt(b, a, data)
    return filteredData


def ExtractDwtFeatures(segment):
    coeffs = pywt.wavedec(segment, 'db4', level=2)
    approxCoeff = coeffs[0]

    meanVal = np.mean(approxCoeff)
    stdVal = np.std(approxCoeff)
    energyVal = np.sum(np.square(approxCoeff))

    return meanVal, stdVal, energyVal


def ExtractAcDctFeatures(segment):
    nLags = min(1000, len(segment) - 1)
    acfValues = sm.tsa.acf(segment, nlags=nLags)
    acSegment = acfValues[:min(100, len(acfValues))]
    dctValues = scipy.fftpack.dct(acSegment, type=2)
    return dctValues


targetFiles = ['ECG_Ali.txt', 'ECG_Mohamed.txt']

thresholdMap = {
    'ECG_Ali.txt': 60000,
    'ECG_Mohamed.txt': 76000
}

globalFeatureMap = []
firstDctResults = []
fileNamesForDct = []

for fileName in targetFiles:
    with open(fileName) as f:
        lines = f.readlines()

    ecgSignal = []
    for i in range(len(lines) - 1):
        lineData = lines[i + 1].split()
        ecgSignal.append(int(lineData[0]))

    ecgX = list(range(len(ecgSignal)))[:1000]
    ecgY = ecgSignal[:1000]

    filteredSignal = ButterBandpassFilter(ecgY, 1, 40, 250, 2)

    arrX = np.array(ecgX)
    arrY = np.array(filteredSignal)

    dx = np.diff(arrX)
    dyDiff = np.diff(arrY)

    dy = np.zeros(len(arrY))
    dy[:-1] = dyDiff / dx

    result = np.array([dy[i] ** 2 for i in range(len(dy))])

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

    currentThreshold = thresholdMap.get(fileName, 60000)
    thresholdPeaks, _ = find_peaks(result, height=currentThreshold)

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
            qrInterval = trueR - trueQ
            rsInterval = trueS - trueR

            dyQs = filteredSignal[trueS] - filteredSignal[trueQ]
            dxQs = trueS - trueQ
            qsSlope = dyQs / dxQs if dxQs != 0 else 0

            segmentStart = max(0, trueR - round(0.2 * fs))
            segmentEnd = min(len(filteredSignal), trueR + round(0.2 * fs))
            beatSegment = filteredSignal[segmentStart:segmentEnd]

            meanDwt, stdDwt, energyDwt = ExtractDwtFeatures(beatSegment)
            dctResult = ExtractAcDctFeatures(beatSegment)

            positiveDct = [v for v in dctResult if v > 0][:24]

            firstDctResults.append(dctResult)
            fileNamesForDct.append(fileName)

            featureEntry = {
                "signalName": fileName,
                "rPeakIndex": trueR,
                "qrInterval": qrInterval,
                "rsInterval": rsInterval,
                "qsSlope": qsSlope,
                "dwtMean": meanDwt,
                "dwtStd": stdDwt,
                "dwtEnergy": energyDwt,
            }

            for i, val in enumerate(positiveDct):
                featureEntry[f"acDctPos_{i+1}"] = val

            globalFeatureMap.append(featureEntry)

    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Analysis for {fileName}", fontsize=16, fontweight='bold')

    plt.subplot(3, 2, 1)
    plt.plot(ecgX, ecgY)
    plt.title("1: Raw ECG Signal")

    plt.subplot(3, 2, 2)
    plt.plot(np.arange(len(filteredSignal)), filteredSignal)
    plt.title("2: Filtered Signal")

    plt.subplot(3, 2, 3)
    plt.plot(np.arange(len(filteredSignal)), dy)
    plt.title("3: Derivative Signal")

    plt.subplot(3, 2, 4)
    plt.plot(np.arange(len(result)), result)
    plt.plot(allPeaks, [result[p] for p in allPeaks], "x", color='blue', label='All Peaks')
    plt.plot(thresholdPeaks, [result[p] for p in thresholdPeaks], "x", color='red', label='Threshold Peaks')
    plt.title("4: Squared & Integrated (All Peaks)")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(np.arange(len(result)), result)
    plt.plot(thresholdPeaks, [result[p] for p in thresholdPeaks], "x", color='red')
    plt.title("5: Integrated with Threshold Peaks")

    plt.subplot(3, 2, 6)
    plt.plot(np.arange(len(filteredSignal)), filteredSignal)

    if len(qPoints) > 0:
        plt.plot(qPoints, filteredSignal[qPoints], 'v', color='green', label='Q')
        plt.plot(rPoints, filteredSignal[rPoints], 'o', color='red', label='R')
        plt.plot(sPoints, filteredSignal[sPoints], '^', color='blue', label='S')

    plt.title("6: Fiducial Points (Q, R, S)")
    plt.legend()

    plt.tight_layout()


dfFeatures = pd.DataFrame(globalFeatureMap)
dfFeatures.to_excel("ECG_Features_Data.xlsx", index=False)

# print(dfFeatures)

if len(firstDctResults) >= 2:
    plt.figure(figsize=(20, 5))

    plt.subplot(121)
    plt.plot(firstDctResults[0])
    plt.title(fileNamesForDct[0])

    plt.subplot(122)
    plt.plot(firstDctResults[1])
    plt.title(fileNamesForDct[1])

plt.show()