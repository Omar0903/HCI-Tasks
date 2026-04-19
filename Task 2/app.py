import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
import pywt
import statsmodels.api as sm
import scipy.fftpack


# =========================
# SIGNAL PROCESSING
# =========================

def butterBandpassFilter(data, lowCutoff, highCutoff, fs, order=2):
    nyq = 0.5 * fs
    low = lowCutoff / nyq
    high = highCutoff / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def movingEnergy(signal, fs):
    dx = np.arange(len(signal))
    dy = np.zeros(len(signal))

    d = np.diff(signal)
    dy[:-1] = d

    result = dy ** 2

    win = round(0.03 * fs)
    cumsum = 0

    for i in range(win):
        cumsum += result[i] / win
        result[i] = cumsum

    for i in range(win, len(result)):
        cumsum += result[i] / win
        cumsum -= result[i - win] / win
        result[i] = cumsum

    return result


# =========================
# FEATURE EXTRACTION
# =========================

def extractDwtFeatures(segment):
    coeffs = pywt.wavedec(segment, 'db4', level=2)
    approx = coeffs[0]

    return np.mean(approx), np.std(approx), np.sum(approx ** 2)


def extractAcDctFeatures(segment):
    acf = sm.tsa.acf(segment, nlags=min(1000, len(segment) - 1))
    acf = acf[:min(100, len(acf))]
    return scipy.fftpack.dct(acf, type=2)


def getPositiveDct(dctVals, limit=24):
    return [v for v in dctVals if v > 0][:limit]


# =========================
# ECG PIPELINE
# =========================

def loadSignal(fileName, limit=1000):
    with open(fileName) as f:
        lines = f.readlines()

    sig = [float(lines[i + 1].split()[0]) for i in range(len(lines) - 1)]
    return np.array(sig[:limit])


def detectRPeaks(result, threshold):
    peaks, _ = find_peaks(result, height=threshold)
    return peaks


def extractQRST(signal, rIndex, fs):
    win = round(0.1 * fs)

    start = max(0, rIndex - win)
    end = min(len(signal), rIndex + win)

    r = start + np.argmax(signal[start:end])

    q = r
    while q > 0 and signal[q - 1] < signal[q]:
        q -= 1

    s = r
    while s < len(signal) - 1 and signal[s + 1] < signal[s]:
        s += 1

    return q, r, s


def extractBeatSegment(signal, rIndex, fs):
    start = max(0, rIndex - round(0.2 * fs))
    end = min(len(signal), rIndex + round(0.2 * fs))
    return signal[start:end]


# =========================
# MAIN PROCESS
# =========================

fs = 250.0

targetFiles = ['ECG_Ali.txt', 'ECG_Mohamed.txt']
thresholdMap = {
    'ECG_Ali.txt': 60000,
    'ECG_Mohamed.txt': 76000
}

globalFeatures = []
firstDct = []
dctNames = []


for fileName in targetFiles:

    ecg = loadSignal(fileName)

    filtered = butterBandpassFilter(ecg, 1, 40, fs)
    result = movingEnergy(filtered, fs)

    allPeaks = find_peaks(result)[0]
    rPeaks = detectRPeaks(result, thresholdMap.get(fileName, 60000))

    qList, rList, sList = [], [], []

    for i, p in enumerate(rPeaks):

        q, r, s = extractQRST(filtered, p, fs)

        qList.append(q)
        rList.append(r)
        sList.append(s)

        if i == 0:

            beat = extractBeatSegment(filtered, r, fs)

            meanDwt, stdDwt, energyDwt = extractDwtFeatures(beat)
            dctVals = extractAcDctFeatures(beat)
            posDct = getPositiveDct(dctVals)

            firstDct.append(dctVals)
            dctNames.append(fileName)

            entry = {
                "signalName": fileName,
                "rPeakIndex": r,
                "dwtMean": meanDwt,
                "dwtStd": stdDwt,
                "dwtEnergy": energyDwt
            }

            for i, v in enumerate(posDct):
                entry[f"acDctPos_{i+1}"] = v

            globalFeatures.append(entry)

    # ================= PLOTS =================
    plt.figure(figsize=(15, 10))
    plt.suptitle(fileName)

    plt.subplot(3, 2, 1)
    plt.plot(ecg); plt.title("Raw")

    plt.subplot(3, 2, 2)
    plt.plot(filtered); plt.title("Filtered")

    plt.subplot(3, 2, 3)
    plt.plot(result); plt.title("Energy")

    plt.subplot(3, 2, 4)
    plt.plot(result)
    plt.plot(allPeaks, result[allPeaks], 'x')
    plt.plot(rPeaks, result[rPeaks], 'ro')
    plt.title("Peaks")

    plt.subplot(3, 2, 5)
    plt.plot(filtered); plt.title("Filtered Signal")

    plt.subplot(3, 2, 6)
    plt.plot(filtered)
    if len(rList):
        plt.plot(rList, filtered[rList], 'ro')
        plt.plot(qList, filtered[qList], 'gv')
        plt.plot(sList, filtered[sList], 'b^')

    plt.title("QRS")


# =========================
# SAVE FEATURES
# =========================

df = pd.DataFrame(globalFeatures)
df.to_excel("ECG_Features_Data.xlsx", index=False)


# =========================
# TEST SIGNAL MATCHING
# =========================

testFile = "Test signal.txt"
test = loadSignal(testFile)

filteredTest = butterBandpassFilter(test, 1, 40, fs)
resultTest = movingEnergy(filteredTest, fs)

rPeaks = detectRPeaks(resultTest, 60000)

p = rPeaks[0]

q, r, s = extractQRST(filteredTest, p, fs)

segment = extractBeatSegment(filteredTest, r, fs)

testMean, testStd, testEnergy = extractDwtFeatures(segment)

print("\nTEST FEATURES")
print(testMean, testStd, testEnergy)


best = None
bestDiff = float("inf")

print("\nMATCHING")
for item in globalFeatures:

    diff = abs(testMean - item["dwtMean"])
    print(item["signalName"], diff)

    if diff < bestDiff:
        bestDiff = diff
        best = item["signalName"]

print("\nBEST MATCH:", best)