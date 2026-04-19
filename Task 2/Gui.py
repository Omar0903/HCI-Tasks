import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import statsmodels.api as sm
import scipy.fftpack
import os


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
# MAIN APPLICATION
# =========================

class ECGAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Signal Analyzer")
        self.root.geometry("1200x800")
        
        self.fs = 250.0
        self.targetFiles = ['ECG_Ali.txt', 'ECG_Mohamed.txt']
        self.thresholdMap = {
            'ECG_Ali.txt': 60000,
            'ECG_Mohamed.txt': 76000
        }
        
        self.globalFeatures = []
        self.firstDct = []
        self.dctNames = []
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Analyze ECG Files", command=self.analyze_ecg_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Test Signal", command=self.test_signal).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear", command=self.clear_plots).pack(side=tk.LEFT, padx=5)
        
        # Results text area
        self.results_text = tk.Text(control_frame, height=2, width=60)
        self.results_text.pack(side=tk.LEFT, padx=20)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tab frames and store them
        self.tab_frames = []
        self.tab_frames.append(self.create_tab_frame("ECG Ali"))
        self.tab_frames.append(self.create_tab_frame("ECG Mohamed"))
        self.tab_frames.append(self.create_tab_frame("DCT Comparison"))
        self.tab_frames.append(self.create_tab_frame("Test Signal"))
    
    def create_tab_frame(self, name):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)
        return frame
    
    def analyze_ecg_files(self):
        self.globalFeatures = []
        self.firstDct = []
        self.dctNames = []
        
        results = []
        
        for idx, fileName in enumerate(self.targetFiles):
            if not os.path.exists(fileName):
                messagebox.showerror("Error", f"File {fileName} not found!")
                return
            
            ecg = loadSignal(fileName)
            filtered = butterBandpassFilter(ecg, 1, 40, self.fs)
            result = movingEnergy(filtered, self.fs)
            
            allPeaks = find_peaks(result)[0]
            rPeaks = detectRPeaks(result, self.thresholdMap.get(fileName, 60000))
            
            qList, rList, sList = [], [], []
            
            for i, p in enumerate(rPeaks):
                q, r, s = extractQRST(filtered, p, self.fs)
                qList.append(q)
                rList.append(r)
                sList.append(s)
                
                if i == 0:
                    beat = extractBeatSegment(filtered, r, self.fs)
                    meanDwt, stdDwt, energyDwt = extractDwtFeatures(beat)
                    dctVals = extractAcDctFeatures(beat)
                    posDct = getPositiveDct(dctVals)
                    
                    self.firstDct.append(dctVals)
                    self.dctNames.append(fileName)
                    
                    entry = {
                        "signalName": fileName,
                        "rPeakIndex": r,
                        "dwtMean": meanDwt,
                        "dwtStd": stdDwt,
                        "dwtEnergy": energyDwt
                    }
                    
                    for j, v in enumerate(posDct):
                        entry[f"acDctPos_{j+1}"] = v
                    
                    self.globalFeatures.append(entry)
                    results.append(f"{fileName}: Mean={meanDwt:.4f}, Std={stdDwt:.4f}, Energy={energyDwt:.2f}")
            
            # Plot in respective tab
            self.plot_ecg_analysis(idx, fileName, ecg, filtered, result, allPeaks, rPeaks, qList, rList, sList)
        
        # Plot DCT comparison
        self.plot_dct_comparison()
        
        # Save to Excel
        dfFeatures = pd.DataFrame(self.globalFeatures)
        dfFeatures.to_excel("ECG_Features_Data.xlsx", index=False)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "\n".join(results))
        
        messagebox.showinfo("Success", "Analysis complete! Features saved to ECG_Features_Data.xlsx")
    
    def plot_ecg_analysis(self, tab_idx, fileName, ecg, filtered, result, allPeaks, rPeaks, qList, rList, sList):
        frame = self.tab_frames[tab_idx]
        
        # Clear previous
        for widget in frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(15, 10))
        fig.suptitle(f"Analysis for {fileName}", fontsize=16, fontweight='bold')
        
        # 1: Raw ECG
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(ecg)
        ax1.set_title("1: Raw ECG Signal")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Amplitude")
        
        # 2: Filtered
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(filtered)
        ax2.set_title("2: Filtered Signal (Bandpass 1-40 Hz)")
        ax2.set_xlabel("Sample")
        ax2.set_ylabel("Amplitude")
        
        # 3: Derivative
        ax3 = fig.add_subplot(3, 2, 3)
        dy = np.zeros(len(filtered))
        dy[:-1] = np.diff(filtered)
        ax3.plot(dy)
        ax3.set_title("3: Derivative Signal")
        ax3.set_xlabel("Sample")
        ax3.set_ylabel("Amplitude")
        
        # 4: Energy with all peaks
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(result)
        ax4.plot(allPeaks, result[allPeaks], 'x', color='blue', label='All Peaks')
        ax4.plot(rPeaks, result[rPeaks], 'o', color='red', label='R Peaks')
        ax4.set_title("4: Squared & Integrated (All Peaks)")
        ax4.set_xlabel("Sample")
        ax4.set_ylabel("Energy")
        ax4.legend()
        
        # 5: Energy with threshold peaks
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(result)
        ax5.plot(rPeaks, result[rPeaks], 'x', color='red')
        ax5.set_title("5: Integrated with Threshold Peaks")
        ax5.set_xlabel("Sample")
        ax5.set_ylabel("Energy")
        
        # 6: QRS points
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.plot(filtered)
        if len(rList):
            ax6.plot(rList, filtered[rList], 'o', color='red', label='R')
            ax6.plot(qList, filtered[qList], 'v', color='green', label='Q')
            ax6.plot(sList, filtered[sList], '^', color='blue', label='S')
        ax6.set_title("6: Fiducial Points (Q, R, S)")
        ax6.set_xlabel("Sample")
        ax6.set_ylabel("Amplitude")
        ax6.legend()
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
    
    def plot_dct_comparison(self):
        frame = self.tab_frames[2]
        
        for widget in frame.winfo_children():
            widget.destroy()
        
        if len(self.firstDct) < 2:
            return
        
        fig = Figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(121)
        ax1.plot(self.firstDct[0])
        ax1.set_title(self.dctNames[0])
        ax1.set_xlabel("Coefficient Index")
        ax1.set_ylabel("DCT Value")
        
        ax2 = fig.add_subplot(122)
        ax2.plot(self.firstDct[1])
        ax2.set_title(self.dctNames[1])
        ax2.set_xlabel("Coefficient Index")
        ax2.set_ylabel("DCT Value")
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
    
    def test_signal(self):
        testFile = "Test signal.txt"
        
        if not os.path.exists(testFile):
            messagebox.showerror("Error", f"Test file {testFile} not found!")
            return
        
        if not self.globalFeatures:
            messagebox.showwarning("Warning", "Please analyze ECG files first!")
            return
        
        testSignal = loadSignal(testFile)
        filteredTest = butterBandpassFilter(testSignal, 1, 40, self.fs)
        result = movingEnergy(filteredTest, self.fs)
        
        threshold = 60000
        thresholdPeaks, _ = find_peaks(result, height=threshold)
        
        if len(thresholdPeaks) == 0:
            messagebox.showerror("Error", "No peaks found in test signal!")
            return
        
        p = thresholdPeaks[0]
        searchWindow = round(0.1 * self.fs)
        startR = max(0, p - searchWindow)
        endR = min(len(filteredTest), p + searchWindow)
        
        trueR = startR + np.argmax(filteredTest[startR:endR])
        
        trueQ = trueR
        while trueQ > 0 and filteredTest[trueQ - 1] < filteredTest[trueQ]:
            trueQ -= 1
        
        trueS = trueR
        while trueS < len(filteredTest) - 1 and filteredTest[trueS + 1] < filteredTest[trueS]:
            trueS += 1
        
        segmentStart = max(0, trueR - round(0.2 * self.fs))
        segmentEnd = min(len(filteredTest), trueR + round(0.2 * self.fs))
        testSegment = filteredTest[segmentStart:segmentEnd]
        
        testMean, testStd, testEnergy = extractDwtFeatures(testSegment)
        
        # Find best match
        bestMatch = None
        minDiff = float('inf')
        resultsTable = []
        
        for item in self.globalFeatures:
            diff = abs(testMean - item["dwtMean"])
            resultsTable.append((item["signalName"], item["dwtMean"], diff))
            
            if diff < minDiff:
                minDiff = diff
                bestMatch = item["signalName"]
        
        resultsTable.sort(key=lambda x: x[2])
        
        # Display results
        result_text = f"\nTEST SIGNAL FEATURES\n"
        result_text += f"="*50 + "\n"
        result_text += f"Mean   : {testMean}\n"
        result_text += f"Std    : {testStd}\n"
        result_text += f"Energy : {testEnergy}\n"
        result_text += f"\nRESULTS\n"
        result_text += f"-"*50 + "\n"
        for name, meanVal, diff in resultsTable:
            result_text += f"{name:20} | {meanVal:12.6f} | {diff:.6f}\n"
        result_text += f"\nFINAL RESULT: {bestMatch}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_text)
        
        # Plot test signal
        self.plot_test_signal(testSignal, filteredTest, result, thresholdPeaks, trueQ, trueR, trueS)
        
        messagebox.showinfo("Test Result", f"Best Match: {bestMatch}")
    
    def plot_test_signal(self, ecg, filtered, result, rPeaks, q, r, s):
        frame = self.tab_frames[3]
        
        for widget in frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(15, 10))
        fig.suptitle("Test Signal Analysis", fontsize=16, fontweight='bold')
        
        # 1: Raw
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(ecg)
        ax1.set_title("1: Raw Test Signal")
        
        # 2: Filtered
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(filtered)
        ax2.set_title("2: Filtered Signal")
        
        # 3: Energy
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(result)
        ax3.set_title("3: Moving Energy")
        
        # 4: Peaks
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(result)
        ax4.plot(rPeaks, result[rPeaks], 'ro')
        ax4.set_title("4: R Peaks Detection")
        
        # 5: Energy with threshold
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(result)
        ax5.plot(rPeaks, result[rPeaks], 'x', color='red')
        ax5.set_title("5: Threshold Peaks")
        
        # 6: QRS
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.plot(filtered)
        ax6.plot(r, filtered[r], 'o', color='red', label='R')
        ax6.plot(q, filtered[q], 'v', color='green', label='Q')
        ax6.plot(s, filtered[s], '^', color='blue', label='S')
        ax6.set_title("6: QRS Detection")
        ax6.legend()
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
    
    def clear_plots(self):
        for i in range(4):
            frame = self.tab_frames[i]
            for widget in frame.winfo_children():
                widget.destroy()
        self.results_text.delete(1.0, tk.END)
        self.globalFeatures = []
        self.firstDct = []
        self.dctNames = []


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGAnalyzerApp(root)
    root.mainloop()