import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import scipy.integrate as integrate
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import butter, filtfilt

# ---------------- Filter Function ----------------
def ButterBandpassFilter(InputSignal, LowCutoff, HighCutoff, SamplingRate, Order):
    Nyq = 0.5 * SamplingRate
    Low = LowCutoff / Nyq
    High = HighCutoff / Nyq
    Numerator, Denominator = butter(Order, [Low, High], btype='band')
    Filtered = filtfilt(Numerator, Denominator, InputSignal)
    return Filtered

# ---------------- Global Variables ----------------
FinalFeaturesTable = None
PatientNames = None

SamplingRate = 176
LowCut = 0.5
HighCut = 20
FilterOrder = 5


# ---------------- Upload Training Data ----------------
def load_training_data():
    global FinalFeaturesTable, PatientNames

    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if not file_path:
        return

    DataFrame = pd.read_excel(file_path, header=None)
    PatientNames = DataFrame.iloc[-1, :].values
    RawSignalsMatrix = DataFrame.iloc[:-1, :].T.values.astype(float)

    FilteredSignals = []

    for signal in RawSignalsMatrix:
        NoDC = signal - np.mean(signal)
        Filtered = ButterBandpassFilter(NoDC, LowCut, HighCut, SamplingRate, FilterOrder)
        FilteredSignals.append(Filtered)

    FinalFeaturesList = []

    for I in range(len(FilteredSignals)):
        CurrentSignal = FilteredSignals[I]

        MeanValue = np.mean(CurrentSignal)
        StdValue = np.std(CurrentSignal)
        MaxPeak = np.max(CurrentSignal)
        Area = integrate.trapezoid(np.abs(CurrentSignal))

        ArModel = AutoReg(CurrentSignal, lags=2).fit()
        ArCoeffs = ArModel.params[0:]

        FinalFeaturesList.append([
            PatientNames[I],
            MeanValue,
            StdValue,
            MaxPeak,
            Area,
            ArCoeffs[0],
            ArCoeffs[1],
            ArCoeffs[2]
        ])

    FinalFeaturesTable = pd.DataFrame(FinalFeaturesList, columns=[
        'Patient Name',
        'Mean',
        'Std',
        'MaxPeak',
        'Area',
        'Ar1',
        'Ar2',
        'Ar3'
    ])

    messagebox.showinfo("Success", "Training Data Loaded Successfully")


# ---------------- Upload Test Signal ----------------
def load_test_signal():

    global FinalFeaturesTable

    if FinalFeaturesTable is None:
        messagebox.showwarning("Warning", "Please load training data first")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return

    TestSignalRaw = np.loadtxt(file_path)

    TestNoDC = TestSignalRaw - np.mean(TestSignalRaw)
    TestFiltered = ButterBandpassFilter(TestNoDC, LowCut, HighCut, SamplingRate, FilterOrder)

    TestMean = np.mean(TestFiltered)
    TestStd = np.std(TestFiltered)
    TestMax = np.max(TestFiltered)
    TestArea = integrate.trapezoid(np.abs(TestFiltered))

    TestArModel = AutoReg(TestFiltered, lags=2).fit()
    TestArCoeffs = TestArModel.params[0:]

    TestVector = np.array([
        TestMean,
        TestStd,
        TestMax,
        TestArea,
        TestArCoeffs[0],
        TestArCoeffs[1],
        TestArCoeffs[2]
    ])

    TrainingMatrix = FinalFeaturesTable.iloc[:, 1:].values

    Distances = []

    for row in TrainingMatrix:
        d = np.linalg.norm(TestVector - row)
        Distances.append(d)

    idx = np.argmin(Distances)
    PredictedPatient = FinalFeaturesTable.iloc[idx, 0]

    result_label.config(text=f"Prediction: {PredictedPatient}")


# ---------------- GUI ----------------
root = tk.Tk()
root.title("Signal Classification System")
root.geometry("400x250")

title = tk.Label(root, text="Signal Identification System", font=("Arial", 16))
title.pack(pady=20)

btn_train = tk.Button(root, text="Upload Training Data (Excel)", command=load_training_data)
btn_train.pack(pady=10)

btn_test = tk.Button(root, text="Upload Test Signal (.txt)", command=load_test_signal)
btn_test.pack(pady=10)

result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 12))
result_label.pack(pady=20)

root.mainloop()