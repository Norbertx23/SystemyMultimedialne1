import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import os
from docx import Document
from docx.shared import Inches
from io import BytesIO

##########################################
### Settings #############################
##########################################

Only_Tests=False
bit_test=8
A = 87.6

DPCM_n=5
DPCM_predictor=np.mean

OutputRaportFile = "raport07.docx"
BaseDir = os.path.dirname(os.path.abspath(__file__))
OutputFolder = os.path.join(BaseDir, "out")

##########################################
### Data Set #############################
##########################################

AudioDir = os.path.join(BaseDir, "SING")

SingFiles=[
'sing_high1.wav',
'sing_low1.wav',
'sing_medium1.wav',


] # list of file names of with Singing Voice

##########################################
### Functions to  ########################
##########################################

def Kwant(data,bit):
    d = 2 ** bit - 1

    if np.issubdtype(data.dtype, np.floating):
        v_min = -1
        v_max = 1
    else:
        v_min = np.iinfo(data.dtype).min
        v_max = np.iinfo(data.dtype).max

    DataF = data.astype(float)
    if v_max - v_min == 0:
        return data
    DataF = (DataF - v_min) / (v_max - v_min)
    DataF = DataF * d
    DataF = np.round(DataF)
    DataF /= d
    DataF = DataF * (v_max - v_min) + v_min

    return DataF.astype(data.dtype)


def A_law_compress(x):
    abs_x = np.abs(x)
    y = np.zeros(x.shape)

    idx = abs_x < (1 / A)

    y[idx] = (A * abs_x[idx]) / (1 + np.log(A))
    y[np.logical_not(idx)] = (1 + np.log(A * abs_x[np.logical_not(idx)])) / (1 + np.log(A))

    return np.sign(x) * y


def A_law_decompress(x):
    abs_x = np.abs(x)
    y = np.zeros(x.shape)

    threshold = 1 / (1 + np.log(A))
    idx = abs_x < threshold

    y[idx] = (abs_x[idx] * (1 + np.log(A))) / A
    y[np.logical_not(idx)] = np.exp(abs_x[np.logical_not(idx)] * (1 + np.log(A)) - 1) / A

    return np.sign(x) * y


def mu_law_compress(x):
    mu = 255.0
    abs_x = np.abs(x)

    y = np.sign(x) * (np.log(1 + mu * abs_x) / np.log(1 + mu))

    return y

def mu_law_decompress(x):
    mu = 255.0
    abs_x = np.abs(x)

    y = np.sign(x) * (1 / mu) * ((1 + mu) ** abs_x - 1)

    return y

def DPCM_compress(x,bit):
    y=np.zeros(x.shape)
    e=0
    for i in range(0,x.shape[0]):
        y[i]=Kwant(x[i]-e,bit)
        e+=y[i]
    return y

def DPCM_decompress(x):
    y = np.zeros(x.shape)
    e = 0

    for i in range(0, x.shape[0]):
        y[i] = x[i] + e

        e = y[i]
    return y

def DPCM_compress_pred(x,bit,n,predictor=np.mean): 
    y=np.zeros(x.shape)
    xp=np.zeros(x.shape)
    e=0
    for i in range(0,x.shape[0]):
        y[i]=Kwant(x[i]-e,bit)
        xp[i]=y[i]+e
        idx=(np.arange(i-n,i,1,dtype=int)+1)
        idx=np.delete(idx,idx<0)
        e=predictor(xp[idx])
    return y

def DPCM_decompress_pred(x,n,predictor=np.mean):
    y = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        y[i] = x[i] + e

        idx = (np.arange(i - n, i, 1, dtype=int) + 1)
        idx = np.delete(idx, idx < 0)

        if idx.size == 0:
            e = 0
        else:
            e = predictor(y[idx])

    return y

##########################################
### Main Program  ########################
##########################################


document = Document()
if not Only_Tests:
    # generate raport
    document.add_heading('Report',0) # tworzenie nagłówków druga wartość to poziom nagłówka 
    document.add_paragraph("Autor: Norbert Świstak")
    document.add_section()
    document.add_heading('Wykresy testujące działanie algorytmów',1)

x=np.linspace(-1,1,1000)
y=0.9*np.sin(np.pi*x*4)

x_alaw_comp=A_law_compress(x)
x_alaw_comp=Kwant(x_alaw_comp,bit_test)
x_alaw_decomp=A_law_decompress(x_alaw_comp)

x_mulaw_comp=mu_law_compress(x)
x_mulaw_comp=Kwant(x_mulaw_comp,bit_test)
x_mulaw_decomp=mu_law_decompress(x_mulaw_comp)

y_alaw_decomp =A_law_decompress(Kwant(A_law_compress(y),bit_test))
y_mulaw_decomp =mu_law_decompress(Kwant(mu_law_compress(y),bit_test))

dpcm_c=DPCM_compress(y,bit_test)
dpcm_dec=DPCM_decompress(dpcm_c)
dpcm_c_p=DPCM_compress_pred(y,bit_test,n=DPCM_n,predictor=DPCM_predictor)
dpcm_dec_p=DPCM_decompress_pred(dpcm_c_p,n=DPCM_n,predictor=DPCM_predictor)

f1,axs=plt.subplots(1,2,num=1,figsize=(6,6)) 
f1.suptitle(f"Test kompresji law dla {bit_test} bitów")
axs[0].plot(x,x_alaw_comp,label="a_law")
axs[0].plot(x,x_mulaw_comp,label="mu_law")
axs[0].set_title("Sygnał po kompresji")
axs[0].legend()

axs[1].plot(x,x_alaw_decomp,label="a_law")
axs[1].plot(x,x_mulaw_decomp,label="mu_law")
axs[1].set_title("Sygnał po dekompresji")
axs[1].legend()

f2,axs=plt.subplots(5,1,num=2,figsize=(8,6)) 
f2.suptitle(f"Test dekompresji dla {bit_test} bitów")
axs[0].plot(x,y,label="Sygnał bazowy")
axs[0].set_title("Sygnał bazowy")

axs[1].plot(x,y_alaw_decomp,label="Sygnał po kompresji A-law")
axs[1].legend()

axs[2].plot(x,y_mulaw_decomp,label="Sygnał po kompresji mu-law")
axs[2].legend()

axs[3].plot(x,dpcm_dec,label="Sygnał po kompresji DPCM bez predykcji")
axs[3].legend()

axs[4].plot(x,dpcm_dec_p,label="Sygnał po kompresji DPCM z predykcją")
axs[4].legend()


if Only_Tests:
    plt.show()
else:
    memfile = BytesIO() 
    f1.savefig(memfile)
    document.add_picture(memfile, width=Inches(6)) # set document size
    memfile.close()
    f1.clf()
    memfile = BytesIO() 
    f2.savefig(memfile)
    document.add_picture(memfile, width=Inches(6)) # set document size
    memfile.close()
    f2.clf()  
    document.add_section()
    document.add_heading("Obserwacje na podstawie odsłuchanych plików ",1)
    document.add_paragraph("Tu proszę odpowiedzieć własnymi słowami na zadanie 2.1")
    document.add_heading("Zadanie 2.2",2)
    document.add_paragraph("Tu proszę zamieścić treść dla zadania 2.2")
    document.add_heading("Zadanie 2.3",2)
    document.add_paragraph("Tu proszę zamieścić treść dla zadania 2.3 (może być tabelka). Uwaga jak nie jesteście w stanie rozpoznać zawrtości nie musice słuchać dla niższych wartości bitowych")
    document.add_section()
    document.add_heading("Podsumowanie i Wnioski",1)
    document.add_paragraph("Tu proszę krótko podsumować wszystko")
    document.save(OutputRaportFile)

    os.makedirs(OutputFolder, exist_ok=True)
    for file in SingFiles:
        Signal, Fs = sf.read(os.path.join(AudioDir,file), dtype='float32') 
        sfile=file.split(os.sep)[-1].split('.')
        for bit in [8,7,6,5,4,3,2]:
            y_alaw_decomp =A_law_decompress(Kwant(A_law_compress(Signal),bit))
            y_mulaw_decomp =mu_law_decompress(Kwant(mu_law_compress(Signal),bit))

            dpcm_c=DPCM_compress(Signal,bit)
            dpcm_dec=DPCM_decompress(dpcm_c)
            dpcm_c_p=DPCM_compress_pred(Signal,bit,n=DPCM_n,predictor=DPCM_predictor)
            dpcm_dec_p=DPCM_decompress_pred(dpcm_c_p,n=DPCM_n,predictor=DPCM_predictor)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_A_LAW_{bit}b.wav"),data=y_alaw_decomp,samplerate=Fs)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_mu_LAW_{bit}b.wav"),data=y_mulaw_decomp,samplerate=Fs)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_DPCM_bp_{bit}b.wav"),data=dpcm_dec,samplerate=Fs)
            sf.write(os.path.join(OutputFolder,f"{sfile[0]}_DPCM_zp_{bit}b.wav"),data=dpcm_dec_p,samplerate=Fs)

            