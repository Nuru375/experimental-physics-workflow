# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 21:42:01 2025

Función normalize_audio
"""

import numpy as np

def normalize_audio(y):
    """
    Normaliza cualquier señal al rango [-1, 1].

    Args:
        y (np.ndarray): Array de audio (puede ser int16, int32, float32, etc.)

    Returns:
        y_norm (np.ndarray): Señal de audio normalizada como float32 en [-1, 1]
    """

    # Convertir a float32 si no lo es
    y = y.astype(np.float32)

    # Evitar división por cero si la señal es silenciosa
    peak = np.max(np.abs(y))
    if peak == 0:
        return y  # ya es todo cero
    else:
        return y / peak


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Aplica un filtro pasa-bajos a una señal.

    Args:
        data (np.ndarray): Señal como un array numpy.
        cutoff_freq (float): Frecuencia de corte del filtro en Hz.
        fs (float): Frecuencia de muestreo de la señal en Hz.
        order (int): Orden del filtro (por defecto es 5).

    Returns:
        filtered (np.ndarray): Señal filtrada.
    """
    from scipy.signal import butter, filtfilt
    
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered


def butter_highpass_filter(data, cutoff, fs, order=4):
    """
    Aplica un filtro pasa-altos a una señal.

    Parametros
    ----------
    data: ndarray
        Señal de audio.
        Debe estar en formato array de numpy.
    cutoff_freq: float
        Frecuencia de corte del filtro en Hz.
    fs: float
        Frecuencia de muestreo de la señal en Hz.
    order: int, opcional
        Orden del filtro.
        Por defecto es 5.

    Retorna
    -------
    filtered: ndarray
        Señal filtrada.
    """
    from scipy.signal import butter, filtfilt
    
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered


def find_peak_freq_old(Fs, y, inter, graf=0, inter_f=list()):
    """
    Grafica el espectro de frecuencia de una señal con sus picos.

    Parametros
    ----------
    

    Retorna
    -------
    peak_freq: float
        Pico de frecuencia.
    """
    import matplotlib.pyplot as plt
    from scipy.signal.windows import hann
    from scipy.fft import rfft, rfftfreq
    
    # Seleccionamos el segmento de interés
    tinicio = inter[0]   # segundos
    tifany = inter[1]    # segundos
    start = round(tinicio * Fs)
    end = round(tifany * Fs)
    yy = y[start:end]
    
    # if graf != 0:
    #     plt.figure()
    #     plt.plot(np.linspace(tinicio,tifany,len(yy)), yy)
    #     plt.xlabel('tiempo [s]')
    #     plt.grid(True)
    #     plt.show()

    window = hann(len(yy))
    yy = yy * window

    n = len(yy)
    frequencies = rfftfreq(n, d=1/Fs)  # ejes de frecuencia
    spectrum = np.abs(rfft(yy))        # módulo de la FFT

    peak_idx = np.argmax(spectrum)
    peak_freq = frequencies[peak_idx]
    
    if graf != 0:
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, spectrum, 'b*', label="Espectro")
        plt.plot(peak_freq, spectrum[peak_idx], 'ro', label=f"Pico: {peak_freq:.2f} Hz")
        plt.xlabel("Frecuencia [Hz]",fontsize=18)
        plt.ylabel("Magnitud del espectro",fontsize=18)
        if inter_f is not []:
            plt.xlim(inter_f[0],inter_f[1])
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()
    
    print(f"Frecuencia pico en el segmento: {peak_freq:.2f} Hz")
    
    return peak_freq


def find_peak_freq(Fs, y, inter, method='gaussian', filt='moving', graf=0, inter_f=list()):
    """
    Grafica el espectro de frecuencia de una señal con sus picos.

    Parametros
    ----------
    

    Retorna
    -------
    peak_freq: float
        Pico de frecuencia.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal.windows import hann
    from scipy.ndimage import uniform_filter1d
    from scipy.fft import rfft, rfftfreq
    
    # Aislamos el segmento de interés
    tinicio = inter[0]   # segundos
    tifany = inter[1]    # segundos
    start = round(tinicio * Fs)
    end = round(tifany * Fs)
    y2 = y[start:end]
    
    # if graf != 0:
    #     plt.figure()
    #     plt.plot(np.linspace(tinicio,tifany,len(yy)), yy)
    #     plt.xlabel('tiempo [s]')
    #     plt.grid(True)
    #     plt.show()
    
    # Método de filtrado
    if filt == 'hann':
        window = hann(len(y2))
        yy = y2 * window
    elif filt == 'moving':
        yy = uniform_filter1d(y2, 5)
        
    if method == 'psd_voigt': # No funciona
        from scipy.signal import savgol_filter
        
        yy = savgol_filter(y2 * hann(len(y2)), 11, 3)
        n = len(yy)
        frequencies = rfftfreq(n, d=1/Fs)  # ejes de frecuencia
        spectrum = np.abs(rfft(yy))        # módulo de la FFT
    else:
        n = len(yy)
        frequencies = rfftfreq(n, d=1/Fs)  # ejes de frecuencia
        spectrum = np.abs(rfft(yy))        # módulo de la FFT

    # Índice del pico inicial
    peak_idx = np.argmax(spectrum)

    # Ajuste
    if method == 'quadratic' and 1 <= peak_idx < len(spectrum) - 1:
        # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        y0, ym1, yp1 = spectrum[peak_idx], spectrum[peak_idx-1], spectrum[peak_idx+1]
        delta = (yp1 - ym1) / (2*(2*y0 - yp1 - ym1))
        refined_bin = peak_idx + delta
        peak_freq = refined_bin*Fs/n
        err = abs(peak_freq - frequencies[peak_idx])
    else:
        # Selección de vecinos para ajuste (a cada lado)
        fit_window = 5
        fit_range = slice(max(
            0, 
            peak_idx - fit_window), 
            min(len(spectrum), peak_idx + fit_window + 1)
            )
        x_fit = frequencies[fit_range]
        y_fit = spectrum[fit_range]
        
        # Parámetros iniciales
        A0, b0 = np.max(y_fit), x_fit[np.argmax(y_fit)]
        
        if method == "gaussian":
            def gaussian(x, A, a, b):
                return A * np.exp(-a * (x - b)**2)
            
            a0 = 1.0 / (0.5 * (x_fit[-1] - x_fit[0]))**2
            try:
                popt, cov = curve_fit(
                    gaussian,
                    x_fit, y_fit,
                    p0=[A0, a0, b0]
                )
                peak_freq = popt[2]
                err = np.sqrt(cov[2,2])
            except RuntimeError:
                peak_freq = frequencies[peak_idx]  # fallback
                err = abs(peak_freq - np.mean([frequencies[peak_idx-1], frequencies[peak_idx+1]]))
        elif method == 'lorentz':
            def lorentz(x, A, b, gamma):
                return A / (1 + ((x - b) / (gamma / 2))**2)
            
            gamma0 = (x_fit[-1] - x_fit[0]) / 2
            try:
                popt, cov = curve_fit(
                    lorentz,
                    x_fit, y_fit,
                    p0=[A0, b0, gamma0]
                )
                peak_freq = popt[1]
                err = np.sqrt(cov[1,1])
            except RuntimeError:
                peak_freq = frequencies[peak_idx]
                err = abs(peak_freq - np.mean([frequencies[peak_idx-1], frequencies[peak_idx+1]]))

        elif method == 'psd_voigt':
            def psd_voigt(x, A, b, w, eta):
                G = np.exp(-4*np.log(2)*(x - b)**2 / w**2)
                L = 1 / (1 + 4*(x - b)**2 / w**2)
                return A * (eta * L + (1 - eta) * G)
            
            w0, eta0 = (x_fit[-1] - x_fit[0]), 0.5
            try:
                popt, cov = curve_fit(
                    psd_voigt, 
                    x_fit, y_fit,
                    p0=[A0, b0, w0, eta0]
                )
                peak_freq = popt[1]
                err = np.sqrt(cov[1,1])
            except RuntimeError:
                peak_freq = frequencies[peak_idx]
                err = abs(peak_freq - np.mean([frequencies[peak_idx-1], frequencies[peak_idx+1]]))
        elif method == 'voigt':
            from scipy.special import voigt_profile
        
            def voigt(x, A, b, sigma, gamma):
                return A * voigt_profile(x - b, sigma, gamma)
        
            sigma0 = (x_fit[-1] - x_fit[0]) / 5
            gamma0 = sigma0
            try:
                popt, cov = curve_fit(
                    voigt,
                    x_fit, y_fit,
                    p0=[A0, b0, sigma0, gamma0]
                    )
                peak_freq = popt[1]
                err = np.sqrt(cov[1, 1])
            except RuntimeError:
                peak_freq = frequencies[peak_idx]
                err = abs(peak_freq - np.mean([frequencies[peak_idx-1], frequencies[peak_idx+1]]))
        else:
            peak_freq = frequencies[peak_idx]
            err = abs(peak_freq - np.mean([frequencies[peak_idx-1], frequencies[peak_idx+1]]))

    
    if graf != 0:
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, spectrum, 'b*', label="Espectro")
        if inter_f != []:
            plt.xlim(inter_f[0], inter_f[1])
            ff = np.linspace(inter_f[0], inter_f[1], 5*abs(inter_f[1]-inter_f[0]))
            if method == 'quadratic':
                # aux = np.polyfit([peak_idx-1,peak_idx,peak_idx+1], [ym1, y0, yp1], 2)
                # aux2 = lambda x: refined_bin - (x - peak_freq)**2
                x_pts = frequencies[peak_idx-1 : peak_idx+2]
                y_pts = spectrum[peak_idx-1 : peak_idx+2]
                coeffs = np.polyfit(x_pts, y_pts, 2)
                quad_fit = lambda x: coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
                peak_ampl = quad_fit(peak_freq)
                plt.plot(ff, quad_fit(ff), 'b', label="Ajuste Cuadrático")
                plt.plot(peak_freq, peak_ampl, 'ro', label=f"Pico: {peak_freq:.2f} Hz")
                plt.ylim(0,peak_ampl*1.2)
            elif method == 'gaussian':
                plt.plot(ff, gaussian(ff, *popt), 'b', label="Ajuste Gaussiano")
                plt.plot(peak_freq, gaussian(peak_freq, *popt), 'ro', label=f"Pico: {peak_freq:.2f} Hz")
            elif method == 'lorentz':
                plt.plot(ff, lorentz(ff, *popt), 'b', label="Ajuste Lorentziano")
                plt.plot(peak_freq, lorentz(peak_freq, *popt), 'ro', label=f"Pico: {peak_freq:.2f} Hz")
            elif method == 'psd_voigt':
                plt.plot(ff, psd_voigt(ff, *popt), 'b', label="Ajuste Pseudo-Voigt")
                plt.plot(peak_freq, psd_voigt(peak_freq, *popt), 'ro', label=f"Pico: {peak_freq:.2f} Hz")
            else:
                plt.plot(frequencies, spectrum, 'b-')
                plt.plot(peak_freq, spectrum[peak_idx], 'ro', label=f"Pico: {peak_freq:.2f} Hz")
        plt.xlabel("Frecuencia [Hz]", fontsize=18)
        plt.ylabel("Magnitud del espectro", fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()
    
    # print(f"Pico de frecuencia: {peak_freq:.2f} Hz")
    
    return peak_freq, err


    