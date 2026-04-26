# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:49:19 2026

@author: Agustin 0. Umedez
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

class Data(object):
    """
    mode=0: 'muchas mediciones de 1 magnitud'
    mode=1: 'conjunto de mediciones x e y (fit)'
    """
   
    def __init__(self, data, mode=0):
        self.mode = mode
        self.data = data
        
    # def __add__(self, new_data):
        
    
    def mean(self):
        if self.mode:
            return np.mean(self.data[0]), np.mean(self.data[1])
        else:
            return np.mean(self.data)
    
    def std(self):
        if self.mode:
            return np.std(self.data[0]), np.std(self.data[1])
        else:
            return np.std(self.data)
    
    def linear_fit(self, stats=False):
        try:
            xs = np.linspace(self.data[0][0], self.data[0][-1])
        except:
            print("So' un wachin, te falta eje x.")
            return
        params, cov = np.polyfit(self.data[0], self.data[1], deg=1, cov=True)
        val = np.polyval(params, xs)
        
        if stats:
            m, p = params
            sdv_m = np.sqrt( cov[0][0] )
            sdv_p = np.sqrt( cov[1][1] )
            return (m, sdv_m), (p, sdv_p), val
        
        return params, val
    
    def k_factor(self, sigmas):
        # factor de cobertura
        conf = {
            1: 0.68,
            2: 0.95,
            3: 0.997
        }
        if self.mode:
            k = st.t.ppf(conf[sigmas], df = len(self.data[0])-1)
        else:
            k = st.t.ppf(conf[sigmas], df = len(self.data)-1)
        return k
    

class DigitalData(Data):
    
    def __init__(self, data, mode=0):
        super().__init__(data, mode)
    
    def ucty_digital(self, acc, rnge, res, lin, temp):
        """
        Se asume:
        - acc: [valor porcentual, valor porcentual]
        - rnge: rango
        - res: resolución
        - lin: [valor porcentual, valor porcentual]
        - temp: [valor porcentual, valor porcentual, temperatura]
        """
        
        if self.mode:
            y = self.data
        else:
            y = self.data.mean()
            
        # if (data := self.data) and len(data.size)==1:
        #     y = data.mean()
        # else:
        #     y = np.array([data[0], data[1]])
            
        ucty_acc = (acc[0]*y + acc[1]*rnge)/np.sqrt(3)
        ucty_lin = (lin[0]*y + lin[1]*rnge)/np.sqrt(3)
        ucty_res = (res*rnge/2)/np.sqrt(3)
        ucty_temp = (temp[0]*y + temp[1]*rnge)*abs(temp[2] - 23)/np.sqrt(3)
        
        ucty = np.sqrt(ucty_acc**2 + ucty_res**2 + ucty_temp**2 + ucty_lin**2)
        return ucty
    
    def quick_ucty_mean(self, *args):
        # acc, rnge, res, lin, temp = *args
        std_x = self.std()
        uB_x = self.ucty_digital(*args)
        ucty_x = np.sqrt(std_x**2 + uB_x**2)
        return ucty_x
    
    def quick_ucty_fit(self, M, P, *args):
        # acc, rnge, res, lin, temp = *args
        m, std_m = M
        p, std_p = P
        
        x, y = self.data
        
        uB_i, uB_o = self.ucty_digital(*args)
        
        uB2_m = np.sum( ((m - y)/(x**2)*uB_i)**2 + (1/x*uB_o)**2 )
        uB_m = np.sqrt(uB2_m)
        
        uB2_p = np.sum( (-p*uB_i)**2 + (1*uB_o)**2 )
        uB_p = np.sqrt(uB2_p)
        
        uC_m = np.sqrt( std_m**2 + uB_m**2 )
        uC_p = np.sqrt( std_p**2 + uB_p**2 )
        return uC_m, uC_p
    
    def fast(self, *args, sigmas=1):
        k = self.k_factor(sigmas)
        
        if self.mode:
            M, P, val = self.linear_fit(stats=True)
            uC_m, uC_p = self.quick_ucty_fit(M, P, *args)
            return (M[0], k*uC_m), (P[0], k*uC_p), val
        else:
            x = self.mean()
            ucty_x = self.quick_ucty_mean(*args)
            return x, k*ucty_x

def graf_doble(datos, ajuste=[], ejes=['x', ['y1', 'y2']]):
    """
    Para 'datos' y 'ajuste':
    - datos[0]: superior
    - datos[1]: inferior
    - datos[0][0]: x
    - datos[0][1]: y
    - datos[0][2]: label
    
    Para 'ejes':
    - ejes[0]: x
    - ejes[1][0]: y superior
    - ejes[1][1]: y inferior
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    if ajuste:
        ax1.plot(ajuste[0][0], ajuste[0][1], 'r', label=ajuste[0][2])
        ax2.plot(ajuste[1][0], ajuste[1][1], 'r', label=ajuste[1][2])
    
    ax1.plot(datos[0][0], datos[0][1], 'bo', markersize=6, label=datos[0][2])
    ax1.set_ylabel(ejes[1][0])
    ax1.grid(True)
    ax1.legend(fontsize=13)
    
    ax2.plot(datos[1][0], datos[1][1], 'b^', markersize=6, label=datos[1][2])
    ax2.set_xlabel(ejes[0])
    ax2.set_ylabel(ejes[1][0])
    ax2.grid(True)
    ax2.legend(fontsize=13)
    
    plt.tight_layout()
    plt.show()
    return

def graf_simple(datos, ajuste=[], ejes=['x', 'y']):
    """
    Para 'datos' y 'ajuste':
    - datos[0]: x
    - datos[1]: y
    - datos[2]: label
    
    Para 'ejes':
    - ejes[0]: x
    - ejes[1]: y
    """
    plt.figure()
    
    if ajuste:
        plt.plot(ajuste[0], ajuste[1], 'r', label=ajuste[2])
    plt.plot(datos[0], datos[1], 'bo', markersize=6, label=datos[2])
    
    plt.xlabel(ejes[0])
    plt.ylabel(ejes[1])
    
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    return


