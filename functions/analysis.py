# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:49:19 2026

@author: Agustin 0. Umedez
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

def linear_fit(x, y, stats=False):
    xs = np.linspace(x[0], x[-1])
    params, cov = np.polyfit(x, y, 1, cov=True)
    val = np.polyval(params, xs)
    
    if stats:
        m, p = params
        sdv_m = np.sqrt( cov[0][0] )
        sdv_p = np.sqrt( cov[1][1] )
        return (m, sdv_m), (p, sdv_p), val
    
    return params, val

def ucty_digital(x, acc, rnge, res, lin, temp):
    """
    Se asume:
    - x: mejor estimación o np.array de datos
    - acc: [valor porcentual, valor porcentual]
    - rnge: rango
    - res: resolución
    - lin: [valor porcentual, valor porcentual]
    - temp: [valor porcentual, valor porcentual, temperatura]
    """
    
    ucty_acc = (acc[0]*x + acc[1]*rnge)/np.sqrt(3)
    ucty_lin = (lin[0]*x + lin[1]*rnge)/np.sqrt(3)
    ucty_res = (res*rnge/2)/np.sqrt(3)
    ucty_temp = (temp[0]*x + temp[1]*rnge)*abs(temp[2] - 23)/np.sqrt(3)
    
    ucty = np.sqrt( ucty_acc**2 + ucty_res**2 + ucty_temp**2 + ucty_lin**2 )
    return ucty

def k_factor(x, sigmas):
    #factor de cobertura
    conf = {
        1: 0.68,
        2: 0.95,
        3: 0.997
    }
    k = st.t.ppf(conf[sigmas], df = len(x)-1 )
    return k

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


