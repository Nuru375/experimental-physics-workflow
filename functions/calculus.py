import numpy as np
import integrals

def euler(f, inter, y0, L):
  """
  Método de Euler para resolver
  y’ = f(t,y) en [t0,TF]
  y(t0) = y0
  Usando L pasos
  y0 puede ser vectorial o escalar
  """
  t = np.linspace(inter[0], inter[1], L + 1)
  h = (inter[1] - inter[0]) / L
  # reservamos lugar en memoria para y
  y = np.zeros((len(y0), L + 1))
  y[:, 0] = y0
  for n in range(L):
    y[:, n + 1] = y[:, n] + h * f(t[n], y[:, n])
  return t, y.T

def rk2(f, inter, y0, L):
  """
  Método de Runge Kutta de orden 2 para resolver
  y’ = f(t,y) en [t0,TF]
  y(t0) = y0
  Usando L pasos
  y0 puede ser vectorial o escalar
  """
  t = np.linspace(inter[0], inter[1], L + 1)
  h = (inter[1] - inter[0]) / L
  # reservamos lugar en memoria para y
  y = np.zeros((len(y0), L + 1))
  y[:, 0] = y0
  for n in range(L):
      k1=h*f(t[n], y[:, n])
      k2=h*f(t[n+1], y[:, n]+k1)
      y[:, n + 1] = y[:, n] + 0.5 * (k1+k2)
  return t, y.T

def rk4(f, inter, y0, L):
  """
  Método de Runge Kutta de orden 4 para resolver
  y’ = f(t,y) en [t0,TF]
  y(t0) = y0
  Usando L pasos
  y0 puede ser vectorial o escalar
  """
  t = np.linspace(inter[0], inter[1], L + 1)
  h = (inter[1] - inter[0]) / L
  # reservamos lugar en memoria para y
  y = np.zeros((len(y0), L + 1))
  y[:, 0] = y0
  for n in range(L):
      k1=h*f(t[n], y[:, n])
      k2=h*f(t[n]+h/2, y[:, n]+k1/2)
      k3=h*f(t[n]+h/2, y[:, n]+k2/2)
      k4=h*f(t[n+1], y[:, n]+k3)
      y[:, n + 1] = y[:, n] + (k1+2*k2+2*k3+k4)/6
  return t, y.T

