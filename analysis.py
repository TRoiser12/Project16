import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def function(v, a, b):
    return a*v**b

s_m = 0.5777    #sphere mass in kg
s_d = 0.05064   #sphere diameter in m

dh = 0.2        #error in height in cm
dm = 1e-4       #error in sphere mass in kg

h_dat, t_dat = np.loadtxt('rawData_1.csv', skiprows = 1, unpack = True, delimiter = ',')

h = h_dat.reshape(-1, 5)[:,0]                                                                   #change in height in cm
t = t_dat.reshape(-1, 5).mean(axis=1)                                                           #contact time in μs
t_trials = t_dat.reshape(-1, 5)
dt = [float((max(t_trials[x]) - min(t_trials[x]))/2) for x in range(0, 15)]                     #uncertainty in time in μs
E = [(s_m * 9.81 * x / 100) for x in h]                                                         #energy in J
dE = [(((s_m+dm)*9.81*(x+dh)-(s_m-dm)*9.81*(x-dh))/2)/100 for x in h]                           #uncertainty in energy in J
v = [float(np.sqrt((2*x)/s_m)) for x in E]                                                      #Speed of ball in m/s
dv = [((np.sqrt((2*(x+d))/(s_m-dm))-np.sqrt((2*(x-d))/(s_m+dm)))/2)for x, d in zip(E, dE)]      #uncertainty in speed in m/s

popt, pcov = optimize.curve_fit(function, v, t)

v_f = np.linspace(0.2, v[-1], 1000)
t_f = function(v_f, popt[0], popt[1])

plt.errorbar(v, t, yerr= dt, xerr = dv, linestyle="none", color = "b")
plt.loglog(v_f, t_f, label = f"t1 = ({popt[0]:.1f}v ± {pcov[0][0]:.1f}) ^ ({popt[1]:.2f} ± {pcov[1][1]:.2f})", color = "b")

h_dat, t_dat = np.loadtxt('rawData_2.csv', skiprows = 1, unpack = True, delimiter = ',')

h = h_dat.reshape(-1, 5)[:,0]                                                                   #change in height in cm
t = t_dat.reshape(-1, 5).mean(axis=1)                                                           #contact time in μs
t_trials = t_dat.reshape(15, 5)
dt = [float((max(t_trials[x]) - min(t_trials[x]))/2) for x in range(0, 15)]                     #uncertainty in time in μs
E = [(s_m * 9.81 * x / 100) for x in h]                                                         #energy in J
dE = [(((s_m+dm)*9.81*(x+dh)-(s_m-dm)*9.81*(x-dh))/2)/100 for x in h]                           #uncertainty in energy in J
v = [float(np.sqrt((2*x)/s_m)) for x in E]                                                      #Speed of ball in m/s
dv = [((np.sqrt((2*(x+d))/(s_m-dm))-np.sqrt((2*(abs(x-d)))/(s_m+dm)))/2)for x, d in zip(E, dE)]      #uncertainty in speed in m/s
print(E)
print(dE)
popt, pcov = optimize.curve_fit(function, v, t)

v_f = np.linspace(0.1, v[-1], 1000)
t_f = function(v_f, popt[0], popt[1])

plt.errorbar(v, t, yerr= dt, xerr = dv, linestyle="", color = "r")
plt.loglog(v_f, t_f, label = f"t2 = ({popt[0]:.1f}v ± {pcov[0][0]:.1f}) ^ ({popt[1]:.2f} ± {pcov[1][1]:.2f})", linestyle="--", color="r")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Contact time (μs)")
plt.legend()
plt.show()