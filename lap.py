#! /usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def binSearch(arr, want):
	start = 0
	end = len(arr)-1
	while end - start > 1:
		mid = start + (end-start)/2
		if(want == arr[mid]):
			return mid, mid
		if(want > arr[mid]):
			start = mid
		else:
			end = mid
	return start, end

class fakeFunction:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.start = x[0]
		self.end = x[-1]

	def getY(self, wantX):
		if(wantX < self.start or wantX > self.end):
			raise ValueError("Trying to retrieve x value out of bounds")
		smaller, bigger = binSearch(self.x, wantX)
		if(smaller == bigger):
			return self.y[smaller]
		xDiff = self.x[bigger]-self.x[smaller]
		yDiff = self.y[bigger] - self.y[smaller]
		ratio = (wantX-self.x[smaller])/xDiff
		return self.y[smaller] + yDiff*ratio

def fakeIntegral(func, start, stop, steps):
	tot = 0
	stepDiff = (stop-start)/ (steps-1.0)
	at = start+stepDiff;
	# print ("Step Diff", stepDiff)
	last = func(start)
	while at < stop:
		this = func(at)
		halfdiff = (this-last)/2.0
		mid = last + halfdiff
		# print ("at", at, "this", this, "mid", mid, "Add", mid*stepDiff)
		tot += (mid*stepDiff)
		at += stepDiff
		last = this
	return tot

def getFourierY(f, t1, t2, evStart, evStop, evNum):
	frequencies = np.linspace(evStart, evStop, evNum)
	real = []
	imag = []
	for freq in frequencies:
		realWeight, err = integrate.quad(lambda x: f(x, freq).real , t1, t2)
		imagWeight, err = integrate.quad(lambda x: f(x, freq).imag , t1, t2)
		realWeight = realWeight / (t2-t1)
		imagWeight = imagWeight / (t2-t1)
		real.append(realWeight)
		imag.append(imagWeight)
	return real, imag


def innerFourier (tFunc, omega):
	def f(time, freq):
		return tFunc(time)*np.e**(-omega*freq*1j*time)
	return f

def anitInnerFourier (tFunc, omega):
	def f(freq, time):
		return tFunc(freq)*np.e**(omega*freq*1j*time)
	return f

timeStart = 0
timeEnd = 1
# period = timeEnd - timeStart
period = 1
angularFrequency = (2*np.pi)/period
freqEnd = 100
freqStart = -freqEnd
numSteps = 999

wrapFreq = 1.0;
# timeFunction = lambda x: np.cos((2*np.pi)*1*x)
# timeFunction = lambda x: np.sin(((2*np.pi)/1)*x)
# timeFunction = lambda x: x
# timeFunction = lambda x: x**2
def timeFunction(x):
	mod = x%1
	if(mod < 0.2):
		return 0.0
	if(mod < 0.4):
		return 0.2
	if(mod < 0.6):
		return 0.4
	if(mod < 0.8):
		return 0.6
	# if(mod < 1):
	return 0.8

timeToFreq = innerFourier(timeFunction, angularFrequency);

timeSteps = np.linspace(timeStart, timeEnd, numSteps)
freqSteps = np.linspace(freqStart, freqEnd, numSteps)
# freqSteps = []
# i = freqStart
# while (i <= freqEnd):
# 	freqSteps.append(i)
# 	i+=1

fig, axs = plt.subplots(2,3)
ax1 = axs[0,0]
ax2 = axs[0,1]
ax3 = axs[0,2]
ax4 = axs[1,0]
ax5 = axs[1,1]
ax6 = axs[1,2]
ax1.axhline(y=0, color=(0,0,0,0.4))
ax1.axvline(x=0, color=(0,0,0,0.4))
ax2.axhline(y=0, color=(0,0,0,0.4))
ax2.axvline(x=0, color=(0,0,0,0.4))
ax3.axhline(y=0, color=(0,0,0,0.4))
ax3.axvline(x=0, color=(0,0,0,0.4))
ax4.axhline(y=0, color=(0,0,0,0.4))
ax4.axvline(x=0, color=(0,0,0,0.4))
ax5.axhline(y=0, color=(0,0,0,0.4))
ax5.axvline(x=0, color=(0,0,0,0.4))
ax6.axhline(y=0, color=(0,0,0,0.4))
ax6.axvline(x=0, color=(0,0,0,0.4))

res = list(map(lambda x: timeToFreq(x, wrapFreq), timeSteps))
x1 = list(map(lambda v: v.real, res))
y1 = list(map(lambda v: v.imag, res))

# y1 = list(map(eFunc, x1))
timeWrapLine, = ax1.plot(x1,y1)

timeWeightX, err = integrate.quad(lambda x: timeToFreq(x, wrapFreq).real, timeStart, timeStart + period)
timeWeightY, err = integrate.quad(lambda x: timeToFreq(x, wrapFreq).imag, timeStart, timeStart + period)
timeWeightCenter = ax1.scatter(timeWeightX, timeWeightY)

x2 = timeSteps
y2 = list(map(timeFunction, x2))
ax2.plot(x2, y2);

y3Real, y3Imag = getFourierY(timeToFreq, timeStart, timeStart+period, freqStart, freqEnd, numSteps)
ax3.plot(freqSteps, y3Real, label="Real")
ax3.plot(freqSteps, y3Imag, label="Imag")

fakeReal = fakeFunction(freqSteps, y3Real)
fakeReal.getY(1.6)
fakeImag = fakeFunction(freqSteps, y3Imag)

fakeFreq = lambda x: fakeReal.getY(x) + fakeImag.getY(x)*1j

freqToTime = anitInnerFourier(fakeReal.getY, angularFrequency);
imagFreqToTime = anitInnerFourier(fakeImag.getY, angularFrequency);

wrapTime = 0.5

res = list(map(lambda f: freqToTime(f, wrapTime), freqSteps))
x4 = list(map(lambda v: v.real, res))
y4 = list(map(lambda v: v.imag, res))
freqWrapLine, = ax4.plot(x4,y4)

freqWeightX = fakeIntegral(lambda f: freqToTime(f, wrapTime).real, freqStart, freqStart+period, 500)
freqWeightY = fakeIntegral(lambda f: freqToTime(f, wrapTime).imag, freqStart, freqStart+period, 500)
freqWeightCenter = ax4.scatter(freqWeightX, freqWeightY, color=(1,0,0,1))

y5Real = []
# y5Imag = []
for time in timeSteps:
	y5Real.append(fakeIntegral(lambda f: freqToTime(f, time).real, freqStart, freqEnd, 500))
	# y5Imag.append(fakeIntegral(lambda f: freqToTime(f, time).imag, freqStart, freqEnd, 100))
# y5Real, y5Imag = fakeIntegral(freqToTime, freqStart, freqEnd, timeStart, timeEnd, numSteps)
ax5.plot(timeSteps, y5Real, label="Real")
# ax5.plot(timeSteps, y5Imag, label="Imag")

y6Real = []

for ts in timeSteps:
	tot = 0
	for freq in range(freqStart, freqEnd+1):
		# if(freq < 0):
		# 	continue
		if(freq == 0.0):
			tot += fakeReal.getY(0.0)
			continue
		cosA = fakeReal.getY(freq)
		sinB = -1* fakeImag.getY(freq)
		# cosA = 2 *fakeReal.getY(freq)
		# sinB = -2* fakeImag.getY(freq)
		tot += (cosA * np.cos(ts*freq*angularFrequency))
		tot += sinB * np.sin(ts*freq*angularFrequency)
	y6Real.append(tot)
ax6.plot(timeSteps, y6Real)


ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')
ax1.set_ylim(ymin=-1.2, ymax=1.2)
ax1.set_xlim(xmin=-1.2, xmax=1.2)
ax4.set_ylim(ymin=-1.2, ymax=1.2)
ax4.set_xlim(xmin=-1.2, xmax=1.2)

ax1.set_title("e. Time wrapped")
ax2.set_title("Time Function")
ax3.set_title("Frequency Function")

ax4.set_title("e. Frequency wrapped")
ax5.set_title("Time Function integral recreated")
ax6.set_title("Time Function acos + bsin recreated")

ax1.set_xlabel("Real")
ax1.set_ylabel("Imag")
ax2.set_xlabel("Time")
ax2.set_ylabel("Mag")
ax3.set_xlabel("Freq")
ax3.set_ylabel("Mag")
ax1.grid(True)
ax2.grid(True)
ax4.grid(True)
ax5.grid(True)
ax6.grid(True)
ax1.set_aspect('equal')
ax4.set_aspect('equal')

fig.set_size_inches(16, 8)
plt.tight_layout()

wrapFreq = 0.1
wrapTime = 0.1

def onclick(event):
	global wrapFreq
	global wrapTime
	res = list(map(lambda x: timeToFreq(x, wrapFreq), timeSteps))
	x1 = list(map(lambda v: v.real, res))
	y1 = list(map(lambda v: v.imag, res))
	timeWrapLine.set_xdata(x1)
	timeWrapLine.set_ydata(y1)

	timeWeightX, err = integrate.quad(lambda x: timeToFreq(x, wrapFreq).real, timeStart, timeEnd)
	timeWeightY, err = integrate.quad(lambda x: timeToFreq(x, wrapFreq).imag, timeStart, timeEnd)
	timeWeightCenter.set_offsets([timeWeightX, timeWeightY])

	res = list(map(lambda f: freqToTime(f, wrapTime), freqSteps))
	x4 = list(map(lambda v: v.real, res))
	y4 = list(map(lambda v: v.imag, res))
	freqWrapLine.set_xdata(x4)
	freqWrapLine.set_ydata(y4)

	freqWeightX = fakeIntegral(lambda f: freqToTime(f, wrapTime).real, freqStart, freqEnd, 500)
	freqWeightY = fakeIntegral(lambda f: freqToTime(f, wrapTime).imag, freqStart, freqEnd, 500)
	freqWeightCenter.set_offsets([freqWeightX, freqWeightY])

	fig.canvas.draw()
	wrapFreq += 0.1
	wrapTime += 0.1


cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show(fig)
