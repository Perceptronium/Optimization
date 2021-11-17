import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def myfunc(x):
    y = np.array(x)
    fx = y.T @ y
    return fx

def plotFunc(func, bounds_lo, bounds_up, trace_xy = None, trace_z = None):
	x = np.linspace(bounds_lo[0], bounds_up[0], 30)
	y = np.linspace(bounds_lo[1], bounds_up[1], 30)
	xMesh, yMesh = np.meshgrid(x, y, indexing='ij')
	zMesh = np.zeros_like(xMesh)
	for i in range(x.shape[0]):
		for j in range(y.shape[0]):
			zMesh[i,j] = func(np.array([xMesh[i,j], yMesh[i,j]]))

	fig = plt.figure(figsize=(12,6))
	#print('test')
	ax1 = fig.add_subplot(121, projection="3d")
	surf = ax1.plot_surface(xMesh, yMesh, zMesh, cmap=cm.coolwarm)
	if trace_xy is not None:
		ax1.plot(trace_xy[:,0], trace_xy[:,1], trace_z[:,0], 'ko-', linewidth=1.0, markersize=2.0)
	fig.colorbar(surf)
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax1.set_zlabel('f')

	ax2 = fig.add_subplot(122)
	surf2 = plt.contourf(xMesh, yMesh, zMesh, cmap=cm.coolwarm)
	if trace_xy is not None:
		ax2.plot(trace_xy[:,0], trace_xy[:,1], 'ko-', linewidth=1.0, markersize=2.0)
	fig.colorbar(surf2)
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
    
	plt.show()
    
#plotFunc(myfunc, [-2,-2], [2,2])
