# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:11:40 2019

@author: Ollie
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.linalg as la

def initialiseemptybox():
    #initialise the number of nodes for empty box
    UserInput = False
    while UserInput != True:
        try:
            N = int(input("Input number of nodes along length of square box:"))
        except ValueError:
            print("Invalid inputs")
        else:
            UserInput = True
    v = np.random.rand(N,N)
    return v,N

def capacitormesh(): 
    #Defining mesh
    boxlength = 15
    #length of box
    incre = 0.1     #increment value of nodes
    N = int(boxlength/incre)      #number of nodes 
    return boxlength,incre,N

def capacitorsize(boxlength,incre,N):
    #Define capacitor
    UserInput1 = False
    while UserInput1 != True:
        try:
            a = float(input("Input capacitor length (0<a<4):"))
        except ValueError:
            print("Invalid inputs")
        else:
            UserInput1 = True
    UserInput2 = False
    while UserInput2 != True:
        try:
            d = float(input("Input capacitor separation (0<d<4):"))
        except ValueError:
            print("Invalid inputs")
        else:
            UserInput2 = True
    
    platestart = int((boxlength - a)/(incre*2))
    platefinish = int(platestart+a/incre)
    plate1height = int((boxlength-d)/(incre*2))
    plate2height = int((plate1height + d/incre))
    return plate1height,plate2height,platestart,platefinish

def initialisecapacitor(plate1height,plate2height,platestart, platefinish,N):
    #Boundary/Initial Conditions
    v = np.zeros((N,N))
    v[plate1height,platestart:platefinish] = 1
    v[plate2height,platestart:platefinish] = -1
    return v
       
def convergerror():
    #Convergence criteria
    UserInput = False
    while UserInput != True:
        try:
            convergerror = float(input("Input convergence criteria:"))
        except ValueError:
            print("Invalid inputs")
        else:
            UserInput = True
    return convergerror
    
def newnode(v,i,j,N):
    up = v[i,j+1] if j < N-1 else 0
    down = v[i,j-1] if j > 0 else 0
    right = v[i+1,j] if i < N-1 else 0
    left  = v[i-1,j] if i > 0 else 0
    return(up + down + right + left)/4

def checkconverged(v1,v2):
    diff = np.abs(v1-v2)
    return np.amax(diff)

def jacobi(v,plate1height,plate2height,platestart,platefinish,N,convergerror,whichone): 
    tic = time.time()
    maxdiff = 1
    i=0
    j=0
    maxdiffplot = []
    itplot = []
    iterations = 0
    while maxdiff>=convergerror:
        v_new = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                v_new[i,j] = newnode(v,i,j,N)     
                j +=1
            i +=1 
        maxdiff = checkconverged(v,v_new)
        v = v_new
        iterations += 1
        maxdiffplot.append(maxdiff)
        itplot.append(iterations)
    print("Number of iterations:", iterations)
    print("Run time:", time.time()-tic,"s")
    return v, itplot, maxdiffplot

def testjacobi(v,N,convergerror):
    plate1height,plate2height,platestart,platefinish = 0,0,0,0  
    compare = convergerror - np.amax(jacobi(v,plate1height,plate2height,platestart,platefinish,N,convergerror,whichone='emptybox')[0])
    if compare <= 0:
        return print("Jacobi solver working")
    else:
        print("Jacobi solver not working")
        return print("Error:", compare)

def gaussseidel(v,plate1height,plate2height,platestart,platefinish,N,convergerror,whichone): 
    tic = time.time()
    maxdiff=1
    i=0
    j=0
    k=1
    l=1
    iterations = 0
    maxdiffplot=[]
    itplot = []
    while maxdiff>=convergerror:
        v_old=v.copy()
        if whichone == 'capacitor':
            for i in range(N):
                for j in range(N):  
                    if(i==plate1height and platestart<=j<=platefinish):
                        v[i,j] = 1
                    elif(i==plate2height and platestart<=j<=platefinish):
                        v[i,j] = -1
                    else:
                        v[i,j] = newnode(v,i,j,N)   
                    j +=2
            for k in range(N):
                for l in range(N):    
                    if(k==plate1height and platestart<=l<=platefinish):
                        v[k,l] = 1
                    elif(k==plate2height and platestart<=l<=platefinish):
                        v[k,l] = -1
                    else:
                        v[k,l] = newnode(v,k,l,N)   
                    l+=2
                i +=2 
                k+=2
            maxdiff = checkconverged(v_old,v)
            iterations += 1
            maxdiffplot.append(maxdiff)
            itplot.append(iterations)
        elif whichone == 'emptybox':
            for i in range(N):
                for j in range(N):  
                    v[i,j] = newnode(v,i,j,N)   
                    j +=2
            for k in range(N):
                for l in range(N):    
                    v[k,l] = newnode(v,k,l,N)   
                    l+=2       
                i+=2
                k+=2
            maxdiff = checkconverged(v_old,v) 
            iterations += 1
            maxdiffplot.append(maxdiff)
            itplot.append(iterations)
        else:
            print("Change plotting function value")
            return np.zeros((N,N))
    print("Number of iterations:", iterations)
    print("Run time:", time.time()-tic,"s")
    return v, itplot, maxdiffplot

def testgaussseidel(v,N,convergerror):
    plate1height,plate2height,platestart,platefinish = 0,0,0,0
    compare = convergerror - np.amax(gaussseidel(v,plate1height,plate2height,platestart,platefinish,N,convergerror,whichone='emptybox')[0])
    if compare <= 0:
        return print("Gauss-Seidel solver working")
        print(compare)
    else:
        print("Gauss-Seidel solver not working")
        return print("Error:", compare)
    
pokerlength = 50e-2 
incre = 1e-3
N = int(pokerlength/incre)

k=59
c=450
rho=7900
diffusivity= k/(c*rho)

Thot = 1000 #degrees celcius
Troom = 25 #degrees celcius
Tcold = 0 #degrees celcius

timestep = 0.1 #s

#initialise current temperature array
def initialtemp(N):
    poker = np.zeros((N))
    poker.fill(Troom)
    return poker

def hotend(poker,N):    
    poker[N-1] = Thot
    return poker

def roomTend(poker,N):
    poker[0] = Troom
    return poker

def coldend(poker,N):
    poker[0] = Tcold
    return poker
        
#initialise matrix
def initialisematrix(N):
    #precalculating inverse saves a lot of time compared to user solve function
    #when evolving temperature
    coeffsmatrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                coeffsmatrix[i,j] = 1 + 2*diffusivity*timestep/(incre**2)
            if i == j+1 or i == j-1:
                coeffsmatrix[i,j] = - diffusivity*timestep/(incre**2)
            if i == 0 and j == 0:
                coeffsmatrix[i,j] = 1 + diffusivity*timestep/(incre**2)
            elif i == N-1 and j == N-1:
                coeffsmatrix[i,j] = 1 + diffusivity*timestep/(incre**2)
    invcoeffsmatrix = la.inv(coeffsmatrix) 
    return invcoeffsmatrix
    
def evolvetemperature(poker,invcoeffsmatrix,N,convergerror,conditions):
    if conditions != 'test':
        hotend(poker,N)
    if conditions == 'cold':
        coldend(poker,N)
    iterations = 0
    pokerplot = []
    tic = time.time()
    maxdiff = 1
    while maxdiff>=convergerror:
        poker_new = invcoeffsmatrix@poker
        if conditions == 'test':
            poker_new[0] = 0
        if conditions!= 'test':
            hotend(poker_new,N)
        if conditions == 'room':
            roomTend(poker_new,N)
        if conditions == 'cold':
            coldend(poker_new,N)
        pokerplot.append(poker_new)
        maxdiff = checkconverged(poker,poker_new)
        poker = poker_new
        iterations += 1
    print("Equilibrium reached in", iterations, "iterations")
    print("Run time:", time.time()-tic,"s")
    if conditions != 'test':
        print("Physical time:", iterations*timestep,"s")
    return poker,pokerplot,iterations

def testevolvetemperature(N,convergerror):
    poker = np.zeros((N))
    poker[0]=0
    poker[1:N-1]=np.random.rand()
    invcoeffsmatrix = initialisematrix(N)
    poker = evolvetemperature(poker,invcoeffsmatrix,N,convergerror,conditions='test')[0]
    print("Error in solver:", checkconverged(np.zeros(N),poker))    
    
def plotting(xstart,xend,xincre,xlabel,ystart,yend,yincre,ylabel,grad1,grad2,contour,
             platestart,platefinish,plate1height,plate2height,capincre,physics,plot_type):
    
    x = np.arange(xstart,xend,xincre)
    y = np.arange(ystart,yend,yincre)
    X,Y = np.meshgrid(x,y)
    
    if physics == 'capacitor':
        plt.plot([platestart*capincre,platefinish*capincre],[plate1height*capincre,plate1height*capincre], linewidth = 5, color = 'k')
        plt.plot([platestart*capincre,platefinish*capincre],[plate2height*capincre,plate2height*capincre], linewidth = 5, color = 'k')
        
    if plot_type == 'contour':
        plt.contourf(X,Y,contour,40)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar()
        plt.show()
        
    elif plot_type == 'quiver':
        plt.contourf(X,Y,contour,40)
        plt.colorbar()
        plt.quiver(X,Y,grad2,grad1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
    elif plot_type == 'stream':
        plt.streamplot(X,Y,grad2,grad1,color=contour,arrowsize=1,minlength=0.01)
        plt.colorbar()
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
            
def runemptybox():

    v,N = initialiseemptybox()
    converge = convergerror()
        
    print("Testing Jacobi solver:")
    testjacobi(np.random.rand(5,5),5,1e-10)
    print("\nRunning Jacobi solver:")
    jac = jacobi(v,0,0,0,0,N,converge,whichone='emptybox')
   
    print("\nTesting Gauss-Seidel solver:")
    testgaussseidel(np.random.rand(7,7),7,1e-10)
    print("\nRunning Gauss-Seidel solver:")
    gauss=gaussseidel(v,0,0,0,0,N,converge,whichone='emptybox')
    
    plt.plot(jac[1],jac[2],label = 'Jacobi, N = %.f' %N)
    plt.plot(gauss[1],gauss[2], label = 'Gauss-Seidel, N = %.f' %N)
    plt.yscale('log')
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
     
def runcapacitor():  
    
    print("Testing Gauss-Siedel solver:")
    testgaussseidel(np.random.rand(7,7),7,1e-10)
    print("\nRunning Gauss-Seidel solver:")
    
    boxlength,incre,N = capacitormesh()
    plate1height,plate2height,platestart,platefinish = capacitorsize(boxlength,incre,N)
    v = initialisecapacitor(plate1height,plate2height,platestart, platefinish,N)
    
    gauss = gaussseidel(v,plate1height,plate2height,platestart,platefinish,N,convergerror(),whichone='capacitor')[0]
    
    grad1,grad2 = np.gradient(gauss)
    grad2 = np.zeros((N,N)) - grad2
    grad1 = np.zeros((N,N)) - grad1
    
    field_mag = np.sqrt(grad2*grad2+grad1*grad1)/incre    
    
    plotting(0,boxlength,incre,'X position (m)',0,boxlength,incre,'Y position (m)',grad1,grad2,gauss,
             platestart,platefinish,plate1height,plate2height,incre,physics = 'capacitor',plot_type = 'quiver')
    
    plotting(0,boxlength,incre,'X position (m)',0,boxlength,incre,'Y position (m)',grad1,grad2,field_mag,
             platestart,platefinish,plate1height,plate2height,incre,physics = 'capacitor',plot_type = 'contour')
    
    plotting(0,boxlength,incre,'X position (m)',0,boxlength,incre,'Y position (m)',grad1,grad2,field_mag,
             platestart,platefinish,plate1height,plate2height,incre,physics = 'capacitor',plot_type = 'stream')
    
def runpoker():
    print("Testing diffusion solver:")
    testevolvetemperature(10,1e-8)
    
    print("\nRunning room temperature diffusion solver:")
    poker,pokerplot,iterations = evolvetemperature(initialtemp(N),initialisematrix(N),N,convergerror(),conditions='room')
    
    plotting(0,pokerlength,incre,'Length along poker (m)',0,iterations*timestep,timestep,'Time (s)',0,0,pokerplot,
                 0,0,0,0,0,'poker',plot_type = 'contour')
 
runemptybox()
runcapacitor()
runpoker()
