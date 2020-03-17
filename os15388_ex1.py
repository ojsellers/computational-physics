# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:34:46 2019

@author: Ollie
"""

"""
On runing file, a menu will appear allowing the user to navigate through the exercise
"""

import numpy as np
import scipy.linalg as la
import time
import math
import matplotlib.pyplot as plt

def transpose(m):
    #transpose matrix by swapping index numbers for each entry, takes square array as argument
    #returns transpose
    t=[]
    for i in range(len(m)):
        tr = []
        for j in range(len(m)):
            tr.append(m[j][i])
        t.append(tr)
    return t

def minor(m,i,j):
    #funtion to return the minor for a certain matrix element i,j
    #arguments are square matrix and indices of matrix element
    return m[np.array(list(range(i))+list(range(i+1,len(m))))[:,np.newaxis], 
           np.array(list(range(j))+list(range(j+1,len(m))))]
          
def det(m):
    #recursive function uses minor function to return the determinant
    #for a square matrix
    #simplest case for 2x2 matrices
    if len(m) == 2 :
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]
    
    # for higher order matrices reducing to 3x3 increases computing speed than for reducing to 2x2    
    if len(m) == 3:
        return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0])
    
    #iterative function to reduce matrices to 3x3s and sum determinants    
    d = 0
    for i in range(len(m)):
        d += ((-1)**i)*m[0][i]*det(minor(m,0,i)) #using Laplace algorithm to find determinant
    return d
    
def inverse(m):
    #returns inverse for input of square array
    #special case for 1x1 so function is robust
    if len(m) == 1:
        return [1 / m[0][0]]
    d = det(m)
    print("Determinant:", d)
    if d == 0:
        a = []
        print("Singular, so no inverse")    #making function robust for singular matrices
        return a
    #simplest case for 2x2 matrices
    if len(m) == 2: 
        return [[m[1][1]/d, -1*m[0][1]/d], [-1*m[1][0]/d, m[0][0]/d]]
    # for higher order matrices by using cofactors
    cofact = []     #initialise cofactor array
    for i in range (len(m)):    #columns
        cofactr = []    #initialise row array
        for j in range(len(m)):     #rows
            cofactr.append((-1)**(i+j)*det(minor(m,i,j)))   #calculate matrix of cofactors with matrix of minors
        cofact.append(cofactr/d)    #append column and divide by determinant  
    inverse = transpose(cofact)    #transpose to find inverse       
    return inverse

def inputmatrix():
    #function to return a matrix that can be input into other functions
    #from user inputs
    valid_input = False    
    while not valid_input:          #while loops to make inputs robust
        q = input("Input 1 for random matrix, 2 for user defined:")     
        if q == '1':    
            while True:           
                try:
                    n = int(input("Input dimensions of matrix:"))
                except ValueError:
                    print("Invalid inputs")
                else:
                    a = np.random.rand(n,n)     #creating random nxn matrix with numpy.random
                    valid_input = True
                    break           
        elif q == '2':
            while True:
                try:         
                    n = int(input("Input dimensions of matrix:"))
                except ValueError:
                    print("Invalid inputs")
                else:
                    a = np.zeros((n,n)) #initialise array with zeros
                    for i in range(n):
                        for j in range(n):
                            while True:
                                print("Input matrix element",i+1,j+1,":") #allows user to manually input matrix
                                try:
                                    inn = float(input())
                                except ValueError:
                                    print("Invalid inputs")
                                else:
                                    a[i][j] = inn
                                    valid_input = True
                                    break
                    break                  
        else:
            print("Invalid inputs")    
    print ("Original matrix:", "\n",a)   
    return a
    
def analyticalinversion(a):
    #function returns inverse matrix with timing for a square array argument
    begin = time.time()
    m = inverse(a)
    print("Inverse:")
    for row in m: 
        print(row)        
    end = time.time()        
    print("Computing time:", end - begin, "s")
    return m
    
def checkai(m,a):
    #function compares original matrix and inverse arguments (m&a) to identity
    #should return nxn of zeros for perfect inverse
    return m@a - np.identity(len(a))
    
def inputvector(m):
    #function returns vector b for user inputs and an argument of the previously
    #specified matrix
    valid_input2 = False
    while not valid_input2:        #making inputs robust with while loops
        q = input("Input 1 for random vector, 2 for user defined:")        
        n = len(m)        
        if q == '1':
            v = np.random.rand(n)   #random vector with numpy.random
            valid_input2 = True            
        elif q == '2':
            v = np.zeros(n)            #initialise vector array with zeros
            for i in range(n):
                while True:
                    print("Input right hand side vector element", i+1, ":")
                    try:
                        inn = float(input())
                    except ValueError:
                        print("Invalid inputs")
                    else:
                        v[i] = inn  #allows user to manually specify eleemens
                        valid_input2 = True
                        break
            break                
        else:
            print("Invalid inputs")       
    print("Vector:", "\n", v, "\n")
    return v

def LU(m, v):
    #function to solve linear equations for arguments of matrix and vector
    #returns array of solutions
    return la.lu_solve(la.lu_factor(m), v)       #LU routine from scipy.linalg library
    
def SV(m,v):    
    #function to solve linear equations for arguments of matrix and vector
    #returns array of solutions
    (u, S, V) = la.svd(m)   #SV routine from scipy.linalg library
    s = np.diag(S)
    for i in range(len(s)): #inverse of diagonal matrix is the inverse of each element
        s[i][i] = 1 / s[i][i]        
    d = u.T@v
    d = s@d
    return V.T@d

def aisolve(m,v):
    #function to solve linear equations for arguments of matrix and vector
    #returns array of solutions for analytical inversion method
    beginai = time.time()    
    print("Analytical inversion")
    inv = inverse(m)
    if det(m) != 0:   #removing singular matrices as no inverse
        x = np.dot(inv,v) 
        print("x,y,z,...:",x)
    endai = time.time()
    print("Run time:", endai-beginai, "s \n")
    
def compare(m,v):   
    #function to print timing for LU and SV solving methods
    #takes square matrix and vector as arguments
    beginlu = time.time()
    print("LU decomposition")
    print("x,y,z,...:", LU(m,v))   
    endlu = time.time()   
    print("Run time:", endlu-beginlu, "s \n") 
    beginsv = time.time()   
    print("Single value decomposition") 
    print("x,y,z,...:", SV(m,v))  
    endsv = time.time()   
    print("Run time:", endsv-beginsv, "s")
    
def T2d(x, y):
    #function to find tension in 2 dimensions
    #arguments are x and y position and returns tension vector
    theta1 = math.atan(x/(8-y))       #calculating theta values from position
    theta2 = math.atan((15-x)/(8-y))    
    
    a = [[math.sin(theta1), - math.sin(theta2)], #create matrix
         [math.cos(theta1), math.cos(theta2)]]   
    
    v = [0 , 70*9.81]        #create vector
    
    return LU(a, v)     #solved with LU decomposition

def T3d(x, y, z):   
    #function to find tension in 3 dimensions
    #arguments are x,y, and z position and returns tension vector
    maxphi = math.atan(8/7.5)    
    if (y>7):   #removing impossible y values
        return [0,0,0]
    if (y==0):
        return [0,0,0]
    #finding first phi value and setting impossible positions to zero tension
    if x != 0:
        phi1 = math.atan(z/x)
        if phi1 > maxphi:
            return [0,0,0]
    if x == 0 :
        phi1 = math.pi/2
        if z != 0:
            return [0,0,0]
    #finding second phi value and setting impossible positions to zero tension 
    if x != 15:
        phi2 = math.atan(z/(15-x))
        if phi2 > maxphi:
            return [0,0,0]
    if x == 15:
        phi2 = math.pi/2
        if z != 0:
            return [0,0,0]
    #for third value an extra step is required because cos is an even function
    #and phi changes sign at x = 7.5 due to the placement of the barrel
    if x < 7.5:
        phi3 = math.atan((8-z)/(7.5-x))
        if phi3 < maxphi:
            return [0,0,0]
        xT3 = math.cos(phi3)
    if x > 7.5:
        phi3 = -math.atan((8-z)/(7.5-x))
        if phi3 < maxphi:
            return [0,0,0]
        xT3 = - math.cos(phi3)
    if x == 7.5:
        phi3 = math.pi/2
        xT3 = 0
    
    theta1 = math.atan(math.sqrt(x**2+z**2)/(8-y))      #theta values are similar to 2d case, but with added z component
    theta2 = math.atan(math.sqrt((15-x)**2+z**2)/(8-y))
    theta3 = math.atan(math.sqrt((7.5-x)**2 + (8-z)**2)/(8-y))  
    #matrix and vector are created for 3d set of linear equations
    a = [[math.sin(theta1)*math.cos(phi1), -math.sin(theta2)*math.cos(phi2), -math.sin(theta3)*xT3], 
         [math.cos(theta1), math.cos(theta2), math.cos(theta3)], 
         [math.sin(theta1)*math.sin(phi1), math.sin(theta2)*math.sin(phi2), -math.sin(theta3)*math.sin(phi3)]]

    v = [0, 70*9.81, 0]
    
    return LU(a,v)  #solved with LU decomposition

def plot2D():
    #function to plot tension in 2 dimensions, takes no arguments
    print("\n 2-dimensional tension plots:")    
    
    while True:     #making inputs robust with while loops
        try:
            s = float(input("Input X step (m):"))   #position increment for calc
        except ValueError:
            print("Invalid inputs")
        else:
            xstep = s
            break
        
    while True:
        try:
            t = float(input("Input Y step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            ystep = t
            break
    
    x = np.arange(0, 15 + xstep, xstep)     #generating arrays for plot
    y = np.arange(0, 7 + ystep, ystep)
    X, Y = np.meshgrid(x, y)    #creating meshgrid
    T1 = np.array([T2d(x,y)[0] for x,y in zip(np.ravel(X), np.ravel(Y))])   #calculating tension for input arrays
    T2 = np.array([T2d(x,y)[1] for x,y in zip(np.ravel(X), np.ravel(Y))])
    T1 = T1.reshape(X.shape) #reshaping tension array for plot
    T2 = T2.reshape(X.shape)
    
    plt.contourf(X,Y,T1, 40, cmap ='Greys') #plotting contour
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.colorbar()
    plt.show() 
    
def max2D():
    #function to print maximum tensions in 2D system
    print("\n 2-dimensional maximum tensions:")    
    while True:     #while loops to make user inputs robust
        try:
            s = float(input("Input X step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            xstep = s
            break
        
    while True:
        try:
            t = float(input("Input Y step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            ystep = t
            break
    
    begin2d = time.time()   #timing calculation
    
    maxx = int(15/xstep)    #determining maximum x and y values
    maxy = int(7/ystep)
    T1max2d = 0     #initialising output variables
    T2max2d = 0
    Tmax2d = 0
    #nested for loop to calculate the components of tension for each position
    #if new component is larger than any previous value, it becomes new maximum
    for x in range(0, maxx + 1):
        x = x*xstep
        for y in range(0, maxy + 1):
            y = y*ystep
            T = T2d(x,y)
            T1 = T[0]
            T2 = T[1]
            Tm = T1+T2
            if T1 > T1max2d:
                T1max2d = T1
                T1xmax2d = x
                T1ymax2d = y
            if T2 > T2max2d:
                T2max2d = T2
                T2xmax2d = x
                T2ymax2d = y
            if Tm > Tmax2d:
                Tmax2d = Tm
                Txmax = x
                Tymax = y
    
    #print functions to return maximums and corresponding positions
    print("\n", "Wire 1: \n")
    print("Maximum Tension:", T1max2d, "N")
    print("X value:", T1xmax2d, "m")
    print("Y value:", T1ymax2d, "m", "\n")
    
    print("Wire 2: \n")
    print("Maximum Tension:", T2max2d, "N")
    print("X value:", T2xmax2d, "m")
    print("Y value:", T2ymax2d, "m", "\n")
    
    print("Maximum Total Tension:", Tmax2d, "N")
    print("X value:", Txmax, "m")
    print("Y value:", Tymax, "m", "\n")
    
    end2d = time.time()
    print("Run time:", end2d-begin2d, "s","\n") #printing time

def plot3D():
    #same as 2D case but for 3 dimensions, plots tension in each wire as a fn of position
    print("\n 3-dimensional tension plots:")
    
    while True:     #while loops to make user inputs robust
        try:
            s = float(input("Input X step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            xstep = s
            break
    
    while True:
        try:
            t = float(input("Input Z step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            zstep = t
            break
        
    while True:
        try:
            b = float(input("Input Y position to plot at (m):"))    #y position to make plot at
        except ValueError:
            print("Invalid inputs")
        else:
            y = b
            break

    #creating x,z arrarys
    x = np.arange(0,15 + xstep, xstep)
    #y = np.arange(0, 7 + ystep, ystep)
    z = np.arange(0,8 +zstep ,zstep)
    X, Z = np.meshgrid(x, z)
    T1 = np.array([T3d(x, y, z)[0] for x,z in zip(np.ravel(X), np.ravel(Z))]) #calculating components of tension for each wire
    T1 = T1.reshape(X.shape)    #and reshaping array for plot
    T2 = np.array([T3d(x, y, z)[1] for x,z in zip(np.ravel(X), np.ravel(Z))])
    T2 = T2.reshape(X.shape)
    T3 = np.array([T3d(x, y, z)[2] for x,z in zip(np.ravel(X), np.ravel(Z))])
    T3 = T3.reshape(X.shape)
    
    #fig2 = plt.figure
    plt.contourf(X,Z,T1, 40, cmap = 'Greys') #plotting a contour for each wire
    plt.colorbar()
    plt.plot(0,0,'ro') 
 #   plt.title('Tension in first wire')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.show()
    
    #fig3 = plt.figure()
    plt.contourf(X,Z,T2, 40, cmap = 'Greys')
    plt.colorbar()
    plt.plot(15,0,'ro')
  #  plt.title('Tension in second wire')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.show()
    
    #fig4 = plt.figure()
    plt.contourf(X,Z,T3, 40, cmap = 'Greys')
    plt.colorbar()
    plt.plot(7.5,8,'ro')
   # plt.title('Tension in third wire')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.show()
      
def max3D():
    #function to calculate maximum tension in 3D system
    print("\n 3-dimensional maximum tensions:")

    while True:     #while loop makes user inputs robust
        try:
            s = float(input("Input X step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            xstep = s
            break
    while True:
        try:
            t = float(input("Input Y step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            ystep = t
            break
    while True:
        try:
            e = float(input("Input Z step (m):"))
        except ValueError:
            print("Invalid inputs")
        else:
            zstep = e
            break

    begin3D = time.time()

    maxx = int(15/xstep)    #calculating max x,y,z
    maxy = int(7/ystep)
    maxz = int(8/zstep)
    T1max3d = 0     #initialising output variables
    T2max3d = 0
    T3max3d = 0
    Tmax3d = 0
    #for loop to calculate tension components at each position and keep largest values
    for x in range(0, maxx + 1):
        x = x*xstep
        for y in range(0, maxy + 1):
            y = y*ystep
            for z in range(0, maxz + 1): 
                z = z*zstep
                T = T3d(x,y,z)
                T1 = T[0]
                T2 = T[1]
                T3 = T[2]
                Tm = T1 + T2 + T3
                if T1 > T1max3d:
                    T1max3d = T1
                    T1xmax3d = x
                    T1ymax3d = y
                    T1zmax3d = z
                if T2 > T2max3d:
                    T2max3d = T2
                    T2xmax3d = x
                    T2ymax3d = y
                    T2zmax3d = z
                if T3 > T3max3d:
                    T3max3d = T3
                    T3xmax3d = x
                    T3ymax3d = y
                    T3zmax3d = z
                if Tm > Tmax3d:
                    Tmax3d = Tm
                    Txmax3d = x
                    Tymax3d = y
                    Tzmax3d = z
    
    #print functions to show maximum tension values and corresponding positions
    print("\n", "Wire 1: \n")
    print("Maximum Tension:", T1max3d, "N")
    print("X value:", T1xmax3d, "m")
    print("Y value:", T1ymax3d, "m")
    print("Z value:", T1zmax3d, "m", "\n")
    
    print("Wire 2: \n")
    print("Maximum Tension:", T2max3d, "N")
    print("X value:", T2xmax3d, "m")
    print("Y value:", T2ymax3d, "m")
    print("Z value:", T2zmax3d, "m", "\n")
    
    print("Wire 3: \n")
    print("Maximum Tension:", T3max3d, "N")
    print("X value:", T3xmax3d, "m")
    print("Y value:", T3ymax3d, "m")
    print("Z value:", T3zmax3d, "m", "\n")
    
    print("Maximum Total Tension:", Tmax3d, "N")
    print("X value:", Txmax3d, "m")
    print("Y value:", Tymax3d, "m")
    print("Z value:", Tzmax3d, "m", "\n")
    
    end3D = time.time()
    
    print("Run time:", end3D - begin3D, "s")
    
#now begin program to run through exercise
UserInput = '1'
while UserInput != 'q':
    UserInput = input('"a" for analytical inversion \n "b" for time comparison \n "c" for testing singular matrices \n "d" for 2D tension \n "e" for 3D tension \n "q" to quit:')    
    print("You have chosen:", UserInput)
    if UserInput == 'q':
        break
    if UserInput == 'a':
        print("Analytical inversion:")
        #calculating inverse of matrix through analytical inversion
        a = inputmatrix()
        m = analyticalinversion(a)
        p = checkai(m, a)
        print("Error:", np.amax(p))

    elif UserInput == 'b':
        print("\n Time comparison of LU and SV decomposition methods:")
        #comparing time of LU and SV methods
        m=inputmatrix()
        v=inputvector(m)
        compare(m,v)

    elif UserInput == 'c':
        print("\n Comparing methods as the matrix becomes singular:")
        #comparison of accuracy as matrix becomes singular
        while True:     #make k value input robust with while loop
                try:
                    t = float(input("Input k value:"))
                except ValueError:
                    print("Invalid inputs")
                else:
                    k = t
                    break
        #creating matrix for input into solve functions
        m = np.zeros((3,3))
        m[0][0]=1   
        m[0][1]=1
        m[0][2]=1
        m[1][0]=1
        m[1][1]=2
        m[1][2]=-1
        m[2][0]=2
        m[2][1]=3
        m[2][2]=k
        
        v = [5,10,15]
        
        aisolve(m,v)
        compare(m,v)
        
    elif UserInput == 'd':
        print("\n 2-dimensional case:")
        #solving 2d tension case
        plot2D()
        max2D()
        
    elif UserInput == 'e':
        print("\n 3-dimensional case:")
        #solving 3d tension case 
        max3D()
        plot3D()
        
    else:
        print("Invalid inputs")

print("Goodbye")   