import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from IPython.display import HTML
import sympy as sp

#Parameters
G = 1  #Pressure Gradient
H = 11 # Half width of channel
a1 = 1 # Large particle radius
a2 = 0.29069767441  #Small particle radius
phitot=0.4 #Total volume fraction
Beta=0.5 #Ratio of small to large particle volume fraction
phi20 = Beta * phitot  #Volume fraction of small particles
phi10 = phitot - phi20 #Volume fraction of large particles
J = 1024 #Number of grid points, note that J+1 is the number of grid points in the z direction for each species
ds = H / J #Grid spacing in z direction
dtau = 0.1 #Time step size
N = int(6000 /dtau)+1 #Number of time steps
s_grid = np.linspace(0, H, J+1) #Grid points in z direction
eps = 0
phiCM=0.59 #Monodisperse critical volume fraction
aavg = (a1**3*a2*phi20+a2**3*a1*phi10)/(a2**3*phi10+a1**3*phi20) #a_avg is the weighted average of particle radii


phi = np.zeros(2*(J+1))
phi[:J+1]=phi10
phi[J+1:]=phi20

#Phi_BM is bidisperse critical volume fraction, also note multiplying by mu_inv is the same as dividing by eta_s 
phi_bm = lambda phi1,phi2,c: phiCM*(1+c*abs((a1-a2)/(a1+a2))**(3/2)*(phi1/(phi1+phi2))**(3/2)*(phi2/(phi1+phi2))) #Bidisperse critical volume fraction 
mu_inv = lambda phi1, phi2: (1 - (phi1+phi2)/phi_bm(phi1,phi2,1.5)) ** 2 
f_hind1 = lambda phi1, phi2: (1 - phi1) * mu_inv(phi1, phi2) #Hinderence function for species 1
f_hind2 = lambda phi1, phi2: (1 - phi2) * mu_inv(phi1, phi2) #Hinderence function for species 2


# Make functions, should be somewhat intiutitve by name
def umax(phi):
    phi1 = phi[:J+1]
    phi2 = phi[J+1:]
    y = G * (H - s_grid) * mu_inv(phi1, phi2)
    return trapezoid(y, s_grid)

def gammadotavg(phi,umax):
    phi1 = phi[:J+1]
    phi2 = phi[J+1:]
    y= G*(H-s_grid)*mu_inv(phi1,phi2)+aavg*umax/H**2
    return 1/H * trapezoid(y,s_grid)

if gammadotavg(phi,umax(phi)) !=1: #Normalize average gammadot to be 1
    G=G/gammadotavg(phi,umax(phi))

def function_maker_k(input,derivwrt): #Make k functions
    phi1, phi2, a1, a2, c, phi_CM = sp.symbols('phi1 phi2 a1 a2 c phi_CM') # necessary parameters
    phi_bm= phi_CM*(1+c*abs((a1-a2)/(a1+a2))**(3/2)*(phi1/(phi1+phi2))**(3/2)*(phi2/(phi1+phi2))) #phi_bm as above

    if input==1: #k_1
        k=(phi1+phi2)**2/(phi_bm)**3*phi1 #defintion as in notes
        if derivwrt ==1: #Derivative with respec to phi 1
            dk=sp.diff(k,phi1)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dk,'numpy')
        else: #derivative with respect to phi 2
            dk=sp.diff(k,phi2)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dk,'numpy')
    else:
        k=(phi1+phi2)**2/(phi_bm)**3*phi2
        if derivwrt ==1:
            dk=sp.diff(k,phi1)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dk,'numpy')
        else:
            dk=sp.diff(k,phi2)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dk,'numpy')


#dk_1/dphi1, dk_1/dphi2,...
dk11= function_maker_k(input=1,derivwrt=1) 

dk12= function_maker_k(input=1,derivwrt=2)

dk21= function_maker_k(input=2,derivwrt=1)

dk22= function_maker_k(input=2,derivwrt=2)


#Definition of functions as in notes
def g11(phi1, phi2, s):
    factor = 2.66*a1**2/9*f_hind1(phi1, phi2)*G*(H-s)
    return factor * dk11(phi1,phi2,a1,a2,1.5,phiCM)
    
def g12(phi1, phi2, s):
    factor = 2.66*a1**2/9*f_hind1(phi1, phi2)*G*(H-s)
    return factor * dk12(phi1,phi2,a1,a2,1.5,phiCM)
    
def g21(phi1, phi2, s):
    factor = 2.66*a2**2/9*f_hind2(phi1, phi2)*G*(H-s)
    return factor * dk21(phi1,phi2,a1,a2,1.5,phiCM)
    
def g22(phi1, phi2, s):
    factor = 2.66*a2**2/9*f_hind2(phi1, phi2)*G*(H-s)
    return factor * dk22(phi1,phi2,a1,a2,1.5,phiCM)

def g11p(phi1, phi2, s):
    factor = 1.33*G*(H-s)
    return factor * dk11(phi1,phi2,a1,a2,1.5,phiCM)

def g12p(phi1, phi2, s):
    factor = 1.33*G*(H-s)
    return factor * dk12(phi1,phi2,a1,a2,1.5,phiCM)
    
def g21p(phi1, phi2, s):
    factor = 1.33*G*(H-s)
    return factor * dk21(phi1,phi2,a1,a2,1.5,phiCM)
    
def g22p(phi1, phi2, s):
    factor = 1.33*G*(H-s)
    return factor * dk22(phi1,phi2,a1,a2,1.5,phiCM)

#Same thing for l function. Note that l is eta_n^i in the notes but differentitation and other things are as above
def function_maker_l(input,derivwrt):
    phi1, phi2, a1, a2, c, phi_CM = sp.symbols('phi1 phi2 a1 a2 c phi_CM')
    phi_bm= phi_CM*(1+c*abs((a1-a2)/(a1+a2))**(3/2)*(phi1/(phi1+phi2))**(3/2)*(phi2/(phi1+phi2)))

    if input==1:
        l=1.33*(phi1+phi2)**2/(phi_bm)**3*phi1*(1-(phi1+phi2)/phi_bm)**(-2)
        if derivwrt ==1:
            dl=sp.diff(l,phi1)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dl,'numpy')
        else:
            dl=sp.diff(l,phi2)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dl,'numpy')
    else:
        l=1.33*(phi1+phi2)**2/(phi_bm)**3*phi2*(1-(phi1+phi2)/phi_bm)**(-2)
        if derivwrt ==1:
            dl=sp.diff(l,phi1)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dl,'numpy')
        else:
            dl=sp.diff(l,phi2)
            return sp.lambdify((phi1,phi2,a1,a2,c,phi_CM),dl,'numpy')
        
dl11= function_maker_l(input=1,derivwrt=1)

dl12= function_maker_l(input=1,derivwrt=2)

dl21= function_maker_l(input=2,derivwrt=1)

dl22= function_maker_l(input=2,derivwrt=2)

#define f functions as in notes
def f11(phi1,phi2,umax):
    factor = aavg*2*a1**2*f_hind1(phi1,phi2)*umax/(9*H**2)
    return factor * dl11(phi1,phi2,a1,a2,1.5,phiCM)

def f12(phi1,phi2,umax):
    factor = aavg*2*a1**2*f_hind1(phi1,phi2)*umax/(9*H**2)
    return factor * dl12(phi1,phi2,a1,a2,1.5,phiCM)

def f21(phi1,phi2,umax):
    factor =aavg* 2*a2**2*f_hind2(phi1,phi2)*umax/(9*H**2)
    return factor * dl21(phi1,phi2,a1,a2,1.5,phiCM)

def f22(phi1,phi2,umax):
    factor = aavg*2*a2**2*f_hind2(phi1,phi2)*umax/(9*H**2)
    return factor * dl22(phi1,phi2,a1,a2,1.5,phiCM)

def f11p(phi1,phi2,umax):
    factor = aavg*umax/(H**2)
    return factor * dl11(phi1,phi2,a1,a2,1.5,phiCM)

def f12p(phi1,phi2,umax):
    factor = aavg*umax/(H**2)
    return factor * dl12(phi1,phi2,a1,a2,1.5,phiCM)

def f21p(phi1,phi2,umax):
    factor = aavg*umax/(H**2)
    return factor * dl21(phi1,phi2,a1,a2,1.5,phiCM)

def f22p(phi1,phi2,umax):
    factor = aavg*umax/(H**2)
    return factor * dl22(phi1,phi2,a1,a2,1.5,phiCM)

#definitions of h1,h2,h1p,h2p
h1= lambda phi1,phi2: - f_hind1(phi1,phi2)*2.66*a1**2*G/9*(phi1+phi2)**2/(phi_bm(phi1,phi2,1.5))**3
h2= lambda phi1,phi2: - f_hind2(phi1,phi2)*2.66*a2**2*G/9*(phi1+phi2)**2/(phi_bm(phi1,phi2,1.5))**3

h1p= lambda phi1,phi2: -1.33*G*(phi1+phi2)**2/(phi_bm(phi1,phi2,1.5))**3
h2p= lambda phi1,phi2: -1.33*G*(phi1+phi2)**2/(phi_bm(phi1,phi2,1.5))**3

#Assemble matrix
def phi_step(phi_prev, dt):
    A = lil_matrix((2*J+2, 2*J+2)) #define A matrix
    b = np.zeros(2*J+2) #define B matrix

    u_max = umax(phi_prev) #define u_max

    phi10 = phi_prev[0] #phi_prev is the values in this timestep. So this initializes the values of phi10 and phi20 to be the values of phi1 and phi2 at 0.
    phi20 = phi_prev[J+1]

    # Defined as in notes
    A[0, 0] = -3*(f11p(phi10,phi20,u_max) + g11p(phi10,phi20,0))/(2*ds)+h1p(phi10,phi20)
    A[0, 1] = 4*(f11p(phi10,phi20,u_max) + g11p(phi10,phi20,0))/(2*ds)
    A[0,2] = -(f11p(phi10,phi20,u_max) + g11p(phi10,phi20,0))/(2*ds)

    A[0, J+1] = -3*(f12p(phi10,phi20,u_max) + g12p(phi10,phi20,0))/(2*ds)
    A[0, J+2] =  4*(f12p(phi10,phi20,u_max) + g12p(phi10,phi20,0))/(2*ds)
    A[0,J+3]= -(f12p(phi10,phi20,u_max) + g12p(phi10,phi20,0))/(2*ds)
    b[0] = 0

    A[1, 0] = -3*(f21p(phi10,phi20,u_max) + g21p(phi10,phi20,0))/(2*ds)
    A[1, 1] = 4*(f21p(phi10,phi20,u_max) + g21p(phi10,phi20,0))/(2*ds)
    A[1,2] = -(f21p(phi10,phi20,u_max) + g21p(phi10,phi20,0))/(2*ds)

    A[1, J+1] = -3*(f22p(phi10,phi20,u_max) + g22p(phi10,phi20,0))/(2*ds)+h2p(phi10,phi20)
    A[1, J+2] =  4*(f22p(phi10,phi20,u_max) + g22p(phi10,phi20,0))/(2*ds)
    A[1,J+3]= -(f22p(phi10,phi20,u_max) + g22p(phi10,phi20,0))/(2*ds)
    
    b[1] = 0
    
    T = np.zeros((2*J-2, 2*J+2))
    for i in range(2*J-2):
        if i < J-1:
            T[i, i+1] = 1
        else:
            T[i, i+3] = 1

    #Define matrix for interior values
    G_mat = np.zeros((2*J-2, 2*J+2))
    
    #Half indices
    phi1half = (phi_prev[:J] + phi_prev[1:J+1]) / 2
    phi2half = (phi_prev[J+1:2*J+1] + phi_prev[J+2:2*J+2]) / 2
    shalf = (s_grid[:-1] + s_grid[1:]) / 2

    #Functions based on half indices
    f11_half= f11(phi1half,phi2half,u_max)
    f12_half = f12(phi1half,phi2half,u_max)
    f21_half = f21(phi1half,phi2half,u_max)
    f22_half= f22(phi1half,phi2half,u_max) 
    g11_half = g11(phi1half, phi2half, shalf)
    g12_half = g12(phi1half, phi2half, shalf)
    g21_half = g21(phi1half, phi2half, shalf)
    g22_half = g22(phi1half, phi2half, shalf)
    h1_arr = h1(phi_prev[:J+1],phi_prev[J+1:])
    h2_arr = h2(phi_prev[:J+1],phi_prev[J+1:])

    idx = np.arange(1, J)
    #Phi 1 terms
    G_mat[idx-1, idx-1] = g11_half[idx-1]+f11_half[idx-1]-ds*h1_arr[idx-1]/2
    G_mat[idx-1, idx] = -(g11_half[idx-1]+g11_half[idx] +f11_half[idx-1]+f11_half[idx]+ds**2/dt) # Add 1+ term
    G_mat[idx-1, idx+1] = g11_half[idx]+f11_half[idx]+ds*h1_arr[idx+1]/2
    G_mat[idx-1, J+idx] = g12_half[idx-1] + f12_half[idx-1]
    G_mat[idx-1, J+idx+1] = -g12_half[idx-1]-g12_half[idx]-f12_half[idx-1]-f12_half[idx]
    G_mat[idx-1, J+idx+2] = g12_half[idx]+ f12_half[idx]
    
    #Phi_2 terms
    G_mat[J+idx-2, idx-1] = g21_half[idx-1]+f21_half[idx-1]
    G_mat[J+idx-2, idx] = -g21_half[idx-1]-g21_half[idx]-f21_half[idx-1]-f21_half[idx]
    G_mat[J+idx-2, idx+1] = g21_half[idx]+f21_half[idx]
    G_mat[J+idx-2, J+idx] = g22_half[idx-1]+f22_half[idx-1]-ds*h2_arr[idx-1]/2
    G_mat[J+idx-2, J+idx+1] = -g22_half[idx-1]-g22_half[idx]-f22_half[idx-1]-f22_half[idx]-ds**2/dt
    G_mat[J+idx-2, J+idx+2] = g22_half[idx]+f22_half[idx]+ds*h2_arr[idx+1]/2
        
    #Set the interior values of A uisng G matrix, and interior values of b using T and phi_prev
    A[2:-2, :] =  -1/(ds**2) * G_mat
    b[2:-2] = 1/dt * T @ phi_prev

    # Set phi1J and phi2J to be the centerline values of phi1 and phi2 in this stime step
    phi1J = phi_prev[J]
    phi2J = phi_prev[2*J+1]

    #Same as theory
    A[-2,J-2]= (f21p(phi1J,phi2J,u_max))/(2*ds)
    A[-2, J-1] =  -4*(f21p(phi1J,phi2J,u_max))/(2*ds)
    A[-2, J] = 3*(f21p(phi1J,phi2J,u_max))/(2*ds)

    A[-2,2*J-1]= (f22p(phi1J,phi2J,u_max))/(2*ds)
    A[-2, 2*J] = -4*(f22p(phi1J,phi2J,u_max))/(2*ds)
    A[-2, 2*J+1] = 3*(f22p(phi1J,phi2J,u_max))/(2*ds)+h2p(phi1J,phi2J)

    b[-2] = 0

    A[-1,J-2]= (f11p(phi1J,phi2J,u_max))/(2*ds)
    A[-1, J-1] =  -4*(f11p(phi1J,phi2J,u_max))/(2*ds)
    A[-1, J] = 3*(f11p(phi1J,phi2J,u_max))/(2*ds)+h1p(phi1J,phi2J)

    A[-1,2*J-1]= f12p(phi1J,phi2J,u_max)/(2*ds)
    A[-1, 2*J] = -4*(f12p(phi1J,phi2J,u_max))/(2*ds)
    A[-1, 2*J+1] = 3*f12p(phi1J,phi2J,u_max)/(2*ds)
    b[-1] = 0
    A_sparse = A.tocsr()
    return spsolve(A_sparse, b)

#Function to advance phi1,phi2 N time step
def phi_advance(phi_init, dtau, N):
    phi = np.zeros((N, 2*J+2))
    phi[0, :] = phi_init
    for n in tqdm(range(1, N)):
        phi_prev = phi[n-1, :]
        phi[n, :] = phi_step(phi_prev, dtau)

    return phi

#Run the acutal code: initialize phi1, phi2. Advance them N time steps with step size of dtau. Print out final profile valeues.
phi_init = np.zeros(2*J+2)
phi_init[:J+1] = phi10
phi_init[J+1:] = phi20
phi = phi_advance(phi_init, dtau, N)
print(phi.shape, phi)
phi_final = phi[-1]             # shape (2*J+2,)
phi1_final = phi_final[:J+1]    # species 1, y=0…H
phi2_final = phi_final[J+1:]    # species 2, y=0…H
print(gammadotavg(phi_final,umax(phi_final)))

# Make the plot
plt.figure()
tau = (N - 1) * dtau
plt.plot(s_grid, phi1_final, color='red', label=r'$\phi_1$')   # red
plt.plot(s_grid, phi2_final, color='blue', label=r'$\phi_2$')  # blue
plt.xlabel(r'$y/a_L$', fontsize=18)  
plt.ylabel(r'$\phi$', fontsize=18)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)  # ticks
plt.ylim(0, 0.35)  # y-axis scale
plt.tight_layout()
plt.show()

integral_1 = np.zeros(phi.shape[0])
integral_2 = np.zeros(phi.shape[0])
for i in range(phi.shape[0]):
    phi_1 = phi[i, :J+1]
    phi_2 = phi[i, J+1:]
    integral_1[i] = trapezoid(phi_1, s_grid)
    integral_2[i] = trapezoid(phi_2, s_grid)



plt.plot(np.arange(0, phi.shape[0]), integral_1, 'r')
plt.plot(np.arange(0, phi.shape[0]), integral_2, 'b')
plt.show()