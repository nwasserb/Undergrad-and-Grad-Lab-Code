import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.optimize import minimize

#######################################################
# new fidelity calculation function
# inputs: probabilities: p_H in H direction, p_D in H direction, p_H in D direction
# outputs, theta, psi, lambda (all in radians)
def calculate_HBasis_new(H_probH, H_probD, D_probH):
    theta = 2 * np.arccos(np.sqrt(H_probH))
    # print("this is calulating psi ")
    


    # psi = np.arccos((2 * H_probD - 1) / np.sin(theta))
    psi = np.arccos((2 * H_probD - 1) / (2*np.cos(theta/2)*np.sin(theta/2)))
    # print((2 * H_probD - 1))
    # print(np.sin(theta))
    # lambdas = np.arcsin((1 - 2 * D_probH) / np.sin(theta))
    lambdas = -np.arccos((1 - 2 * D_probH) / (2*np.cos(theta/2)*np.sin(theta/2)))
    
 
    print((1 - 2 * D_probH) / (2*np.cos(theta/2)*np.sin(theta/2)))

    print(theta)
    print(psi)
    print(lambdas)
    return theta, psi, lambdas
#######################################################

# def calculate_HBasis(HprobsH, HprobsV, DprobsV):
#     theta = 2 * np.arccos(np.sqrt(HprobsH))
    
#     Sig = (np.sqrt(2)*(np.sqrt(HprobsV)) - np.cos(theta/2))/np.sin(theta/2)
#     if Sig > 0:
#         psi = 0
#     else:
#         psi = np.pi

#     Sig2 = (np.sqrt(2)*(np.sqrt(DprobsV)) - np.cos(psi)*np.cos(theta/2))/(np.cos(psi)*np.sin(theta/2))
#     if Sig2 > 0:
#         lambdas = 0
#     else:
#         lambdas = np.pi
        
#     return theta, psi, lambdas

def fidelity(theta, psi, lambdas):
    """ Compute the fidelity between two density matrices defined by unitary matrices. """
    # Define the unitary matrix T
    T =  np.array([[1, 0], [0, 1]])
    
    # Define the unitary matrix U based on the parameters

  
    U = np.array([
        [np.cos(theta/2), -np.exp(1j * lambdas)*np.sin(theta/2)],
        [np.exp(1j * psi) * np.sin(theta/2), np.exp(1j * (psi + lambdas)) * np.cos(theta/2)]
    ])
    print(U)
    fidelity = 1/4 * np.abs((np.trace(np.dot((np.conjugate(U).T),T))))**2
    print("Fidelity: ", fidelity)
    return fidelity

def main():
    # theta, psi, lambdas = calculate_HBasis(1, 0, .5)
    theta, psi, lambdas = calculate_HBasis_new(0.0, .97, .97)
    fid = fidelity(theta, psi, lambdas)

    # print(
    #     np.sin(theta/2)**2
    # )
    # print((np.abs(1/np.sqrt(2))*(np.cos(theta/2)+np.exp(1j * psi)*np.sin(theta/2)))**2)
    # print(
    #     np.abs(1/np.sqrt(2) * ( np.exp(1j*psi)*np.sin(theta/2)+np.exp(1j * (psi + lambdas) )*np.cos(theta)))**2
    # )
    
if __name__ == "__main__":
    main()