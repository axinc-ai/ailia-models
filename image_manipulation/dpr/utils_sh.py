'''
    construct shading using sh basis
'''
import numpy as np


def SH_basis(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:, 0]
    norm_Y = normal[:, 1]
    norm_Z = normal[:, 2]

    sh_basis = np.zeros((numElem, 9))
    att = np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])
    sh_basis[:, 0] = 0.5 / np.sqrt(np.pi) * att[0]

    sh_basis[:, 1] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Y * att[1]
    sh_basis[:, 2] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_Z * att[1]
    sh_basis[:, 3] = np.sqrt(3) / 2 / np.sqrt(np.pi) * norm_X * att[1]

    sh_basis[:, 4] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_X * att[2]
    sh_basis[:, 5] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_Y * norm_Z * att[2]
    sh_basis[:, 6] = np.sqrt(5) / 4 / np.sqrt(np.pi) * (3 * norm_Z ** 2 - 1) * att[2]
    sh_basis[:, 7] = np.sqrt(15) / 2 / np.sqrt(np.pi) * norm_X * norm_Z * att[2]
    sh_basis[:, 8] = np.sqrt(15) / 4 / np.sqrt(np.pi) * (norm_X ** 2 - norm_Y ** 2) * att[2]

    return sh_basis


def get_shading(normal, SH):
    '''
        get shading based on normals and SH
        normal is Nx3 matrix
        SH: 9 x m vector
        return Nxm vector, where m is the number of returned images
    '''
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)

    return shading
