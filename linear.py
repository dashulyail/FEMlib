import classes
import numpy as np


def matrix_K(list_of_elements, list_of_points):
    """ построение глобальной матрицы жесткости
        с учетом граничных условий"""

    # построение пустой квадратной матрицы
    size = len(list_of_points)
    K = np.zeros((size * 3, size * 3))

    for elm in list_of_elements:
        if elm.__class__.__name__ == 'LineFE':
            #  k11 | k12
            #  k21 | k22
            
            # K_e = Q^T * K * Q
            # Q - матрица поворота
            K_e = np.transpose(elm.Q) @ np.array([
                [elm.E*elm.A/elm.L, 0, 0, -elm.E*elm.A/elm.L, 0, 0],
                [0, 12*elm.E*elm.I_y/elm.L**3, 6*elm.E*elm.I_y/elm.L**2,
                0, -12*elm.E*elm.I_y/elm.L**3, 6*elm.E*elm.I_y/elm.L**2],
                [0, 6*elm.E*elm.I_y/elm.L**2, 4*elm.E*elm.I_y/elm.L,
                0, -6*elm.E*elm.I_y/elm.L**2, 2*elm.E*elm.I_y/elm.L],
                [-elm.E*elm.A/elm.L, 0, 0, elm.E*elm.A/elm.L, 0, 0],
                [0, -12*elm.E*elm.I_y/elm.L**3, -6*elm.E*elm.I_y/elm.L**2,
                0, 12*elm.E*elm.I_y/elm.L**3, -6*elm.E*elm.I_y/elm.L**2],
                [0, 6*elm.E*elm.I_y/elm.L**2, 2*elm.E*elm.I_y/elm.L,
                0, -6*elm.E*elm.I_y/elm.L**2, 4*elm.E*elm.I_y/elm.L]
            ]) @ elm.Q
            # k11
            K[3*(elm.pnt[0].number-1):3*elm.pnt[0].number,
              3*(elm.pnt[0].number-1):3*elm.pnt[0].number] += \
                K_e[0:3,0:3]
            # k12
            K[3*(elm.pnt[0].number-1):3*elm.pnt[0].number,
              3*(elm.pnt[1].number-1):3*elm.pnt[1].number] += \
                K_e[0:3,3:6]
            # k21
            K[3*(elm.pnt[1].number-1):3*elm.pnt[1].number,
              3*(elm.pnt[0].number-1):3*elm.pnt[0].number] += \
                K_e[3:6,0:3]
            # k22
            K[3*(elm.pnt[1].number-1):3*elm.pnt[1].number,
              3*(elm.pnt[1].number-1):3*elm.pnt[1].number] += \
                K_e[3:6,3:6]

    for elm in list_of_elements:
        if elm.__class__.__name__ == 'LineTriFE':

            #  k11 | k12 | k13
            #  k21 | k22 | k23
            #  k31 | k32 | k33
            # с помощью метода matrix_B для каждого элемента
            # вычисляется матрица жесткости для треугольного плоского КЭ
            B, area = elm.matrix_B()
            K_e = np.transpose(B) @ elm.D @ B * elm.h * area
            # k11
            K[3*(elm.pnt[0].number-1):3*elm.pnt[0].number-1,
              3*(elm.pnt[0].number-1):3*elm.pnt[0].number-1] += \
                K_e[0:2,0:2]
            # k12
            K[3*(elm.pnt[0].number-1):3*elm.pnt[0].number-1,
              3*(elm.pnt[1].number-1):3*elm.pnt[1].number-1] += \
                K_e[0:2,2:4]
            # k13
            K[3*(elm.pnt[0].number-1):3*elm.pnt[0].number-1,
              3*(elm.pnt[2].number-1):3*elm.pnt[2].number-1] += \
                K_e[0:2,4:6]
            # k21
            K[3*(elm.pnt[1].number-1):3*elm.pnt[1].number-1,
              3*(elm.pnt[0].number-1):3*elm.pnt[0].number-1] += \
                K_e[2:4,0:2]
            # k22
            K[3*(elm.pnt[1].number-1):3*elm.pnt[1].number-1,
              3*(elm.pnt[1].number-1):3*elm.pnt[1].number-1] += \
                K_e[2:4,2:4]
            # k23
            K[3*(elm.pnt[1].number-1):3*elm.pnt[1].number-1,
              3*(elm.pnt[2].number-1):3*elm.pnt[2].number-1] += \
                K_e[2:4,4:6]
            # k31
            K[3*(elm.pnt[2].number-1):3*elm.pnt[2].number-1,
              3*(elm.pnt[0].number-1):3*elm.pnt[0].number-1] += \
                K_e[4:6,0:2]
            # k32
            K[3*(elm.pnt[2].number-1):3*elm.pnt[2].number-1,
              3*(elm.pnt[1].number-1):3*elm.pnt[1].number-1] += \
                K_e[4:6,2:4]
            # k33
            K[3*(elm.pnt[2].number-1):3*elm.pnt[2].number-1,
              3*(elm.pnt[2].number-1):3*elm.pnt[2].number-1] += \
                K_e[4:6,4:6]


    for i,s in enumerate(K):
    # проверка на наличие пустых строки
        if s.any() == False:
        # false если все значения false или 0
            K[i,i] = 1
    
    # присвоение граничных условий
    for pnt in list_of_points:
        if pnt.boundary_cond != None:
            if pnt.boundary_cond.count(1):
                K[3*(pnt.number-1)] = 0 # зануление строки
                K[:, 3*(pnt.number-1)] = 0 # зануление столбца
                K[3*(pnt.number-1), 3*(pnt.number-1)] = 1
            if pnt.boundary_cond.count(3):
                K[3*(pnt.number-1)+1] = 0 # зануление строки
                K[:, 3*(pnt.number-1)+1] = 0 # зануление столбца
                K[3*(pnt.number-1)+1, 3*(pnt.number-1)+1] = 1
            if pnt.boundary_cond.count(5):
                K[3*(pnt.number-1)+2] = 0 # зануление строки
                K[:, 3*(pnt.number-1)+2] = 0 # зануление столбца
                K[3*(pnt.number-1)+2, 3*(pnt.number-1)+2] = 1
    return K
