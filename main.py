import classes
import linear
import numpy as np
TASK = 'plane stress'


def output(list_of_elements, list_of_points):
    """простейший вывод результатов вычислений на экран"""
    # Вывод заголовка и шапки таблицы
    print('\n')
    '''print('{:^40}'.format('Таблица усилий в элементах:'))
    print('{:<5}|{:>10}|{:>10}|{:>10}'
            .format('№ эл.', 'Nx (т/м2)',
                    'Ny (т/м2)', 'Txy (т/м2)'))
    print('{:_^40}'.format(''))
    # вывод результатов вычислений
    for elm in list_of_elements:
        print('{:<5}|{:>10.2f}|{:>10.2f}|{:>10.2f}'
        .format(elm.number, elm.sigma_x,
                elm.sigma_y,elm.tau_xy))
    print('\n')'''
    # Вывод заголовка и шапки таблицы
    print('{:^30}'.format('Таблица перемещений узлов:'))
    print('{:<5}|{:>10}|{:>10}|{:>10}'
        .format('№ уз.', 'X (мм)', 'Z (мм)','Uy (рад*1000)'))
    print('{:_^30}'.format(''))
    # вывод результатов вычислений
    for pnt in list_of_points:
        print('{:<5}|{:>10.4f}|{:>10.4f}|{:>10.4f}'
            .format(pnt.number, pnt.displace_x*1000,
                    pnt.displace_z*1000, pnt.rotate_y*1000))


if __name__ == '__main__':
    # зададим список узлов в формате classes.Point(number, x, y, boundary_cond)
    lp = list([
            classes.Point(1,0,0,(1, 3)), classes.Point(2,1,0,(1, 3)),
            classes.Point(3,0,1,None), classes.Point(4,1,1,None),
            ])
    # зададим список узлов в формате:
    # classes.LineTriFE(number, E, v, h, *pnt) для плоских КЭ
    # classes.LineFE(number, E, b, h, *pnt) для стержневых КЭ
    le = list([
            classes.LineTriFE(1,2.9e6,0.2,0.1,lp[0],lp[1],lp[3]),
            classes.LineTriFE(2,2.9e6,0.2,0.1,lp[0],lp[3],lp[2]),
            classes.LineFE(3,2.9e6,0.2,0.2,lp[2],lp[3])
            ])
    # вектор внешних сил
    P = np.array([0,0,0,
                  0,0,0,
                  200,0,0,
                  0,0,0])
    # вычисляем глобальную матрицу жесткости
    K = linear.matrix_K(le, lp)
    # вычисляем вектор линейные перемещения узлов U = K^-1 P
    U = np.linalg.solve(K, P.transpose())
    # присваиваем полученные перемещения классу "узел"
    for i in lp:
        i.displace_x += U[3 * (i.number-1)]
        i.displace_z += U[3 * (i.number-1)+1]
        i.rotate_y += U[3 * (i.number-1)+2]
    output(le, lp)