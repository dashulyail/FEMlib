import main
import numpy as np


class LineTriFE:
    """ Класс описывающий поведение конечных элементов """
    def __init__(self, number, E, v, h, *pnt):
        """присвоение характеристик при создании объекта"""
        # номер конечного элемента
        self.number = number
        # список узлов принадлежащих КЭ
        self.pnt = pnt
        # модуль Юнга
        self.E = E
        # коэффициент Пуассона
        self.v = v
        # толщина конечного элемента
        self.h = h
        #начальное значение напряжений
        self.sigma_x = 0
        self.sigma_y = 0
        self.tau_xy = 0
        # присвоение матрицы упругости D конечному элементу
        if main.TASK == 'plane stress':
                    self.D=self.E/(1-self.v**2)*\
                    np.array([[1,self.v,0],
                       [self.v,1,0],
                       [0,0,(1-self.v)/2]])
        if main.TASK == 'plane strain':
                    self.D=self.E/(1+self.v)/(1-2*self.v)*\
                    np.array([[1-self.v,self.v,0],
                             [self.v,1-self.v,0],
                             [0,0,(1-2*self.v)/2]])


    def matrix_B(self):
        """Построение матрицы функций формы и вычисление площади КЭ"""
        a = self.pnt[1].x - self.pnt[0].x
        b = self.pnt[0].y - self.pnt[1].y
        c = self.pnt[0].x - self.pnt[2].x
        d = self.pnt[2].y - self.pnt[0].y
        s = -b - d
        t = -a - c
        area = (a*d - b*c)/2
        B = np.array([[s, 0, d, 0, b, 0],
                      [0, t, 0, c, 0, a],
                      [t, s, c, d, a, b]])/2/area
        return B, area


    def define_stress(self):
        """функция определяет от полученных перемещений системы
            напряженя возникающий в КЭ"""
        if len(self.pnt) == 3:
            '''Вычисление напряжений в 3-х узловых
                конечных элементах'''
            # определение геометрических параметров
            B = self.matrix_B()[0]
            # вычисление приращения напряжений в линейной постановке
            d_sigma = self.D @ B @ np.array(
                [[self.pnt[0].displace_x,
                self.pnt[0].displace_y,
                self.pnt[1].displace_x,
                self.pnt[1].displace_y,
                self.pnt[2].displace_x,
                self.pnt[2].displace_y]]).reshape(6,1)
            self.sigma_x = d_sigma[0,0]
            self.sigma_y = d_sigma[1,0]
            self.tau_xy = d_sigma[2,0]


class LineFE:
    """Класс, описывающий поведение стержневых элементов"""
    # TODO: дописать возможность врезки шарниров
    # TODO: дописать метод вычисления внутренних усилий в КЭ
    # TODO: переписать класс для различных параметрических сечений
    # сейчас реализованы только прямоугольные сечения КЭ
    def __init__(self, number, E, b, h, *pnt):
        self.number = number # Номер элемента
        self.E = E # Модуль упругости
        self.b = b # ширина элемента
        self.h = h # высота элемента
        self.A = self.b*self.h # площадь элемента
        self.I_y = (self.b*self.h**3)/12 # Момент инерции
        self.pnt = pnt # список объектов узлов
        # вычисление длины КЭ
        self.L = np.sqrt((self.pnt[1].x - self.pnt[0].x)**2\
                         + (self.pnt[1].y - self.pnt[0].y)**2)
        # угол поворота стержня
        self.phi = np.arctan2(self.pnt[1].y - self.pnt[0].y, self.pnt[1].x - self.pnt[0].x)
        # Временные переменные для вычисления матрицы поворота стержня
        sinPhi = np.sin(self.phi)
        cosPhi = np.cos(self.phi)
        # Матрица поворота стержня
        self.Q = np.array([[cosPhi, sinPhi, 0, 0, 0, 0],
                           [-sinPhi, cosPhi, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, cosPhi, sinPhi, 0],
                           [0, 0, 0, -sinPhi, cosPhi, 0],
                           [0, 0, 0, 0, 0, 1]])


class Point:
    """Класс описывающий поведение узлов"""
    def __init__(self, number, x, y,
                boundary_cond):
        # номер узла
        self.number = number
        # коордиаты узла
        self.x = x
        self.y = y
        # полное перемещение от внешних сил
        self.displace_x = 0
        self.displace_z = 0
        self.rotate_y = 0
        # граничные условия в формате (1, 3)
        # где 1 - гор. связь; 3 - верт. связь
        self.boundary_cond = boundary_cond

# пока что не включал эту матрицу
def make_q(le, lp):
    """ построение матрицы внутренних сил """
    length = len(lp)
    q = np.zeros(length*2).reshape(length*2,1)
    for elm in le:
        B, area = elm.matrix_B()
        q_elm = B.transpose() @\
            np.array(
                [elm.sigma_x + elm.d_sigma_x,
                elm.sigma_y + elm.d_sigma_y,
                elm.tau_xy + elm.d_tau_xy])*area*elm.h
        for k, point in enumerate(elm.pnt):
            assert np.shape(q[2*(point.num-1):2*point.num]) == (2,1) # нарушена размерность
            assert np.shape(q_elm.reshape(6,1)[k*2:2*(k+1)]) == (2,1) # нарушена размерность
            q[2*(point.num-1):2*point.num] += q_elm.reshape(6,1)[k*2:2*(k+1)]
    for point in lp:
        if point.boundary_cond == (1,): q[2*(point.num-1)] = 0
        if point.boundary_cond == (3,): q[2*point.num-1] = 0
        if point.boundary_cond == (1,3):
            q[2*(point.num-1)] = 0
            q[2*point.num-1] = 0 
    return q
