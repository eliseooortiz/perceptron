import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib as mpl

from perceptron import Perceptron


class Ventana:
    puntos, clase_deseada = np.array([]), []
    sin_evaluar=np.array([])
    perceptron=None
    epoca_actual=0
    epocas_maximas=0
    rango=0.1
    rango_inicializado=False
    pesos_inicializados=False
    perceptron_entrenado=False
    linea=None
    texto_de_epoca = None
    termino=False
    def __init__(self):
        mpl.rcParams['toolbar'] = 'None'
        self.fig, self.grafica_perceptron = plt.subplots()
        self.fig.canvas.set_window_title('Perceptron')
        self.fig.set_size_inches(10, 6, forward=True)
        plt.subplots_adjust(bottom=0.3)
        self.grafica_perceptron.set_xlim(-1.0,1.0)
        self.grafica_perceptron.set_ylim(-1.0,1.0)
        self.fig.suptitle("Algoritmo del perceptron")
        self.grafica_perceptron.set_title("Perceptron")
        

        cordenadas_rango = plt.axes([0.275, 0.15, 0.175, 0.05])
        coordenadas_epcoas = plt.axes([0.7125, 0.15, 0.15, 0.05])
        coordenadas_pesos = plt.axes([0.225, 0.05, 0.125, 0.05])
        coordenadas_entrenar = plt.axes([0.4625, 0.05, 0.1, 0.05])
        coordenadas_evaluar = plt.axes([0.7, 0.05, 0.1, 0.05])
        self.text_box_rango = TextBox(cordenadas_rango, "Rango de aprendizaje:")
        self.text_box_epocas = TextBox(coordenadas_epcoas, "Epocas maximas:")
        boton_pesos = Button(coordenadas_pesos, "Inicializar pesos")
        boton_entrenar = Button(coordenadas_entrenar, "Entrenar")
        boton_evaluar = Button(coordenadas_evaluar, "Evaluar")
        self.text_box_epocas.on_submit(self.validar_epocas)
        self.text_box_rango.on_submit(self.validar_rango)
        boton_pesos.on_clicked(self.inicializar_pesos)
        boton_entrenar.on_clicked(self.entrenar_perceptron)
        boton_evaluar.on_clicked(self.evaluar)
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()
    def __onclick(self, event):
        if event.inaxes == self.grafica_perceptron:
            current_point = [event.xdata, event.ydata]
            if self.perceptron_entrenado:
                self.grafica_perceptron.plot(event.xdata, event.ydata,'k.')
                self.sin_evaluar=np.append(self.sin_evaluar, [event.xdata, event.ydata]).reshape([len(self.sin_evaluar) + 1, 2])
            else:
                self.puntos = np.append(self.puntos, current_point).reshape([len(self.puntos) + 1, 2])
                is_left_click = event.button == 1               
                self.clase_deseada.append(0 if is_left_click else 1)
                self.grafica_perceptron.plot(event.xdata, event.ydata, 'b.' if is_left_click else 'rx')
            self.fig.canvas.draw()
    def entrenar_perceptron(self, event):
        if self.pesos_inicializados and not self.perceptron_entrenado:
            while not self.termino and self.epoca_actual < self.perceptron.epocas_maximas:
                self.termino = True
                self.epoca_actual += 1
                for i, x in enumerate(self.puntos):
                    x = np.insert(x, 0, -1.0)
                    error = self.clase_deseada[i] - self.perceptron.pw(x)
                    if error != 0:
                        self.termino = False
                        self.perceptron.pesos = \
                            self.perceptron.pesos + np.multiply((self.perceptron.rango * error), x)
                        self.graficar_linea()
            self.grafica_perceptron.text(-0.25, 0.9,
                              'Convergio' if self.termino else 'No convergio',
                              fontsize=10)
            self.texto_de_epoca.set_text("epocas: %s" % self.epoca_actual)
            plt.pause(0.1)
            self.perceptron_entrenado = True

    
    def evaluar(self,event):
        if(self.perceptron_entrenado and len(self.sin_evaluar)>0):
            self.grafica_perceptron.clear()     
            self.grafica_perceptron.set_title("Perceptron")   
            self.grafica_perceptron.set_xlim(-1.0,1.0)
            self.grafica_perceptron.set_ylim(-1.0,1.0)
            for j,k in enumerate(self.puntos):
                self.grafica_perceptron.plot(k[0], k[1], 'b.' if not self.clase_deseada[j] else 'rx')            
            self.grafica_perceptron.plot(self.linea.get_xdata(),self.linea.get_ydata(), 'y-')
            self.grafica_perceptron.text(0.5, 0.8,'Epoca: %s' % self.epoca_actual, fontsize=10)
            
            for  i,x in enumerate(self.sin_evaluar):
                x = np.insert(x, 0, -1.0)
                self.grafica_perceptron.plot(x[1], x[2],
                                    'gx' if self.perceptron.pw(x)
                                    else 'g.')
            plt.pause(0.1)
        
    def inicializar_pesos(self, event):
        if self.rango_inicializado and self.epocas_maximas>0 and len(self.puntos)>0 and not self.perceptron_entrenado:
            self.perceptron = Perceptron(self.rango, self.epocas_maximas, [-1.0,1])
            self.perceptron.inicializar_pesos()
            self.pesos_inicializados = True
            self.graficar_linea()

        
    
    def graficar_linea(self):
        x1 = np.array([self.puntos[:, 0].min() - 2, self.puntos[:, 0].max() + 2])
        m = -self.perceptron.pesos[1] / self.perceptron.pesos[2]
        c = self.perceptron.pesos[0] / self.perceptron.pesos[2]
        x2 = m * x1 + c
        
        if not self.linea:
            self.linea, = self.grafica_perceptron.plot(x1, x2, 'y-')
            self.texto_de_epoca = self.grafica_perceptron.text(0.5, 0.8,
                                                        'Epoca: %s' % self.epoca_actual,
                                                        fontsize=10)
        else:
            self.linea.set_xdata(x1)
            self.linea.set_ydata(x2)
            self.texto_de_epoca.set_text('Epoca: %s' % self.epoca_actual)
        self.fig.canvas.draw()
        plt.pause(0.1)

    def validar_rango(self, expression):

        try:
            r=float(expression)
            if(r>0 and r<1):
                self.rango =float(expression)
            else:
                self.rango =0.1    
        except ValueError:
            self.rango =0.1
        finally:
            self.text_box_rango.set_val(self.rango)
            self.rango_inicializado=True

    def validar_epocas(self, expression):
        try:
            self.epocas_maximas =int(expression)
        except ValueError:
            self.epocas_maximas =50
        finally:
            self.text_box_epocas.set_val(self.epocas_maximas)
        


if __name__ == '__main__':
    Ventana()