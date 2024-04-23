#////////////////////////////// IMPORTS //////////////////////////////////////////////
import numpy as np
import random
from sympy import Matrix
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.spinner import Spinner

    #////////////////////////////// EXCEPTIONS //////////////////////////////////////////////

class MultiplicacionInvalidaDeMatriz(Exception):
    def __init__(self):
        super().__init__( f"Se esta multiplicando una matriz de un tamaño 1 x n columnas." )

class MatrizNoEsInvertible(Exception):
    def __init__(self):
        super().__init__( f"La invertible es 0, por lo tanto no existe (No es cuadrada o son productos iguales)" )

class MensajeVacio(Exception):
    def __init__(self):
        super().__init__( f"El mensaje no tiene nada, esta vacio." )

class CaracteresInvalidosMensaje(Exception):
    def __init__(self):
        super().__init__( f"El mensaje tiene caracteres que no están definidos en el alfabeto." )

class MensajeNoEsString(Exception):
    def __init__(self):
        super().__init__( f"El mensaje es un tipo de dato diferente de str." )

class SizeInvalida(Exception):
    def __init__(self):
        super().__init__( f"El valor dado para Size es diferente de int." )

class ClaveDemasiadoGrande(Exception):
    def __init__(self):
        super().__init__( f"El size ingresado para la clave es demasiado grande." )

#////////////////////////////// FUNCTIONS //////////////////////////////////////////////

class EncryptionApp(App):
    def build(self):
        # GridLayout = Matrix de filas de columnas con widgets (botones, labels, etc.)
        # Cols o Rows, el orden de la matrix en el GridLayout
        # Padding = El espacio entre el Contenido_Error del widget y su borde (entre más grande, más pequeño sera el borde del widget)
        # Spacing = El tamaño entre cada widget (entre más grande, más separado estara cada widget)

        ContenedorPrincipal = GridLayout(cols = 1,padding=20,spacing=10)

        # Label y text input con el mensaje.
        ContenedorPrincipal.add_widget( Label(text="Mensaje", font_size = 20) )
        self.Input_Mensaje = TextInput(font_size = 24, multiline = False)
        ContenedorPrincipal.add_widget(self.Input_Mensaje)

        # Conectar con el callback con el evento on_text_validate para determinar si el mensaje es valido.
        self.Input_Mensaje.bind( on_text_validate = self.Validar_Mensaje)

        #---------------------------------- Boton - Label Generar Clave -------------------------------------------

        Btn_clave = Button(text="Generar Clave",font_size = 40)
        ContenedorPrincipal.add_widget(Btn_clave)

        # Conectar con el callback con el evento press del boton de generar clave.
        Btn_clave.bind( on_press= self.GenerarClaveAutomatica )

        """
        ContenedorPrincipal.add_widget( Label(text="Clave generada", font_size = 20) )
        self.Input_Clave = TextInput(font_size=17)
        ContenedorPrincipal.add_widget(self.Input_Clave)
        """

        # Crear el grid para la clave
        ContenedorClave = GridLayout(cols=2)

        # Para la clave: size_hint = (0.7, 1)  Ancho = 70%, Altura = 100%
        # Para el tamaño: size_hint = (0.3, 1)  Ancho = 30%, Altura = 100%

        # Crear y agregar los labels para la clave
        ContenedorClave.add_widget( Label(text="Clave", font_size = 20, size_hint = (0.7, 1) ))

        ContenedorClave.add_widget( Label(text="Tamano Clave", font_size = 20, size_hint = (0.3, 1) ) )

        # Crea y agrega el TextInput
        self.Input_Clave = TextInput(multiline = False, size_hint = (0.7, 1), font_size = 12)
        ContenedorClave.add_widget(self.Input_Clave)

        # Crea y configura el Spinner
        self.ListaTamanoClave = Spinner(text="2", values=("2", "3", "4", "5"), size_hint = (0.3, 1))
        ContenedorClave.add_widget(self.ListaTamanoClave)

        ContenedorPrincipal.add_widget(ContenedorClave)
        #---------------------------------- Boton - Label Encriptar --------------------------------------------

        Btn_encriptar = Button(text="Encriptar",font_size=40)
        ContenedorPrincipal.add_widget(Btn_encriptar)

        # Conectar con el callback con el evento press del boton encriptar.
        Btn_encriptar.bind( on_press= self.EncriptarMensaje )

        ContenedorPrincipal.add_widget( Label(text="Mensaje Encriptado", font_size = 20) )
        self.Input_MensajeEncriptado = TextInput(font_size=24)
        ContenedorPrincipal.add_widget(self.Input_MensajeEncriptado)

        #---------------------------------- Boton - Label Desencriptar ------------------------------------------

        Btn_desencriptar = Button(text="Desencriptar",font_size=40)
        ContenedorPrincipal.add_widget(Btn_desencriptar)

        # Conectar con el callback con el evento press del boton encriptar.
        Btn_desencriptar.bind( on_press= self.DesencriptarMensaje )

        ContenedorPrincipal.add_widget( Label(text="Mensaje Desencriptado", font_size = 20) )
        self.Input_MensajeDesencriptado = TextInput(font_size=24)
        ContenedorPrincipal.add_widget(self.Input_MensajeDesencriptado)

        # Size Hint (widht, height) = representa el espacio que se quiere utilizar en su totalidad 1 = 100% de la Window screen.
        # Size = tamaño que va tomar, en este caso el tamaño de la Window screen de ancho y alto.
        Scroll_ContenedorPrincipal = ScrollView(size_hint=(1, 1), size=(Window.width, Window.height))
        Scroll_ContenedorPrincipal.add_widget(ContenedorPrincipal)

        # Se retorna el widget o "raiz" que contiene a todos los demás
        return Scroll_ContenedorPrincipal
    
    #////////////////////////////// CONSTANTS //////////////////////////////////////////////

    Diccionario_encrypt = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 
                        'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, '0': 26, '1': 27, '2': 28, '3': 29, '4': 30,
                        '5': 31, '6': 32, '7': 33, '8': 34, '9': 35, '.': 36, ',': 37, ':': 38, '?': 39, ' ': 40, 'a': 41, 'b': 42, 'c': 43, 'd': 44, 'e': 45,
                        'f': 46, 'g': 47, 'h': 48, 'i': 49, 'j': 50, 'k': 51, 'l': 52, 'm': 53, 'n': 54, 'o': 55, 'p': 56, 'q': 57, 'r': 58, 's': 59, 't': 60,
                        'u': 61, 'v': 62, 'w': 63, 'x': 64, 'y': 65, 'z': 66}

    Diccionario_decrypt = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J', '10': 'K', '11': 'L', '12': 'M',
                        '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y',
                        '25': 'Z', '26': '0', '27': '1', '28': '2', '29': '3', '30': '4', '31': '5', '32': '6', '33': '7', '34': '8', '35': '9', '36': '.',
                        '37': ',', '38': ':', '39': '?', '40': ' ', '41': 'a', '42': 'b', '43': 'c', '44': 'd', '45': 'e', '46': 'f', '47': 'g', '48': 'h',
                        '49': 'i', '50': 'j', '51': 'k', '52': 'l', '53': 'm', '54': 'n', '55': 'o', '56': 'p', '57': 'q', '58': 'r', '59': 's', '60': 't',
                        '61': 'u', '62': 'v', '63': 'w', '64': 'x', '65': 'y', '66': 'z'}

    Modulo = 67

    #////////////////////////////// FUNCTIONS //////////////////////////////////////////////
        
    def hill_genkey(size):
        """
        Hill Key Generation
        :size: matrix size
        :return: size x size matrix containing the key
        """

        if type(size) != int:
            raise SizeInvalida

        if size > 100:
            raise ClaveDemasiadoGrande

        matrix = []

        L = []

        # Relleno una lista con tantos valores aleatorios como elementos a rellenar en la matriz determinada por size (size * size)

        for x in range(size * size):
            L.append(random.randrange(40))


        # Se crea la matrix clave con los valores generados, de tamaño size * size

        matrix = np.array(L).reshape(size, size)

        return matrix


    def hill_cipher(self, message, key):
        """
        Hill cipher
        :message: message to cipher (plaintext)
        :key: key to use when ciphering the message (as it is returned by
            uoc_hill_genkey() )
        :return: ciphered text
        """
        #print(key)

        ciphertext = ''

        # Variables

        matrix_mensaje = []
        list_temp = []
        cifrado_final = ''
        ciphertext_temp = ''
        cont = 0

        if type(message) != str:
            raise MensajeNoEsString

        # Si el tamaño del mensaje es menor o igual al tamaño de la clave

        if len(message) <= len(key):

            # Convertir el tamaño del mensaje al tamaño de la clave, si no es igual, se añaden 'X' hasta que sean iguales los tamaños.

            while len(message) < len(key):
                message = message + 'X'

            # Crear la matriz para el mensaje

            for i in range(0, len(message)):
                try:
                    matrix_mensaje.append(self.Diccionario_encrypt[message[i]])
                except KeyError:
                    raise CaracteresInvalidosMensaje

            # Se crea la matriz

            matrix_mensaje = np.array(matrix_mensaje)

            # Se multiplica la matriz clave por la de mensaje

            cifrado = np.matmul(key, matrix_mensaje)

            # Se obtiene el modulo sobre el diccionario de cada celda

            cifrado = cifrado % self.Modulo

            # Se codifica de valores numericos a los del diccionario, añadiendo a ciphertext el valor en el diccionario pasandole como indice la i posicion de la variable cifrado

            for i in range(0, len(cifrado)):
                ciphertext += self.Diccionario_decrypt[str(cifrado[i])]
        else:

        # Si el tamaño del mensaje es menor o igual al tamaño de la clave

            # Si al dividir en trozos del tamaño de la clave, existe algun trozo que tiene menos caracteres que la long. de la clave se añaden tantas 'X' como falten

            while len(message) % len(key) != 0:
                message = message + 'X'

            # Se troce el mensaje en subsstrings de tamaño len(key) y se alamcenan como valores de un array

            matrix_mensaje = [message[i:i + len(key)] for i in range(0,
                            len(message), len(key))]

            # Para cada valor del array (grupo de caracteres de la longitud de la clave)

            for bloque in matrix_mensaje:

                # Crear la matriz para el bloque

                for i in range(0, len(bloque)):
                    list_temp.append(self.Diccionario_encrypt[bloque[i]])

                # Se crea la matriz de ese bloque

                matrix_encrypt = np.array(list_temp)

                # Se multiplica la matriz clave por la del bloque
                try:
                    cifrado = np.matmul(key, matrix_encrypt)
                except Exception:
                    raise MultiplicacionInvalidaDeMatriz

                # Se obtiene el modulo sobre el diccionario de cada celda

                cifrado = cifrado % self.Modulo

                # Se codifica de valores numericos a los del diccionario, añadiendo a ciphertext el valor en el diccionario pasandole como indice la i posicion de la variable cifrado

                for i in range(0, len(cifrado)):
                    ciphertext_temp += self.Diccionario_decrypt[str(cifrado[i])]

                # Se inicializan las variables para el nuevo bloque

                matrix_encrypt = []
                list_temp = []

            # Se añade el mensaje encriptado a la variable que contiene el mensaje encriptado completo

            ciphertext = ciphertext_temp

        # --------------------------------

        return ciphertext


    def hill_decipher(self, message, key):
        """
        Hill decipher
        :message: message to decipher (ciphertext)
        :key: key to use when deciphering the message (as it is returned by
            uoc_hill_genkey() )
        :return: plaintext corresponding to the ciphertext
        """

        plaintext = ''

        matrix_mensaje = []
        plaintext_temp = ''
        list_temp = []
        matrix_inversa = []
        matrix_mensaje = [message[i:i + len(key)] for i in range(0,
                        len(message), len(key))]

        # Se calcula la matriz inversa aplicando el modulo
        try:
            matrix_inversa = Matrix(key).inv_mod(self.Modulo)
        except ValueError:
            raise MatrizNoEsInvertible

        # Se transforma en una matriz

        matrix_inversa = np.array(matrix_inversa)

        # Se pasan los elementos a float

        matrix_inversa = matrix_inversa.astype(float)

        # Para cada bloque

        for bloque in matrix_mensaje:

            # Se encripta el mensaje encriptado

            for i in range(0, len(bloque)):
                list_temp.append(self.Diccionario_encrypt[bloque[i]])

            # Se convierte a matriz

            matrix_encrypt = np.array(list_temp)

            # Se multiplica la matriz inversa por el bloque

            cifrado = np.matmul(matrix_inversa, matrix_encrypt)

            # Se le aplica a cada elemento el modulo

            cifrado = np.remainder(cifrado, self.Modulo).flatten()

            # Se desencripta el mensaje

            for i in range(0, len(cifrado)):
                plaintext_temp += self.Diccionario_decrypt[str(int(cifrado[i]))]

            matrix_encrypt = []
            list_temp = []
        plaintext = plaintext_temp

        # Se eleminan las X procedentes de su addicion en la encriptacion para tener bloques del tamaño de la clave

        try:
            while plaintext[-1] == 'X':
                plaintext = plaintext.rstrip(plaintext[-1])
        except IndexError:
            raise MensajeVacio

        return plaintext

    # instance es el widget que generó el evento
    # Value es el valor actual que tiene el widget

    def GenerarClaveAutomatica(self, value):
        Tamano_Clave = int(self.ListaTamanoClave.text)
        Clave = self.hill_genkey(Tamano_Clave)
        #Transformando np.matrix a una lista.
        Clave = np.matrix.tolist(Clave)

        Cadena = str(Clave)
        # Eliminar los corchetes
        ClaveString = Cadena.replace('[', '').replace(']', '')
        self.Input_Clave.text = ClaveString

    def EncriptarMensaje(self, value):

        # Verificar si la clave y el mensaje son validos.
        MensajeValido = self.Validar_Mensaje(value)
        ClaveValida = self.Validar_Clave(value)

        if MensajeValido and ClaveValida:
            try:
                # Realizar la encriptacion.
                Clave = self.Generar_Clave(value)
                Mensaje = self.Input_Mensaje.text

                MensajeEncriptado = self.hill_cipher(Mensaje, Clave)
                self.Input_MensajeEncriptado.text = MensajeEncriptado

            except Exception as err:
                return self.Mostrar_error( err )

    def DesencriptarMensaje(self, value):
        try:
            MensajeEncriptado = self.Input_MensajeEncriptado.text
            Clave = self.Generar_Clave(value)
            MensajeDesencriptado = ""

            # Verificar si hay algo en como mensaje encriptado de lo contrario, se toma
            # El mensaje como la clave.
            if MensajeEncriptado != "":
                # Realizar la desencriptacion.
                MensajeDesencriptado = self.hill_decipher(MensajeEncriptado, Clave)
            else:
                MensajeValido = self.Validar_Mensaje(value)

                if MensajeValido:
                    Mensaje = self.Input_Mensaje.text
                    MensajeDesencriptado = self.hill_decipher(Mensaje, Clave)

            self.Input_MensajeDesencriptado.text = MensajeDesencriptado

        except Exception as err:
            return self.Mostrar_error( err )
        
    def Mostrar_error( self, err ):
        Contenido_Error = GridLayout(cols = 1)
        Contenido_Error.add_widget( Label(text= str(err) ) )
        Btn_cerrar = Button(text="Cerrar" )
        Contenido_Error.add_widget( Btn_cerrar )
        Popup_widget = Popup(title="Error", content = Contenido_Error)
        Btn_cerrar.bind( on_press= Popup_widget.dismiss)
        Popup_widget.open()
        return False

    def Validar_Mensaje(self, value):
        """
        Verificar si el mensaje tiene caracteres validos para la encriptacion,
        es decir si sus caracteres estan en el diccionario de letras en Encryption Logic

        """
        try:
            Mensaje = self.Input_Mensaje.text  # El texto del TextInput
            Diccionario_encrypt_ref = self.Diccionario_encrypt  # Diccionario de encriptación

            # Recorrer cada caracter del mensaje y verificar si se encuentra en el diccionario.
            for caracter in Mensaje:
                if caracter not in Diccionario_encrypt_ref.keys():
                    raise Exception(f"Caracter no válido: {caracter}")
                
            # Verificar si el mensaje esta vacio.
            if len(Mensaje) == 0:
                raise Exception(f"El mensaje no tiene nada, esta vacio." )
            
            # Verificar que el mensaje tenga minimo 1 elemento.
            if len(Mensaje) == 1:
                raise Exception(f"El mensaje apenas tiene 1 elemento, tiene que ser más que una letra o numero." )
            
            return True
        
        except Exception as err:
            return self.Mostrar_error(err)

    def Validar_Clave(self, value):
        """
        Funcion que verifica si la clave es valida para generarse.
        """
        TamanoClave = self.ListaTamanoClave.text
        Clave = self.Input_Clave.text

        try:
            # Verificar que la clave tenga algo.
            if len(Clave) == 0:
                raise Exception(f"La clave no tiene nada, esta vacia." )
            
            # Obteniendo el tamaño de la matrix
            Tamano_matrix = int(TamanoClave) ** 2
            # Obteniendo la cantidad de enteros en la clave
            Cnt_enteros = 0

            # Seperar todos los elementos en una lista, se separa cuando se encuentre una coma.
            Lista_string = Clave.split(",")
            Lista_string = [elemento.strip() for elemento in Lista_string] # Eliminando todos los espacios vacios que hayan.

            # Obtener la cantidad de elementos y verificar si es un entero.
            for Elemento in Lista_string:
                if Elemento.isdigit():
                    Cnt_enteros += 1
                else:
                    raise ValueError(f"Hay elementos que no son entero o están vacios.")

            # Verificar que la clave tenga la cantidad suficiente de elementos.
            if Cnt_enteros != Tamano_matrix:
                raise ValueError(f"Elementos incompletos (se necesitan {Tamano_matrix} enteros y {Tamano_matrix-1} comas) NO puede ser mayor o menor.")

            return True
        
        except Exception as err:
            return self.Mostrar_error(err)

    def Generar_Clave(self, value):
        TamanoClave = int(self.ListaTamanoClave.text)
        Clave = self.Input_Clave.text

        # Transformar cada elemento a entero.
        Lista_string = Clave.split(",")
        Lista_enteros = [int(elemento) for elemento in Lista_string]

        Clave_Matrix = []

        for _ in range(TamanoClave):
            ListaFila = []
            for _ in range(TamanoClave):
                Entero = Lista_enteros.pop(0)
                ListaFila.append(Entero)
            # Append the row to the Key
            Clave_Matrix.append(ListaFila)

        return Clave_Matrix


if __name__ == "__main__":
    EncryptionApp().run()
