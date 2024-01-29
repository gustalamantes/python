def announce(f):
    def wrapper():
        print("About to run the function...")
        f()
        print("Done running the function.")
    return wrapper

def anuncio(f):
    def wrapper():
        print("Ingresa un nombre")
        f()
        print("Nombre agregado")
    return wrapper

@announce
def hello():
    print("Hello, world!")

hello()

@anuncio
def captura():
    input("Nombre: ")

captura()