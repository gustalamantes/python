class point():
    def __init__(self, arg1, arg2):
        self.x = arg1
        self.y = arg2

p = point(2,8)
print(p.x)
print(p.y)


class Vuelo():
    def __init__(self, capacidad):
        self.capacidad = capacidad
        self.pasajeros = []

    def add_passenger(self, name):
        if not self.open_seats():
            return False
        self.pasajeros.append(name)
        return True

    def open_seats(self):
        return self.capacidad - len(self.pasajeros)
    
    
vuelo = Vuelo(3)
people = [{"ind":1, "name":"Juan"},{"ind":2,"name":"Pedro"},{"ind":3,"name":"Jose"},{"ind":4,"name":"Raul"},{"ind":5,"name":"Hugo"}]
people.sort(key=lambda people: people["ind"])
for persona in people:
    #print(persona["name"])
    aceptar = vuelo.add_passenger(persona)
    if aceptar:
        print(f"Asiento asignado a {persona}")
    else:
        print(f"Asiento no disponible para {persona}")
