#Funcion para sacar la raiz cuadrada
def sqrt(number, tolerance=1e-10):
    if number < 0:
        raise ValueError("El numero no puede ser negativo")
    
    guess = number
    while True:
        new_guess = 0.5 * (guess + number / guess)
        if abs(new_guess - guess) < tolerance:
            return new_guess
        guess = new_guess

# Funcion para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return sqrt(distance)

# Funcion principal para clasificar un punto de prueba
def knn_classifier(training_data, test_point, k):
    distances = []
    
    # Calcular la distancia entre el punto de prueba y cada punto de entrenamiento
    for data_point, label in training_data:
        distance = euclidean_distance(data_point, test_point)
        distances.append((data_point, label, distance))
    
    # Ordenar las distancias de menor a mayor
    distances.sort(key=lambda x: x[2])
    
    # Obtener los k vecinos mas cercanos
    k_nearest_neighbors = distances[:k]
    
    # Contar las apariciones de cada clase entre los k vecinos mas cercanos
    class_counts = {}
    for neighbor in k_nearest_neighbors:
        label = neighbor[1]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Obtener la clase con mas apariciones entre los k vecinos mas cercanos
    predicted_class = max(class_counts, key=class_counts.get)
    
    return predicted_class

# Nuevos datos de entrenamiento
new_training_data = [
    ([158, 58], 'M'),
    ([158, 59], 'M'),
    ([158, 63], 'M'),
    ([160, 59], 'M'),
    ([163, 60], 'M'),
    ([163, 61], 'M'),
    ([160, 64], 'L'),
    ([163, 64], 'L'),
    ([165, 61], 'L'),
    ([165, 62], 'L'),
    ([165, 65], 'L'),
    ([168, 62], 'L'),
    ([168, 63], 'L'),
    ([168, 66], 'L'),
    ([170, 63], 'L'),
    ([170, 64], 'L'),
    ([170, 68], 'L')
]

# Punto de prueba
test_point = [161, 61]

# Valor de k (numero de vecinos mas cercanos a considerar)
k = 5

# Clasificar el punto de prueba utilizando KNN
predicted_class = knn_classifier(new_training_data, test_point, k)

# Imprimir la clase predicha para el punto de prueba
print("La clase predicha para el punto de prueba", test_point, "es:", predicted_class)