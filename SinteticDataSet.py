import pandas as pd
import numpy as np

# Seed
np.random.seed(42)

# Generate Data
n = 600
horas_estudio = np.random.randint(1, 21, size=n)
calificacion = 50 + 2.5 * horas_estudio + np.random.normal(0, 5, size=n)

# Create  Data
data = {
    'Study_hours': horas_estudio,
    'Grades': calificacion
}
df = pd.DataFrame(data)

# Save data
df.to_csv('grades.csv', index=False)
