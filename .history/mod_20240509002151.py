import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Імпорт даних з CSV файлу
df = pd.read_csv('car.csv')

# Знайдемо середню ціну продажу для кожного міста
average_prices = df.groupby('Місто')['Ціна'].mean()

# Виведемо середні ціни для кожного міста
print("Середня ціна продажу для кожного міста:")
print(average_prices)

# Побудуємо графік залежності ціни від пробігу
plt.scatter(df['Пробіг'], df['Ціна'], color='blue')
plt.title('Залежність ціни від пробігу')
plt.xlabel('Пробіг')
plt.ylabel('Ціна')
plt.grid(True)

model = LinearRegression()
model.fit(df[['Пробіг']], df['Ціна'])

# Побудуємо прогнозовану лінію
plt.plot(df['Пробіг'], model.predict(df[['Пробіг']]), color='red')

plt.show()
