import pandas as pd

data = pd.read_csv('C:/Users/Doğukan/OneDrive/Masaüstü/Veri.csv')


df = pd.DataFrame(data)

# Boş değerleri diğer araçların kilometrelerinin ortalamasıyla doldurun
ortalama_km = df['km'].mean()
df['km'].fillna(ortalama_km, inplace=True)

data = pd.get_dummies(data, columns=['marka', 'model'])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_normal = scaler.fit_transform(data)

from sklearn.model_selection import train_test_split

X = data.drop('fiyat', axis=1)
y = data['fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(data)