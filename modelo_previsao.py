# modelo_previsao.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Carregando os dados
df = pd.read_csv("vendas.csv")
df['data'] = pd.to_datetime(df['data'])
df['mes'] = df['data'].dt.to_period('M')

# Agrupando por produto e mês
df_mes = df.groupby(['produto', 'mes']).agg({'quantidade': 'sum'}).reset_index()

# Pivotando os dados
dados_ml = df_mes.pivot(index='mes', columns='produto', values='quantidade').fillna(0)

# Separando os dados
X = dados_ml[:-1]
y = dados_ml.shift(-1)[:-1]

# Treinando o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Previsão
previsao = modelo.predict([dados_ml.iloc[-1]])
previsao_df = pd.DataFrame(previsao, columns=dados_ml.columns)

# Ordenando a previsão
previsao_ordenada = previsao_df.T.sort_values(by=0, ascending=False)
print("🔮 Previsão de produtos mais vendidos no próximo mês:")
print(previsao_ordenada)

# Gráfico
previsao_ordenada.head(5).plot(kind='bar', legend=False)
plt.title("Top 5 produtos mais vendidos - Previsão")
plt.ylabel("Quantidade estimada")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
