# Previsão de Consumo de Energia com Perceptron Multicamadas (MLP)

## 📋 Descrição do Projeto

Este projeto implementa uma rede neural artificial (Perceptron Multicamadas - MLP) para prever o consumo de energia elétrica em edifícios comerciais com base em variáveis ambientais e operacionais.

### Variáveis

**Entradas:**
- **x1**: Temperatura ambiente (°C)
- **x2**: Umidade relativa (%)
- **x3**: Nível de ocupação do prédio (pessoas por 100 m²)

**Saída:**
- **y**: Consumo de energia (kWh)

## 🏗️ Arquitetura da Rede Neural

```
Camada de Entrada:  3 neurônios (x1, x2, x3)
         ↓
Camada Oculta:     10 neurônios (ReLU)
         ↓
Camada de Saída:    1 neurônio (y)
```

### Configurações
- **Função de Ativação (oculta):** ReLU
- **Otimizador:** Adam
- **Função de Perda:** MSE (Mean Squared Error)
- **Early Stopping:** Patience de 20 épocas
- **Épocas Máximas:** 300

## 📁 Estrutura do Projeto

```
redes_neurais/
│
├── consumo_energia_train.csv      # Dados de treinamento (8001 amostras)
├── consumo_energia_test.csv       # Dados de teste (2001 amostras)
├── consumo_energia_full.csv       # Dataset completo (10001 amostras)
├── mlp_consumo_energia.ipynb      # Notebook principal com toda a implementação
├── README.md                      # Este arquivo
│
└── (após executar o notebook)
    ├── modelo_mlp_consumo_energia.keras  # Modelo treinado
    ├── scaler_X.pkl                      # Escalador das features
    └── scaler_y.pkl                      # Escalador do target
```

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0 seaborn>=0.11.0 scikit-learn>=1.0.0 tensorflow>=2.10.0 jupyter>=1.0.0 joblib>=1.0.0
```

### 2. Executar o Notebook

```bash
jupyter notebook mlp_consumo_energia.ipynb
```

Ou abra o arquivo diretamente no VS Code.

### 3. Executar as Células

Execute todas as células sequencialmente (ou use "Run All") para:
1. Carregar e explorar os dados
2. Pré-processar os dados
3. Construir e treinar o modelo
4. Avaliar o desempenho
5. Gerar visualizações e relatórios

## 📊 Etapas do Projeto

### 1. Exploração de Dados
- Estatísticas descritivas
- Histogramas das variáveis
- Gráficos de dispersão
- Matriz de correlação

### 2. Pré-processamento
- Normalização/Padronização (StandardScaler)
- Divisão treino/validação (90/10)
- Conjunto de teste separado

### 3. Construção da Rede Neural
- Arquitetura MLP: 3-10-1
- Configuração de hiperparâmetros
- Implementação do early stopping

### 4. Treinamento e Validação
- Treinamento por até 300 épocas
- Monitoramento da perda (train/validation)
- Curvas de aprendizado

### 5. Avaliação
- Métricas: MSE, RMSE, MAE, R²
- Gráficos: Real vs Previsto
- Análise de resíduos

### 6. Apresentação
- Relatório com principais resultados
- Interpretações e conclusões
- Recomendações

## 📈 Métricas de Avaliação

O modelo é avaliado usando as seguintes métricas:

- **MSE (Mean Squared Error)**: Erro quadrático médio
- **RMSE (Root Mean Squared Error)**: Raiz do erro quadrático médio
- **MAE (Mean Absolute Error)**: Erro absoluto médio
- **R² (Coeficiente de Determinação)**: Proporção da variância explicada

## 🎯 Resultados Esperados

O notebook gera automaticamente:

✓ Visualizações completas da análise exploratória  
✓ Curvas de perda durante o treinamento  
✓ Comparação de métricas entre treino/validação/teste  
✓ Gráficos de valores reais vs previstos  
✓ Análise de distribuição dos resíduos  
✓ Relatório final com interpretações  

## 💡 Uso do Modelo Treinado

Após o treinamento, o modelo pode ser usado para fazer previsões:

```python
import numpy as np
from tensorflow import keras
import joblib

# Carregar modelo e escaladores
model = keras.models.load_model('modelo_mlp_consumo_energia.keras')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Fazer previsão
def prever_consumo(temperatura, umidade, ocupacao):
    X_novo = np.array([[temperatura, umidade, ocupacao]])
    X_novo_scaled = scaler_X.transform(X_novo)
    y_pred_scaled = model.predict(X_novo_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred[0, 0]

# Exemplo
consumo = prever_consumo(25, 50, 30)
print(f"Consumo previsto: {consumo:.2f} kWh")
```

## 📚 Tecnologias Utilizadas

- **Python 3.x**
- **NumPy**: Computação numérica
- **Pandas**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualizações
- **Scikit-learn**: Pré-processamento e métricas
- **TensorFlow/Keras**: Implementação da rede neural
- **Jupyter Notebook**: Ambiente de desenvolvimento

## 👨‍💻 Autor

Carlos Meliga //

- Trabalho desenvolvido para a disciplina de Inteligência Artificial Computacional  

## 📝 Licença

Este projeto é para fins educacionais.
