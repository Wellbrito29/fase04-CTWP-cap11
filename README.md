# Implementando Algoritmos de Machine Learning com Scikit-learn  
**Capítulo 3**

---

### Desenvolvedores  
- **CELESTE LEITE DOS SANTOS** - RM 559312  
- **LUMA SANTOS DE OLIVEIRA** - RM 560146  
- **WELLIGTON NASCIMENTO** - RM 552157  
- **RICARDO ARAÚJO DE OLIVEIRA** - RM 561182  

---

### Link da Atividade  
[Google Colab](https://colab.research.google.com/drive/1yzjNbtNg_SQEM0M4uZsBB7z3FRw_NVi6?usp=sharing#scrollTo=xe9ree5pUiPb)  

---

## Análise das Acurácias e Relatórios de Classificação para Cada Modelo  

### 1. **Regressão Logística**  
**Acurácia**: 0.90  

**Relatório de Classificação**:  

| Classe | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|---------|  
| 0      | 0.88      | 0.93   | 0.90     | 30      |  
| 1      | 0.92      | 0.88   | 0.90     | 32      |  
| 2      | 0.90      | 0.88   | 0.89     | 34      |  

**Métricas gerais**:  

| Métrica        | Valor |  
|----------------|-------|  
| Accuracy       | 0.90  |  
| Macro Avg      | 0.90  |  
| Weighted Avg   | 0.90  |  

---

### 2. **KNN (K-Nearest Neighbors)**  
**Acurácia**: 0.88  

**Relatório de Classificação**:  

| Classe | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|---------|  
| 0      | 0.85      | 0.90   | 0.88     | 30      |  
| 1      | 0.90      | 0.88   | 0.89     | 32      |  
| 2      | 0.88      | 0.85   | 0.86     | 34      |  

**Métricas gerais**:  

| Métrica        | Valor |  
|----------------|-------|  
| Accuracy       | 0.88  |  
| Macro Avg      | 0.88  |  
| Weighted Avg   | 0.88  |  

---

### 3. **SVM (Kernel RBF)**  
**Acurácia**: 0.92  

**Relatório de Classificação**:  

| Classe | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|---------|  
| 0      | 0.90      | 0.93   | 0.92     | 30      |  
| 1      | 0.94      | 0.91   | 0.92     | 32      |  
| 2      | 0.91      | 0.91   | 0.91     | 34      |  

**Métricas gerais**:  

| Métrica        | Valor |  
|----------------|-------|  
| Accuracy       | 0.92  |  
| Macro Avg      | 0.92  |  
| Weighted Avg   | 0.92  |  

---

### 4. **SVM (Kernel Polinomial)**  
**Acurácia**: 0.89  

**Relatório de Classificação**:  

| Classe | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|---------|  
| 0      | 0.87      | 0.90   | 0.88     | 30      |  
| 1      | 0.91      | 0.88   | 0.89     | 32      |  
| 2      | 0.88      | 0.88   | 0.88     | 34      |  

**Métricas gerais**:  

| Métrica        | Valor |  
|----------------|-------|  
| Accuracy       | 0.89  |  
| Macro Avg      | 0.89  |  
| Weighted Avg   | 0.89  |  

---

### 5. **SVM (Kernel Linear)**  
**Acurácia**: 0.87  

**Relatório de Classificação**:  

| Classe | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|---------|  
| 0      | 0.85      | 0.87   | 0.86     | 30      |  
| 1      | 0.88      | 0.88   | 0.88     | 32      |  
| 2      | 0.88      | 0.85   | 0.86     | 34      |  

**Métricas gerais**:  

| Métrica        | Valor |  
|----------------|-------|  
| Accuracy       | 0.87  |  
| Macro Avg      | 0.87  |  
| Weighted Avg   | 0.87  |  

---

### 6. **Decision Tree**  
**Acurácia**: 0.85  

**Relatório de Classificação**:  

| Classe | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|---------|  
| 0      | 0.83      | 0.87   | 0.85     | 30      |  
| 1      | 0.88      | 0.84   | 0.86     | 32      |  
| 2      | 0.85      | 0.85   | 0.85     | 34      |  

**Métricas gerais**:  

| Métrica        | Valor |  
|----------------|-------|  
| Accuracy       | 0.85  |  
| Macro Avg      | 0.85  |  
| Weighted Avg   | 0.85  |  

---

### 7. **Random Forest**  
**Acurácia**: 0.91  

**Relatório de Classificação**:  

| Classe | Precision | Recall | F1-Score | Support |  
|--------|-----------|--------|----------|---------|  
| 0      | 0.90      | 0.90   | 0.90     | 30      |  
| 1      | 0.91      | 0.91   | 0.91     | 32      |  
| 2      | 0.91      | 0.91   | 0.91     | 34      |  

**Métricas gerais**:  

| Métrica        | Valor |  
|----------------|-------|  
| Accuracy       | 0.91  |  
| Macro Avg      | 0.91  |  
| Weighted Avg   | 0.91  |  

---

## Insights Relevantes  

- **SVM com Kernel RBF** apresentou a maior acurácia (0.92), destacando-se como o modelo mais eficaz.  
- **Random Forest** obteve excelente desempenho com acurácia de 0.91, mostrando-se robusto.  
- **Regressão Logística** e **KNN** também demonstraram boa performance com acurácias de 0.90 e 0.88, respectivamente.  
- **Decision Tree** teve a menor acurácia (0.85), sugerindo que pode não ser ideal para este problema.  

---

### Limpeza de Dados  

**Procedimento**:  
- Utilizou-se o método **IQR (Interquartile Range)** para remover outliers.  
- Após remoção:  

| Shape Original | Shape Após Remoção |  
|----------------|---------------------|  
| (210, 8)       | (182, 8)           |  

**Impacto nos Modelos**:  
Após remoção, houve leve melhora na acurácia da maioria dos modelos, com destaque para o **SVM (Kernel RBF)**, que atingiu **0.93**.  

---

### Conclusão  

Modelos como **SVM (Kernel RBF)** e **Random Forest** foram os mais adequados para o problema proposto. A limpeza e normalização dos dados provaram-se essenciais para maximizar a eficácia dos modelos.  

---
