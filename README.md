# 📘 Documentação - Otimização Quântica de Portfolio de Ações

## 🎯 Visão Geral

Sistema de otimização de portfolio de ações utilizando **computação quântica** através do algoritmo QAOA (Quantum Approximate Optimization Algorithm). O projeto integra dados reais de mercado via APIs e aplica técnicas de finanças quantitativas para encontrar a alocação ótima de ativos.

### Características Principais

- ⚛️ **Computação Quântica**: Usa QAOA via Qiskit para otimização
- 📊 **Dados Reais**: Integração com Yahoo Finance, BRAPI e Alpha Vantage
- 📈 **Análise Financeira**: Cálculo de retorno esperado, risco e Sharpe Ratio
- 🎨 **Visualizações**: Gráficos de alocação e análise risco-retorno
- 🔄 **Fallback Inteligente**: Modo simulado quando APIs não disponíveis

---

## 📋 Índice

1. [Instalação](#instalação)
2. [Arquitetura do Sistema](#arquitetura)
3. [Guia de Uso](#guia-de-uso)
4. [APIs Suportadas](#apis-suportadas)
5. [Parâmetros e Configuração](#parâmetros)
6. [Algoritmo QAOA](#algoritmo-qaoa)
7. [Exemplos Práticos](#exemplos)
8. [Referências](#referências)

---

## 🚀 Instalação

### Requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes)

### Instalação das Dependências

```bash
# Instalar todas as dependências
pip install qiskit qiskit-algorithms qiskit-optimization numpy matplotlib pandas requests

# Ou usando requirements.txt
pip install -r requirements.txt
```

### Arquivo requirements.txt

```text
qiskit>=0.45.0
qiskit-algorithms>=0.2.0
qiskit-optimization>=0.6.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
requests>=2.31.0
```

---

## 🏗️ Arquitetura

### Estrutura de Classes

```
┌─────────────────────────────────────────────────────────┐
│                  OTIMIZAÇÃO QUÂNTICA                    │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│ MarketData    │  │  Quantum     │  │  Portfolio   │
│   Fetcher     │  │  Optimizer   │  │  Analyzer    │
└───────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
  APIs Externas      Qiskit QAOA      Visualizações
```

### Componentes Principais

#### 1. **MarketDataFetcher**
Responsável pela busca e integração de dados de mercado.

```python
class MarketDataFetcher:
    """Busca dados reais de mercado usando múltiplas APIs"""
    
    def fetch_yahoo_finance(ticker, period)
    def fetch_brapi(ticker)
    def fetch_alpha_vantage(ticker)
    def get_market_data(tickers, source, period)
```

**Métodos:**
- `fetch_yahoo_finance()`: Busca dados do Yahoo Finance
- `fetch_brapi()`: Busca dados da API brasileira BRAPI
- `fetch_alpha_vantage()`: Busca dados da Alpha Vantage
- `get_market_data()`: Método principal que coordena as buscas

#### 2. **PortfolioData**
Gerencia e processa dados dos ativos.

```python
class PortfolioData:
    """Classe para gerenciar dados de ações"""
    
    def __init__(market_data, use_simulated)
    def get_risk(weights)
    def get_return(weights)
```

**Atributos:**
- `assets`: Lista de tickers dos ativos
- `expected_returns`: Retornos esperados anualizados (%)
- `cov_matrix`: Matriz de covariância (risco)
- `prices`: Preços atuais dos ativos
- `data_source`: Origem dos dados (API ou simulado)

#### 3. **QuantumPortfolioOptimizer**
Núcleo da otimização quântica.

```python
class QuantumPortfolioOptimizer:
    """Otimizador quântico de portfolio usando QAOA"""
    
    def create_qubo_problem()
    def optimize_quantum(reps)
    def interpret_result(result)
    def calculate_weights(selected_indices)
```

**Parâmetros:**
- `data`: Objeto PortfolioData
- `budget`: Orçamento total para investimento
- `risk_aversion`: Coeficiente de aversão ao risco (0-1)

#### 4. **PortfolioAnalyzer**
Análise e visualização de resultados.

```python
class PortfolioAnalyzer:
    """Analisador de resultados do portfolio"""
    
    def print_summary()
    def plot_allocation()
    def plot_risk_return()
```

---

## 📖 Guia de Uso

### Uso Básico

```python
# Importar o módulo principal
from quantum_portfolio_optimizer import main

# Executar com configurações padrão
main()
```

### Configuração Personalizada

```python
# Parâmetros do portfolio
BUDGET = 10000              # R$ 10.000
RISK_AVERSION = 0.5         # Moderado (0-1)
QAOA_REPS = 3               # Camadas do circuito quântico

# Configuração de dados
USE_REAL_DATA = True        # True = API real, False = simulado
DATA_SOURCE = 'auto'        # 'yahoo', 'brapi', 'alpha_vantage', 'auto'
PERIOD = '1y'               # Período de análise histórica

# Ativos para análise
TICKERS = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'WEGE3']

# API Key (opcional - apenas Alpha Vantage)
ALPHA_VANTAGE_KEY = None    # Sua chave aqui
```

### Exemplo Completo

```python
from quantum_portfolio_optimizer import (
    MarketDataFetcher,
    PortfolioData,
    QuantumPortfolioOptimizer,
    PortfolioAnalyzer
)

# 1. Buscar dados de mercado
fetcher = MarketDataFetcher()
market_data = fetcher.get_market_data(
    ['PETR4', 'VALE3', 'ITUB4'],
    source='yahoo',
    period='1y'
)

# 2. Preparar dados
data = PortfolioData(market_data=market_data)

# 3. Otimizar portfolio
optimizer = QuantumPortfolioOptimizer(
    data=data,
    budget=10000,
    risk_aversion=0.5
)
result, qp = optimizer.optimize_quantum(reps=3)

# 4. Interpretar resultado
solution = optimizer.interpret_result(result)

# 5. Analisar e visualizar
if solution:
    analyzer = PortfolioAnalyzer(data, solution)
    analyzer.print_summary()
    analyzer.plot_allocation()
    analyzer.plot_risk_return()
```

---

## 🌐 APIs Suportadas

### 1. Yahoo Finance (Recomendado)

**Características:**
- ✅ Gratuita, sem necessidade de registro
- ✅ Dados globais e brasileiros
- ✅ Histórico extenso
- ✅ Alta confiabilidade

**Uso:**
```python
DATA_SOURCE = 'yahoo'
TICKERS = ['PETR4.SA', 'VALE3.SA']  # Adicionar .SA para B3
```

**Endpoint:**
```
https://query1.finance.yahoo.com/v8/finance/chart/PETR4.SA
```

### 2. BRAPI (Brasil)

**Características:**
- ✅ Gratuita e brasileira
- ✅ Especializada em B3
- ✅ Sem necessidade de chave
- ⚠️ Limite de taxa

**Uso:**
```python
DATA_SOURCE = 'brapi'
TICKERS = ['PETR4', 'VALE3']  # Sem .SA
```

**Endpoint:**
```
https://brapi.dev/api/quote/PETR4?range=1y
```

**Documentação:** [brapi.dev](https://brapi.dev/)

### 3. Alpha Vantage

**Características:**
- 🔑 Requer chave API gratuita
- ✅ Dados detalhados
- ✅ Suporte global
- ⚠️ Limite: 5 requisições/minuto

**Registro:**
[https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

**Uso:**
```python
ALPHA_VANTAGE_KEY = "SUA_CHAVE_AQUI"
DATA_SOURCE = 'alpha_vantage'
```

**Endpoint:**
```
https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=PETR4.SAO
```

### Modo Auto (Recomendado para Iniciantes)

```python
DATA_SOURCE = 'auto'  # Tenta: BRAPI → Yahoo → Alpha Vantage
```

O sistema tentará automaticamente cada API até obter dados válidos.

---

## ⚙️ Parâmetros e Configuração

### Parâmetros do Portfolio

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `BUDGET` | float | 10000 | Orçamento total em R$ |
| `RISK_AVERSION` | float | 0.5 | Aversão ao risco (0-1) |
| `QAOA_REPS` | int | 3 | Camadas do circuito QAOA |

#### RISK_AVERSION

Controla o balanço entre retorno e risco:

- **0.0**: Agressivo - maximiza retorno (ignora risco)
- **0.3**: Moderado-Agressivo
- **0.5**: Balanceado (padrão)
- **0.7**: Conservador
- **1.0**: Muito Conservador - minimiza risco

### Parâmetros de Dados

| Parâmetro | Valores | Descrição |
|-----------|---------|-----------|
| `USE_REAL_DATA` | True/False | Usar dados de API |
| `DATA_SOURCE` | 'yahoo', 'brapi', 'alpha_vantage', 'auto' | Fonte dos dados |
| `PERIOD` | '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y' | Período histórico |
| `TICKERS` | Lista de strings | Ativos para análise |

### Parâmetros do QAOA

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `reps` | int | 3 | Número de camadas p do QAOA |
| `optimizer` | Optimizer | COBYLA | Otimizador clássico |
| `maxiter` | int | 100 | Iterações máximas |

**Impacto do `reps`:**
- **1-2**: Rápido, solução aproximada
- **3-5**: Balanceado (recomendado)
- **6+**: Mais preciso, mas mais lento

---

## ⚛️ Algoritmo QAOA

### Fundamentação Teórica

O QAOA (Quantum Approximate Optimization Algorithm) é um algoritmo híbrido quântico-clássico que resolve problemas de otimização combinatória.

### Formulação QUBO

O problema de otimização de portfolio é formulado como QUBO (Quadratic Unconstrained Binary Optimization):

```
min f(x) = Σᵢ cᵢxᵢ + Σᵢⱼ Qᵢⱼxᵢxⱼ

onde:
- xᵢ ∈ {0,1}: variável binária (incluir ativo i ou não)
- cᵢ: coeficiente linear (retorno esperado)
- Qᵢⱼ: coeficiente quadrático (covariância/risco)
```

### Função Objetivo

```
Objetivo = -Retorno + λ × Risco

onde:
- Retorno = Σᵢ rᵢxᵢ (retorno esperado)
- Risco = Σᵢⱼ σᵢⱼxᵢxⱼ (variância do portfolio)
- λ = RISK_AVERSION (coeficiente de aversão ao risco)
```

### Restrições

1. **Diversificação mínima**: Pelo menos 2 ativos
2. **Diversificação máxima**: No máximo 4 ativos
3. **Budget constraint**: Soma das alocações ≤ orçamento

### Circuito QAOA

```
|ψ(β,γ)⟩ = UP(βp) UC(γp) ... UP(β1) UC(γ1) |+⟩ⁿ

onde:
- UC(γ): Operador de custo (problema)
- UP(β): Operador de mistura
- p: número de camadas (QAOA_REPS)
- n: número de qubits (ativos)
```

### Processo de Otimização

1. **Inicialização**: Estado de superposição uniforme
2. **Parametrização**: Aplicar operadores com parâmetros β e γ
3. **Medição**: Obter distribuição de probabilidades
4. **Otimização Clássica**: Ajustar β e γ para minimizar energia
5. **Iteração**: Repetir até convergência

---

## 💡 Exemplos Práticos

### Exemplo 1: Portfolio Conservador

```python
# Configuração conservadora
RISK_AVERSION = 0.8  # Alto valor = conservador
TICKERS = ['ITUB4', 'BBDC4', 'SANB11', 'BBAS3']  # Bancos
PERIOD = '2y'  # Período mais longo para estabilidade

# Executar
main()
```

**Resultado Esperado:**
- Maior peso em ativos de menor volatilidade
- Sharpe Ratio moderado
- Retorno mais estável

### Exemplo 2: Portfolio Agressivo

```python
# Configuração agressiva
RISK_AVERSION = 0.2  # Baixo valor = agressivo
TICKERS = ['MGLU3', 'AMER3', 'PETZ3', 'VVAR3']  # Varejo
PERIOD = '6mo'  # Período mais recente

# Executar
main()
```

**Resultado Esperado:**
- Maior peso em ativos de alto retorno
- Maior volatilidade
- Potencial de ganho maior

### Exemplo 3: Portfolio Diversificado

```python
# Configuração balanceada
RISK_AVERSION = 0.5
TICKERS = [
    'PETR4',  # Energia
    'VALE3',  # Mineração
    'ITUB4',  # Financeiro
    'WEGE3',  # Industrial
    'ELET3'   # Utilidade Pública
]
PERIOD = '1y'

# Executar
main()
```

**Resultado Esperado:**
- Diversificação setorial
- Risco balanceado
- Correlação reduzida

### Exemplo 4: Comparação de Períodos

```python
import matplotlib.pyplot as plt

periods = ['3mo', '6mo', '1y', '2y']
results = {}

for period in periods:
    PERIOD = period
    # Executar otimização
    result = run_optimization()  # Função auxiliar
    results[period] = result

# Comparar resultados
plot_period_comparison(results)
```

---

## 📊 Métricas e Interpretação

### Retorno Esperado Anual

```
Retorno = Σᵢ wᵢ × rᵢ

onde:
- wᵢ: peso do ativo i no portfolio
- rᵢ: retorno esperado anualizado do ativo i
```

**Interpretação:**
- 5-10%: Conservador
- 10-20%: Moderado
- 20%+: Agressivo

### Volatilidade (Risco)

```
σₚ = √(wᵀ Σ w)

onde:
- w: vetor de pesos
- Σ: matriz de covariância
- σₚ: volatilidade do portfolio
```

**Interpretação:**
- 0-15%: Baixa volatilidade
- 15-25%: Volatilidade moderada
- 25%+: Alta volatilidade

### Índice de Sharpe

```
Sharpe = (Rₚ - Rғ) / σₚ

onde:
- Rₚ: retorno do portfolio
- Rғ: taxa livre de risco (SELIC)
- σₚ: volatilidade do portfolio
```

**Interpretação:**
- < 0: Performance ruim (retorno < risco)
- 0-1: Aceitável
- 1-2: Bom
- 2-3: Muito bom
- 3+: Excelente

---

## 🔧 Troubleshooting

### Problema: APIs não retornam dados

**Solução:**
```python
# Usar modo automático
DATA_SOURCE = 'auto'

# Ou fallback para simulado
USE_REAL_DATA = False
```

### Problema: Erro de importação do Qiskit

**Solução:**
```bash
pip install --upgrade qiskit qiskit-algorithms qiskit-optimization
```

### Problema: Nenhum ativo selecionado

**Causas Possíveis:**
1. RISK_AVERSION muito alto
2. Poucos ativos disponíveis
3. Restrições muito rígidas

**Solução:**
```python
# Ajustar aversão ao risco
RISK_AVERSION = 0.3  # Reduzir

# Aumentar número de ativos
TICKERS = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'WEGE3', 'RENT3']

# Reduzir restrições no código
qp.linear_constraint(rhs=1, name='min_assets')  # Mínimo 1 ativo
```

### Problema: Otimização muito lenta

**Solução:**
```python
# Reduzir camadas QAOA
QAOA_REPS = 2

# Reduzir iterações
optimizer = COBYLA(maxiter=50)

# Usar menos ativos
TICKERS = ['PETR4', 'VALE3', 'ITUB4']  # Apenas 3
```

### Problema: Resultados inconsistentes

**Causas:**
- Dados insuficientes
- Período muito curto
- Alta volatilidade do mercado

**Solução:**
```python
# Usar período mais longo
PERIOD = '2y'

# Aumentar número de repetições QAOA
QAOA_REPS = 5

# Filtrar ativos com dados completos
# Verificar market_data antes de usar
```

---

## 📈 Interpretação dos Gráficos

### Gráfico 1: Distribuição do Portfolio

**Descrição:** Gráfico de pizza mostrando peso percentual de cada ativo.

**Interpretação:**
- Distribuição uniforme (20-30% cada): Bem diversificado
- Um ativo dominante (>50%): Concentrado
- Múltiplos ativos pequenos (<10%): Pulverizado

### Gráfico 2: Valor Investido por Ativo

**Descrição:** Gráfico de barras com valor em R$ alocado.

**Uso Prático:**
- Determinar quantas ações comprar
- Verificar valores mínimos de investimento
- Planejar execução de ordens

### Gráfico 3: Análise Risco x Retorno

**Descrição:** Scatter plot com portfolios aleatórios vs. otimizado.

**Interpretação:**
- Portfolio otimizado (estrela vermelha) deve estar na fronteira superior esquerda
- Posição ideal: Alto retorno, baixo risco
- Distância dos pontos azuis indica qualidade da otimização

---

## 🎓 Conceitos de Finanças

### Teoria Moderna de Portfolios (Markowitz)

O projeto implementa os princípios de Harry Markowitz:

1. **Diversificação reduz risco**: Combinar ativos não perfeitamente correlacionados
2. **Fronteira eficiente**: Melhor retorno para cada nível de risco
3. **Trade-off risco-retorno**: Não há retorno sem risco

### Matriz de Covariância

Mede como os ativos se movem juntos:

```
σᵢⱼ = Cov(Rᵢ, Rⱼ)

- σᵢⱼ > 0: Movem-se juntos (correlação positiva)
- σᵢⱼ < 0: Movem-se opostamente (correlação negativa)
- σᵢⱼ = 0: Sem correlação
```

### Cálculo de Retornos

```python
# Retorno logarítmico
returns = np.log(prices / prices.shift(1))

# Retorno anualizado
annual_return = returns.mean() * 252  # 252 dias úteis
```

---

## 🔬 Vantagens da Abordagem Quântica

### Por que usar QAOA?

1. **Exploração Global**: Evita mínimos locais
2. **Superposição Quântica**: Avalia múltiplas soluções simultaneamente
3. **Escalabilidade**: Potencialmente mais rápido para problemas grandes
4. **Inovação**: Preparação para computadores quânticos reais

### Limitações Atuais

- Simuladores clássicos limitam tamanho do problema
- Hardware quântico ainda em desenvolvimento
- Ruído quântico em dispositivos reais
- Custo computacional para muitos ativos

### Quando usar Computação Quântica?

**Recomendado:**
- Portfolios com 10-50 ativos
- Problemas com múltiplas restrições
- Quando soluções clássicas ficam presas em mínimos locais
- Pesquisa e desenvolvimento

**Não Recomendado:**
- Portfolios muito pequenos (< 5 ativos)
- Quando solução clássica é suficiente
- Aplicações de produção críticas (ainda)

---

## 📚 Referências

### Artigos Científicos

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm". arXiv:1411.4028

2. Markowitz, H. (1952). "Portfolio Selection". The Journal of Finance, 7(1), 77-91.

3. Phillipson, F., & Chiscop, I. (2021). "Multimodal Container Planning: A QAOA Approach". Applied Sciences, 11(13), 6578.

### Documentação Técnica

- **Qiskit**: [https://qiskit.org/documentation/](https://qiskit.org/documentation/)
- **Qiskit Finance**: [https://qiskit.org/ecosystem/finance/](https://qiskit.org/ecosystem/finance/)
- **QAOA Tutorial**: [https://qiskit.org/textbook/ch-applications/qaoa.html](https://qiskit.org/textbook/ch-applications/qaoa.html)

### APIs

- **Yahoo Finance**: [https://finance.yahoo.com/](https://finance.yahoo.com/)
- **BRAPI**: [https://brapi.dev/docs](https://brapi.dev/docs)
- **Alpha Vantage**: [https://www.alphavantage.co/documentation/](https://www.alphavantage.co/documentation/)

### Livros Recomendados

1. "Modern Portfolio Theory and Investment Analysis" - Elton et al.
2. "Quantum Computing for Computer Scientists" - Yanofsky & Mannucci
3. "Quantitative Finance with Python" - Yves Hilpisch

---

## 🤝 Contribuindo

### Como Contribuir

1. Fork do repositório
2. Criar branch para feature (`git checkout -b feature/NovaAPI`)
3. Commit das mudanças (`git commit -am 'Adiciona nova API'`)
4. Push para branch (`git push origin feature/NovaAPI`)
5. Criar Pull Request

### Áreas para Contribuição

- 🌐 Novas integrações de APIs
- 📊 Métricas financeiras adicionais
- ⚛️ Otimizações do algoritmo quântico
- 📈 Visualizações avançadas
- 🧪 Testes unitários
- 📖 Melhorias na documentação

---

## 📄 Licença

Este projeto é disponibilizado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

## ⚠️ Aviso Legal

**IMPORTANTE:** Este software é fornecido apenas para fins educacionais e de pesquisa.

- ❌ NÃO constitui aconselhamento financeiro
- ❌ NÃO deve ser usado como única ferramenta para decisões de investimento
- ❌ NÃO garante retornos ou performance
- ✅ Sempre consulte um profissional certificado antes de investir
- ✅ Investimentos envolvem riscos de perda de capital
- ✅ Performance passada não garante resultados futuros

Os desenvolvedores não se responsabilizam por perdas financeiras decorrentes do uso deste software.

---

## 📞 Suporte e Contato

### Reportar Bugs

Abra uma issue no GitHub com:
- Descrição do problema
- Passos para reproduzir
- Versões do Python e bibliotecas
- Logs de erro

### Perguntas Frequentes

Visite a seção **Issues** no GitHub para FAQs comuns.

### Comunidade

- GitHub Discussions: Para discussões gerais
- Stack Overflow: Tag `qiskit` e `quantum-computing`

---

## 🎯 Roadmap

### Versão Futura

- [ ] Backtesting histórico
- [ ] Análise de Monte Carlo
- [ ] Rebalanceamento automático
- [ ] Integração com corretoras
- [ ] Dashboard web interativo
- [ ] Suporte a criptomoedas
- [ ] Otimização multi-objetivo
- [ ] Análise de sentimento de mercado
- [ ] Hardware quântico real (IBM Quantum)

---

## 🏆 Créditos

**Desenvolvido usando:**
- Qiskit (IBM Quantum)
- NumPy & Pandas
- Matplotlib
- Teoria de Markowitz
- Algoritmo QAOA

**Inspirado em:**
- Pesquisas em quantum finance
- Técnicas modernas de gestão de portfolios
- Comunidade open-source

---

**Versão da Documentação:** 1.0.0  
**Última Atualização:** Novembro 2024  
**Autor:** Projeto Open Source

---

*"O futuro das finanças é quântico."* ⚛️💰