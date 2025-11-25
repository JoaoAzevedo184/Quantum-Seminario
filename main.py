"""
OTIMIZAÇÃO QUÂNTICA DE PORTFOLIO DE AÇÕES
Usa QAOA (Quantum Approximate Optimization Algorithm) para otimizar alocação de ativos
Integrado com APIs reais: Yahoo Finance e Alpha Vantage
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
import pandas as pd
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. INTEGRAÇÃO COM APIs DE DADOS REAIS
# ============================================================================

class MarketDataFetcher:
    """Busca dados reais de mercado usando múltiplas APIs"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.cache = {}
        
    def fetch_yahoo_finance(self, ticker, period='1y'):
        """
        Busca dados do Yahoo Finance (API gratuita)
        período: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
        try:
            # Yahoo Finance API v8 (gratuita)
            base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
            
            # Ajustar ticker brasileiro (ex: PETR4.SA)
            if not ticker.endswith('.SA') and not ticker.startswith('^'):
                ticker = f"{ticker}.SA"
            
            params = {
                'interval': '1d',
                'range': period
            }
            
            response = requests.get(f"{base_url}{ticker}", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data['chart']['result'][0]
                
                # Extrair preços
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                closes = quotes['close']
                
                # Criar DataFrame
                df = pd.DataFrame({
                    'date': pd.to_datetime(timestamps, unit='s'),
                    'close': closes
                })
                
                df = df.dropna()
                
                print(f"   ✓ {ticker}: {len(df)} dias de dados")
                return df
                
            else:
                print(f"   ✗ Erro ao buscar {ticker}: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ✗ Erro {ticker}: {str(e)}")
            return None
    
    def fetch_alpha_vantage(self, ticker):
        """
        Busca dados da Alpha Vantage (requer API key gratuita)
        Registre-se em: https://www.alphavantage.co/support/#api-key
        """
        if not self.api_key:
            print("   ⚠️ Alpha Vantage requer API key (gratuita)")
            return None
        
        try:
            # Ajustar ticker brasileiro
            symbol = ticker.replace('.SA', '.SAO')
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df['close'] = df['4. close'].astype(float)
                df = df[['close']]
                df = df.reset_index()
                df.columns = ['date', 'close']
                
                print(f"   ✓ {ticker}: {len(df)} dias de dados")
                return df
            else:
                print(f"   ✗ Alpha Vantage: {data.get('Note', 'Erro desconhecido')}")
                return None
                
        except Exception as e:
            print(f"   ✗ Erro Alpha Vantage {ticker}: {str(e)}")
            return None
    
    def fetch_brapi(self, ticker):
        """
        Busca dados da BRAPI (API brasileira gratuita)
        Documentação: https://brapi.dev/
        """
        try:
            # BRAPI usa tickers sem .SA
            clean_ticker = ticker.replace('.SA', '')
            
            url = f"https://brapi.dev/api/quote/{clean_ticker}"
            params = {
                'range': '1y',
                'interval': '1d',
                'fundamental': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    
                    if 'historicalDataPrice' in result:
                        prices = result['historicalDataPrice']
                        
                        df = pd.DataFrame(prices)
                        df['date'] = pd.to_datetime(df['date'], unit='s')
                        df = df[['date', 'close']]
                        df = df.sort_values('date')
                        
                        print(f"   ✓ {clean_ticker}: {len(df)} dias de dados")
                        return df
                
                print(f"   ✗ BRAPI: Sem dados para {clean_ticker}")
                return None
            else:
                print(f"   ✗ BRAPI erro: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ✗ Erro BRAPI {ticker}: {str(e)}")
            return None
    
    def get_market_data(self, tickers, source='yahoo', period='1y'):
        """
        Busca dados de múltiplos ativos
        source: 'yahoo', 'alpha_vantage', 'brapi', ou 'auto' (tenta todos)
        """
        print(f"\n🌐 Buscando dados de mercado ({source.upper()})...")
        print(f"   Período: {period}")
        print(f"   Ativos: {', '.join(tickers)}\n")
        
        data_dict = {}
        
        for ticker in tickers:
            df = None
            
            if source == 'auto':
                # Tentar múltiplas fontes
                df = (self.fetch_brapi(ticker) or 
                      self.fetch_yahoo_finance(ticker, period) or
                      self.fetch_alpha_vantage(ticker))
            elif source == 'yahoo':
                df = self.fetch_yahoo_finance(ticker, period)
            elif source == 'brapi':
                df = self.fetch_brapi(ticker)
            elif source == 'alpha_vantage':
                df = self.fetch_alpha_vantage(ticker)
            
            if df is not None and len(df) > 0:
                data_dict[ticker] = df
        
        if len(data_dict) == 0:
            print("\n❌ Nenhum dado foi obtido!")
            return None
        
        print(f"\n✅ Dados obtidos para {len(data_dict)} ativos")
        return data_dict

# ============================================================================
# 2. ANÁLISE DE DADOS REAIS
# ============================================================================

class PortfolioData:
    """Classe para gerenciar dados de ações com integração de APIs"""
    
    def __init__(self, market_data=None, use_simulated=False):
        """
        market_data: dicionário com dados reais {ticker: DataFrame}
        use_simulated: usar dados simulados se True
        """
        if use_simulated or market_data is None:
            self._load_simulated_data()
        else:
            self._load_real_data(market_data)
    
    def _load_simulated_data(self):
        """Carrega dados simulados (fallback)"""
        print("\n📊 Usando dados simulados...")
        
        self.assets = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'MGLU3']
        self.expected_returns = np.array([15.2, 18.5, 12.3, 11.8, 8.5])
        self.cov_matrix = np.array([
            [0.0625, 0.0312, 0.0156, 0.0125, 0.0094],
            [0.0312, 0.0900, 0.0200, 0.0180, 0.0120],
            [0.0156, 0.0200, 0.0400, 0.0300, 0.0100],
            [0.0125, 0.0180, 0.0300, 0.0361, 0.0090],
            [0.0094, 0.0120, 0.0100, 0.0090, 0.0484]
        ])
        self.prices = np.array([28.50, 65.30, 24.80, 14.20, 3.15])
        self.data_source = "simulado"
    
    def _load_real_data(self, market_data):
        """Processa dados reais do mercado"""
        print("\n📈 Processando dados reais do mercado...")
        
        self.assets = list(market_data.keys())
        self.market_data = market_data
        self.data_source = "API real"
        
        # Calcular retornos diários
        returns_dict = {}
        
        for ticker, df in market_data.items():
            df = df.sort_values('date')
            df['returns'] = df['close'].pct_change()
            returns_dict[ticker] = df['returns'].dropna()
            
        # Alinhar datas (intersecção)
        all_returns = pd.DataFrame(returns_dict)
        all_returns = all_returns.dropna()
        
        # Estatísticas
        print(f"   • Período analisado: {len(all_returns)} dias")
        print(f"   • Data início: {all_returns.index[0].strftime('%d/%m/%Y')}")
        print(f"   • Data fim: {all_returns.index[-1].strftime('%d/%m/%Y')}")
        
        # Retornos esperados anualizados (%)
        self.expected_returns = (all_returns.mean() * 252 * 100).values
        
        # Matriz de covariância anualizada
        self.cov_matrix = all_returns.cov().values * 252
        
        # Preço atual (último preço)
        self.prices = np.array([
            market_data[ticker]['close'].iloc[-1] 
            for ticker in self.assets
        ])
        
        # Mostrar estatísticas
        print("\n   📊 Estatísticas dos Ativos:")
        stats_df = pd.DataFrame({
            'Ativo': self.assets,
            'Retorno Anual (%)': self.expected_returns,
            'Volatilidade (%)': np.sqrt(np.diag(self.cov_matrix)) * 100,
            'Preço Atual (R$)': self.prices
        })
        print(stats_df.to_string(index=False))
        
    def get_risk(self, weights):
        """Calcula risco do portfolio (volatilidade)"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def get_return(self, weights):
        """Calcula retorno esperado do portfolio"""
        return np.dot(weights, self.expected_returns)

# ============================================================================
# 3. FORMULAÇÃO DO PROBLEMA QUÂNTICO
# ============================================================================

class QuantumPortfolioOptimizer:
    """Otimizador quântico de portfolio usando QAOA"""
    
    def __init__(self, data, budget=10000, risk_aversion=0.5):
        self.data = data
        self.budget = budget
        self.risk_aversion = risk_aversion
        self.n_assets = len(data.assets)
        
    def create_qubo_problem(self):
        """
        Cria o problema QUBO (Quadratic Unconstrained Binary Optimization)
        Objetivo: Maximizar retorno - risk_aversion * risco
        """
        qp = QuadraticProgram('portfolio_optimization')
        
        # Variáveis binárias: x[i] = 1 se incluir ação i no portfolio
        for i in range(self.n_assets):
            qp.binary_var(name=f'x_{i}')
        
        # Coeficientes lineares (retornos esperados)
        linear = {}
        for i in range(self.n_assets):
            linear[f'x_{i}'] = -self.data.expected_returns[i]  # Negativo para maximizar
        
        # Coeficientes quadráticos (penalização por risco)
        quadratic = {}
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i <= j:
                    key = (f'x_{i}', f'x_{j}')
                    quadratic[key] = self.risk_aversion * self.data.cov_matrix[i, j]
        
        # Função objetivo
        qp.minimize(linear=linear, quadratic=quadratic)
        
        # Restrição: selecionar pelo menos 2 e no máximo 4 ativos
        constraint_linear = {f'x_{i}': 1 for i in range(self.n_assets)}
        qp.linear_constraint(
            linear=constraint_linear,
            sense='>=',
            rhs=2,
            name='min_assets'
        )
        qp.linear_constraint(
            linear=constraint_linear,
            sense='<=',
            rhs=4,
            name='max_assets'
        )
        
        return qp
    
    def optimize_quantum(self, reps=3):
        """
        Executa otimização quântica usando QAOA
        reps: número de camadas do circuito QAOA
        """
        print("🔬 Iniciando otimização quântica com QAOA...")
        
        # Criar problema QUBO
        qp = self.create_qubo_problem()
        print(f"\n📊 Problema criado: {self.n_assets} ativos")
        
        # Converter para QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Configurar QAOA
        optimizer = COBYLA(maxiter=100)
        sampler = Sampler()
        
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps
        )
        
        # Resolver com QAOA
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        result = qaoa_optimizer.solve(qubo)
        
        return result, qp
    
    def interpret_result(self, result):
        """Interpreta resultado quântico e calcula alocação"""
        # Extrair seleção de ativos
        selected = []
        for i in range(self.n_assets):
            if result.x[i] > 0.5:  # Variável binária
                selected.append(i)
        
        if len(selected) == 0:
            print("⚠️ Nenhum ativo selecionado")
            return None
        
        # Calcular pesos ótimos usando Markowitz nos ativos selecionados
        weights = self.calculate_weights(selected)
        
        return {
            'selected_indices': selected,
            'selected_assets': [self.data.assets[i] for i in selected],
            'weights': weights,
            'allocation': weights * self.budget
        }
    
    def calculate_weights(self, selected_indices):
        """Calcula pesos ótimos para ativos selecionados (método clássico)"""
        n = len(selected_indices)
        
        # Subconjunto de retornos e covariância
        returns_subset = self.data.expected_returns[selected_indices]
        cov_subset = self.data.cov_matrix[np.ix_(selected_indices, selected_indices)]
        
        # Otimização de Markowitz: minimizar variância para retorno alvo
        inv_cov = np.linalg.inv(cov_subset)
        ones = np.ones(n)
        
        # Pesos de variância mínima
        weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
        
        return weights

# ============================================================================
# 4. ANÁLISE E VISUALIZAÇÃO
# ============================================================================

class PortfolioAnalyzer:
    """Analisador de resultados do portfolio"""
    
    def __init__(self, data, solution):
        self.data = data
        self.solution = solution
    
    def print_summary(self):
        """Imprime resumo do portfolio otimizado"""
        print("\n" + "="*70)
        print("📈 PORTFOLIO OTIMIZADO - RESULTADO QUÂNTICO")
        print("="*70)
        print(f"🔗 Fonte de Dados: {self.data.data_source}")
        
        selected = self.solution['selected_assets']
        weights = self.solution['weights']
        allocation = self.solution['allocation']
        
        print(f"\n🎯 Ativos Selecionados: {len(selected)}")
        print("-" * 70)
        
        df = pd.DataFrame({
            'Ativo': selected,
            'Peso (%)': weights * 100,
            'Alocação (R$)': allocation,
            'Retorno Esperado (%)': self.data.expected_returns[self.solution['selected_indices']],
            'Preço Atual (R$)': self.data.prices[self.solution['selected_indices']]
        })
        
        print(df.to_string(index=False))
        
        # Métricas do portfolio
        portfolio_return = np.dot(weights, 
                                 self.data.expected_returns[self.solution['selected_indices']])
        
        portfolio_risk = np.sqrt(np.dot(weights.T, 
                                       np.dot(self.data.cov_matrix[np.ix_(
                                           self.solution['selected_indices'],
                                           self.solution['selected_indices'])], 
                                       weights)))
        
        sharpe_ratio = portfolio_return / (portfolio_risk * 100)
        
        print("\n" + "-" * 70)
        print(f"💰 Retorno Esperado Anual: {portfolio_return:.2f}%")
        print(f"📊 Volatilidade (Risco): {portfolio_risk*100:.2f}%")
        print(f"⚡ Índice de Sharpe: {sharpe_ratio:.2f}")
        print("="*70)
    
    def plot_allocation(self):
        """Visualiza alocação do portfolio"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico de pizza - Alocação
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.solution['selected_assets'])))
        
        axes[0].pie(self.solution['weights'], 
                   labels=self.solution['selected_assets'],
                   autopct='%1.1f%%',
                   colors=colors,
                   startangle=90)
        axes[0].set_title('Distribuição do Portfolio (%)', fontsize=14, fontweight='bold')
        
        # Gráfico de barras - Valores em R$
        axes[1].bar(self.solution['selected_assets'],
                   self.solution['allocation'],
                   color=colors,
                   edgecolor='black',
                   linewidth=1.5)
        axes[1].set_ylabel('Alocação (R$)', fontsize=12)
        axes[1].set_title('Valor Investido por Ativo', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(self.solution['allocation']):
            axes[1].text(i, v + 100, f'R$ {v:.0f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('portfolio_quantum_allocation.png', dpi=300, bbox_inches='tight')
        print("\n💾 Gráfico salvo: portfolio_quantum_allocation.png")
        plt.show()
    
    def plot_risk_return(self):
        """Plota fronteira eficiente e portfolio otimizado"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simular múltiplos portfolios aleatórios
        n_portfolios = 1000
        returns = []
        risks = []
        
        for _ in range(n_portfolios):
            weights = np.random.random(len(self.data.assets))
            weights /= weights.sum()
            
            ret = np.dot(weights, self.data.expected_returns)
            risk = np.sqrt(np.dot(weights.T, np.dot(self.data.cov_matrix, weights)))
            
            returns.append(ret)
            risks.append(risk)
        
        # Plotar portfolios aleatórios
        ax.scatter(risks, returns, c='lightblue', alpha=0.5, s=20, label='Portfolios Aleatórios')
        
        # Portfolio quântico otimizado
        selected = self.solution['selected_indices']
        weights = self.solution['weights']
        
        opt_return = np.dot(weights, self.data.expected_returns[selected])
        opt_risk = np.sqrt(np.dot(weights.T, 
                                 np.dot(self.data.cov_matrix[np.ix_(selected, selected)], 
                                       weights)))
        
        ax.scatter(opt_risk, opt_return, c='red', s=200, marker='*', 
                  label='Portfolio Quântico Otimizado', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Risco (Volatilidade)', fontsize=12)
        ax.set_ylabel('Retorno Esperado (%)', fontsize=12)
        ax.set_title('Análise Risco x Retorno - Otimização Quântica', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('portfolio_quantum_risk_return.png', dpi=300, bbox_inches='tight')
        print("💾 Gráfico salvo: portfolio_quantum_risk_return.png")
        plt.show()

# ============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "OTIMIZAÇÃO QUÂNTICA DE PORTFOLIO" + " "*21 + "║")
    print("║" + " "*18 + "Usando QAOA + Qiskit + APIs" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    # ========================================================================
    # CONFIGURAÇÕES
    # ========================================================================
    
    # Parâmetros do portfolio
    BUDGET = 10000  # R$ 10.000
    RISK_AVERSION = 0.5  # Moderado (0 = só retorno, 1 = muito conservador)
    QAOA_REPS = 3  # Camadas do circuito quântico
    
    # Configuração de dados
    USE_REAL_DATA = True  # True = API real, False = simulado
    DATA_SOURCE = 'auto'  # 'yahoo', 'brapi', 'alpha_vantage', ou 'auto'
    PERIOD = '1y'  # Período de análise
    
    # Ativos para análise (ações brasileiras)
    TICKERS = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'WEGE3']
    
    # API Key (opcional - apenas para Alpha Vantage)
    # Obtenha gratuitamente em: https://www.alphavantage.co/support/#api-key
    ALPHA_VANTAGE_KEY = None  # Cole sua chave aqui se tiver
    
    print(f"\n⚙️ Parâmetros:")
    print(f"   • Orçamento: R$ {BUDGET:,.2f}")
    print(f"   • Aversão ao Risco: {RISK_AVERSION}")
    print(f"   • Repetições QAOA: {QAOA_REPS}")
    print(f"   • Fonte de Dados: {'API Real' if USE_REAL_DATA else 'Simulado'}")
    
    # ========================================================================
    # BUSCAR DADOS DE MERCADO
    # ========================================================================
    
    market_data = None
    
    if USE_REAL_DATA:
        fetcher = MarketDataFetcher(api_key=ALPHA_VANTAGE_KEY)
        market_data = fetcher.get_market_data(TICKERS, source=DATA_SOURCE, period=PERIOD)
        
        if market_data is None or len(market_data) < 2:
            print("\n⚠️ Dados insuficientes. Usando modo simulado...")
            USE_REAL_DATA = False
    
    # Carregar dados
    data = PortfolioData(
        market_data=market_data if USE_REAL_DATA else None,
        use_simulated=not USE_REAL_DATA
    )
    
    print(f"\n✅ Dados carregados: {len(data.assets)} ativos")
    
    # ========================================================================
    # OTIMIZAÇÃO QUÂNTICA
    # ========================================================================
    
    # Criar otimizador
    optimizer = QuantumPortfolioOptimizer(data, BUDGET, RISK_AVERSION)
    
    # Executar otimização quântica
    result, qp = optimizer.optimize_quantum(reps=QAOA_REPS)
    
    print(f"\n✅ Otimização quântica concluída!")
    print(f"   • Valor da função objetivo: {result.fval:.4f}")
    
    # Interpretar resultado
    solution = optimizer.interpret_result(result)
    
    if solution:
        # Análise
        analyzer = PortfolioAnalyzer(data, solution)
        analyzer.print_summary()
        analyzer.plot_allocation()
        analyzer.plot_risk_return()
        
        print("\n" + "="*70)
        print("✨ Análise completa!")
        print("🎓 Portfolio otimizado usando computação quântica (QAOA)")
        print("📊 Baseado em dados " + ("REAIS" if USE_REAL_DATA else "SIMULADOS"))
        print("="*70)
        
        # Dicas
        print("\n💡 DICAS:")
        print("   • Ajuste RISK_AVERSION para mudar perfil (0-1)")
        print("   • Use USE_REAL_DATA=True para dados de APIs")
        print("   • Para Alpha Vantage, registre-se e adicione sua chave")
        print("   • Tente diferentes períodos: '1mo', '3mo', '6mo', '1y', '2y'")
        
    else:
        print("\n❌ Não foi possível encontrar solução viável")
        print("   Tente ajustar os parâmetros ou usar mais ativos")

if __name__ == "__main__":
    main()