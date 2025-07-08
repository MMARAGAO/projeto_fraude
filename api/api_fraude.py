"""
🔐 API de Detecção de Fraudes em Cartões de Crédito
==================================================

Sistema completo para detecção de fraudes utilizando Machine Learning.
Desenvolvido com Flask e modelo XGBoost otimizado.

Autor: [Seu Nome]
Data: 7 de julho de 2025
Versão: 2.0.0
"""

# =============================================================================
# 📦 IMPORTAÇÕES E CONFIGURAÇÕES
# =============================================================================

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional

# Tentar importar flask_cors, se não existir, continuar sem CORS
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("⚠️ flask_cors não instalado. CORS não será habilitado.")
    print("💡 Para instalar: pip install flask-cors")

# =============================================================================
# 📁 CRIAÇÃO DE DIRETÓRIOS NECESSÁRIOS
# =============================================================================

def create_directories():
    """Cria os diretórios necessários para a aplicação"""
    directories = [
        '../logs',
        'logs',
        'models'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Não foi possível criar diretório {directory}: {e}")

# Criar diretórios antes de configurar logging
create_directories()

# =============================================================================
# ⚙️ CONFIGURAÇÃO DO LOGGING
# =============================================================================

def setup_logging():
    """Configura o sistema de logging de forma robusta"""
    try:
        # Tentar criar log no diretório pai primeiro
        log_handlers = [logging.StreamHandler()]
        
        # Tentar adicionar file handler
        log_paths = ['../logs/api_fraude.log', 'logs/api_fraude.log', 'api_fraude.log']
        
        for log_path in log_paths:
            try:
                # Criar diretório se necessário
                log_dir = os.path.dirname(log_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                
                # Testar se consegue escrever no arquivo
                with open(log_path, 'a') as f:
                    f.write('')
                
                log_handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
                print(f"📝 Log configurado em: {log_path}")
                break
                
            except Exception as e:
                continue
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=log_handlers
        )
        
        return True
        
    except Exception as e:
        print(f"⚠️ Erro ao configurar logging: {e}")
        # Configuração mínima apenas para console
        logging.basicConfig(level=logging.INFO)
        return False

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# 🚀 INICIALIZAÇÃO DA APLICAÇÃO
# =============================================================================

app = Flask(__name__)

# Configurar CORS se disponível
if CORS_AVAILABLE:
    CORS(app)

# Configurações da aplicação
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# =============================================================================
# 📊 CARREGAMENTO DOS MODELOS E ARTEFATOS
# =============================================================================

class ModelManager:
    """Gerenciador dos modelos e artefatos do ML"""
    
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.model_loaded = False
        
    def load_models(self):
        """Carrega todos os artefatos necessários"""
        try:
            # Caminhos dos modelos (tentar múltiplas localizações)
            base_paths = ['models/', '../models/', './']
            
            model_files = {
                'modelo': 'modelo_fraude.pkl',
                'scaler': 'scaler_amount.pkl',
                'features': 'feature_names.pkl',
                'metadata': 'modelo_metadata.pkl'
            }
            
            # Encontrar o caminho correto dos modelos
            model_paths = {}
            for model_name, filename in model_files.items():
                found = False
                for base_path in base_paths:
                    full_path = os.path.join(base_path, filename)
                    if os.path.exists(full_path):
                        model_paths[model_name] = full_path
                        found = True
                        break
                
                if not found:
                    # Tentar com scaler alternativo se não encontrar o metadata
                    if model_name == 'metadata':
                        logger.warning(f"⚠️ Arquivo {filename} não encontrado - continuando sem metadados")
                        model_paths[model_name] = None
                        continue
                    elif model_name == 'scaler':
                        # Criar um scaler dummy se necessário
                        logger.warning(f"⚠️ Arquivo {filename} não encontrado - criando scaler temporário")
                        model_paths[model_name] = None
                        continue
                    else:
                        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {filename}")
            
            # Carregar modelos
            logger.info("🔄 Carregando modelos...")
            
            # Modelo principal (obrigatório)
            self.modelo = joblib.load(model_paths['modelo'])
            logger.info("✅ Modelo principal carregado")
            
            # Scaler (criar um temporário se não existir)
            if model_paths['scaler']:
                self.scaler = joblib.load(model_paths['scaler'])
                logger.info("✅ Scaler carregado")
            else:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                # Valores representativos para treinar o scaler
                amount_sample = np.array([[0], [1], [25], [77], [2125.87]])
                self.scaler.fit(amount_sample)
                logger.info("⚠️ Scaler temporário criado")
            
            # Features (tentar carregar ou criar padrão)
            if model_paths['features']:
                self.feature_names = joblib.load(model_paths['features'])
                logger.info("✅ Feature names carregadas")
            else:
                # Features padrão do dataset de fraude
                self.feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount_Norm']
                logger.info("⚠️ Feature names padrão criadas")
            
            # Metadata (opcional)
            if model_paths['metadata']:
                self.metadata = joblib.load(model_paths['metadata'])
                logger.info("✅ Metadados carregados")
            else:
                self.metadata = {
                    'modelo_tipo': 'XGBoost',
                    'versao_modelo': '1.0.0',
                    'numero_features': len(self.feature_names),
                    'data_treinamento': 'Não disponível'
                }
                logger.info("⚠️ Metadados padrão criados")
            
            self.model_loaded = True
            logger.info("✅ Todos os modelos carregados com sucesso!")
            
            # Log das informações do modelo
            logger.info(f"📊 Modelo: {self.metadata.get('modelo_tipo', 'Desconhecido')}")
            logger.info(f"📋 Features: {self.metadata.get('numero_features', 'N/A')}")
            logger.info(f"📅 Treinado em: {self.metadata.get('data_treinamento', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {str(e)}")
            self.model_loaded = False
            return False
    
    def validate_input(self, dados: Dict) -> Tuple[bool, str]:
        """Valida os dados de entrada"""
        required_fields = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Verificar campos obrigatórios
        missing_fields = [field for field in required_fields if field not in dados]
        if missing_fields:
            return False, f"Campos obrigatórios não encontrados: {missing_fields}"
        
        # Verificar tipos de dados
        for field in required_fields:
            try:
                float(dados[field])
            except (ValueError, TypeError):
                return False, f"Campo '{field}' deve ser um número válido"
        
        # Verificar ranges válidos
        if dados['Amount'] < 0:
            return False, "Amount deve ser maior ou igual a zero"
        
        return True, "Dados válidos"
    
    def preprocess_data(self, dados: Dict) -> np.ndarray:
        """Preprocessa os dados para predição"""
        features = []
        
        # Adicionar features V1 a V28
        for i in range(1, 29):
            features.append(float(dados[f'V{i}']))
        
        # Normalizar Amount e adicionar
        amount_normalized = self.scaler.transform([[float(dados['Amount'])]])[0][0]
        features.append(amount_normalized)
        
        return np.array([features])

# Instância global do gerenciador
model_manager = ModelManager()

# =============================================================================
# 🏠 ROTA PRINCIPAL E DOCUMENTAÇÃO
# =============================================================================

@app.route('/')
def home():
    """Página principal com documentação da API"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔐 API de Detecção de Fraudes</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .status { padding: 10px; border-radius: 5px; margin: 20px 0; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { background: #28a745; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }
            .method.get { background: #17a2b8; }
            code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔐 API de Detecção de Fraudes v2.0.0</h1>
                <p>Sistema de Machine Learning para detecção de fraudes em cartões de crédito</p>
            </div>
            
            <div class="status {{ status_class }}">
                <strong>Status do Sistema:</strong> {{ status_message }}
            </div>
            
            <h2>📋 Endpoints Disponíveis</h2>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /</h3>
                <p>Página principal com documentação da API</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /health</h3>
                <p>Verificação de saúde da API e do modelo</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /predict</h3>
                <p>Realiza predição de fraude para uma transação</p>
                <p><strong>Entrada:</strong> JSON com campos V1-V28 e Amount</p>
                <p><strong>Saída:</strong> Resultado da predição e probabilidade</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /test</h3>
                <p>Teste com dados fictícios (transação normal)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /test/fraud</h3>
                <p>Teste com dados fictícios (transação fraudulenta)</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /info</h3>
                <p>Informações detalhadas sobre o modelo</p>
            </div>
            
            <h2>🧪 Exemplo de Uso</h2>
            <pre><code>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "V1": -1.359807,
    "V2": -0.072781,
    ...
    "V28": -0.021053,
    "Amount": 149.62
  }'</code></pre>
            
            <h2>📊 Informações do Modelo</h2>
            <ul>
                <li><strong>Algoritmo:</strong> {{ model_type }}</li>
                <li><strong>Features:</strong> {{ num_features }}</li>
                <li><strong>Versão:</strong> {{ model_version }}</li>
                <li><strong>Treinado em:</strong> {{ train_date }}</li>
                <li><strong>CORS:</strong> {{ cors_status }}</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Verificar status do modelo
    if model_manager.model_loaded:
        status_class = "success"
        status_message = "✅ Sistema operacional - Modelo carregado"
        model_info = model_manager.metadata or {}
    else:
        status_class = "error"
        status_message = "❌ Sistema com problemas - Modelo não carregado"
        model_info = {}
    
    return render_template_string(html_template,
                                 status_class=status_class,
                                 status_message=status_message,
                                 model_type=model_info.get('modelo_tipo', 'N/A'),
                                 num_features=model_info.get('numero_features', 'N/A'),
                                 model_version=model_info.get('versao_modelo', 'N/A'),
                                 train_date=model_info.get('data_treinamento', 'N/A'),
                                 cors_status="Habilitado" if CORS_AVAILABLE else "Desabilitado")

# =============================================================================
# 🏥 ROTA DE HEALTH CHECK
# =============================================================================

@app.route('/health')
def health_check():
    """Verificação de saúde da API"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'api_status': 'healthy',
            'model_loaded': model_manager.model_loaded,
            'version': '2.0.0',
            'cors_enabled': CORS_AVAILABLE
        }
        
        if model_manager.model_loaded:
            # Teste rápido do modelo
            test_data = np.zeros((1, len(model_manager.feature_names)))
            _ = model_manager.modelo.predict(test_data)
            status['model_status'] = 'healthy'
        else:
            status['model_status'] = 'error'
            status['api_status'] = 'degraded'
        
        return jsonify(status), 200 if status['api_status'] == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'api_status': 'error',
            'model_status': 'error',
            'error': str(e)
        }), 500

# =============================================================================
# 🔮 ROTA DE PREDIÇÃO PRINCIPAL
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """Realiza predição de fraude para uma transação"""
    start_time = datetime.now()
    
    try:
        # Verificar se o modelo está carregado
        if not model_manager.model_loaded:
            return jsonify({
                'erro': 'Modelo não carregado',
                'status': 'ERROR',
                'timestamp': start_time.isoformat()
            }), 503
        
        # Obter dados da requisição
        dados = request.get_json(force=True)
        if not dados:
            return jsonify({
                'erro': 'Dados JSON não fornecidos',
                'status': 'ERROR',
                'timestamp': start_time.isoformat()
            }), 400
        
        # Validar dados de entrada
        is_valid, validation_message = model_manager.validate_input(dados)
        if not is_valid:
            return jsonify({
                'erro': validation_message,
                'status': 'VALIDATION_ERROR',
                'timestamp': start_time.isoformat()
            }), 400
        
        # Preprocessar dados
        entrada = model_manager.preprocess_data(dados)
        
        # Fazer predição
        resultado = model_manager.modelo.predict(entrada)[0]
        probabilidades = model_manager.modelo.predict_proba(entrada)[0]
        prob_fraude = probabilidades[1]
        
        # Determinar nível de risco
        if prob_fraude >= 0.8:
            risk_level = "MUITO_ALTO"
        elif prob_fraude >= 0.6:
            risk_level = "ALTO"
        elif prob_fraude >= 0.4:
            risk_level = "MEDIO"
        elif prob_fraude >= 0.2:
            risk_level = "BAIXO"
        else:
            risk_level = "MUITO_BAIXO"
        
        # Calcular tempo de processamento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Log da predição
        logger.info(f"Predição realizada: Resultado={resultado}, Probabilidade={prob_fraude:.4f}, Tempo={processing_time:.2f}ms")
        
        # Resposta estruturada
        response = {
            'resultado': {
                'fraude': int(resultado),
                'probabilidade_fraude': round(float(prob_fraude), 4),
                'probabilidade_normal': round(float(probabilidades[0]), 4),
                'status': 'FRAUDE' if resultado == 1 else 'NORMAL',
                'nivel_risco': risk_level
            },
            'metadados': {
                'timestamp': start_time.isoformat(),
                'processing_time_ms': round(processing_time, 2),
                'modelo_tipo': model_manager.metadata.get('modelo_tipo', 'N/A'),
                'versao_api': '2.0.0'
            },
            'recomendacao': {
                'acao': 'BLOQUEAR' if resultado == 1 else 'APROVAR',
                'confianca': 'ALTA' if abs(prob_fraude - 0.5) > 0.3 else 'MEDIA'
            }
        }
        
        return jsonify(response), 200
        
    except KeyError as e:
        error_msg = f'Campo obrigatório não encontrado: {str(e)}'
        logger.warning(f"Erro de validação: {error_msg}")
        return jsonify({
            'erro': error_msg,
            'status': 'VALIDATION_ERROR',
            'timestamp': start_time.isoformat()
        }), 400
        
    except Exception as e:
        error_msg = f'Erro interno do servidor: {str(e)}'
        logger.error(f"Erro na predição: {error_msg}")
        return jsonify({
            'erro': error_msg,
            'status': 'INTERNAL_ERROR',
            'timestamp': start_time.isoformat()
        }), 500

# =============================================================================
# 🧪 ROTAS DE TESTE
# =============================================================================

@app.route('/test')
def test_normal():
    """Teste com dados fictícios de transação normal"""
    dados_teste = {
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'Amount': 149.62
    }
    
    return _execute_test(dados_teste, "Transação Normal (Esperado: NORMAL)")

@app.route('/test/fraud')
def test_fraud():
    """Teste com dados fictícios de transação fraudulenta"""
    dados_teste = {
        "V1": -4.832, "V2": 3.483, "V3": -5.069, "V4": 2.672,
        "V5": -2.281, "V6": -2.578, "V7": -3.292, "V8": -0.753,
        "V9": -1.822, "V10": -4.174, "V11": 3.498, "V12": -3.282,
        "V13": -0.237, "V14": -7.749, "V15": 0.543, "V16": -2.298,
        "V17": -2.493, "V18": -0.569, "V19": 0.992, "V20": 0.798,
        "V21": -0.137, "V22": 0.141, "V23": -0.206, "V24": 0.502,
        "V25": 0.219, "V26": -0.167, "V27": 0.096, "V28": -0.017,
        "Amount": 1.00
    }
    
    return _execute_test(dados_teste, "Transação Fraudulenta (Esperado: FRAUDE)")

def _execute_test(dados_teste: Dict, descricao: str):
    """Executa teste com dados fornecidos"""
    try:
        if not model_manager.model_loaded:
            return jsonify({
                'erro': 'Modelo não carregado para teste',
                'status': 'ERROR'
            }), 503
        
        # Preprocessar e predizer
        entrada = model_manager.preprocess_data(dados_teste)
        resultado = model_manager.modelo.predict(entrada)[0]
        probabilidades = model_manager.modelo.predict_proba(entrada)[0]
        
        return jsonify({
            'teste': {
                'descricao': descricao,
                'dados_entrada': dados_teste
            },
            'resultado': {
                'fraude': int(resultado),
                'probabilidade_fraude': round(float(probabilidades[1]), 4),
                'probabilidade_normal': round(float(probabilidades[0]), 4),
                'status': 'FRAUDE' if resultado == 1 else 'NORMAL'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erro no teste: {str(e)}")
        return jsonify({
            'erro': f'Erro durante o teste: {str(e)}',
            'status': 'TEST_ERROR'
        }), 500

# =============================================================================
# 📊 ROTA DE INFORMAÇÕES DO MODELO
# =============================================================================

@app.route('/info')
def model_info():
    """Retorna informações detalhadas sobre o modelo"""
    if not model_manager.model_loaded:
        return jsonify({
            'erro': 'Modelo não carregado',
            'status': 'ERROR'
        }), 503
    
    try:
        info = {
            'modelo': {
                'tipo': model_manager.metadata.get('modelo_tipo', 'N/A'),
                'versao': model_manager.metadata.get('versao_modelo', 'N/A'),
                'data_treinamento': model_manager.metadata.get('data_treinamento', 'N/A'),
                'numero_features': model_manager.metadata.get('numero_features', 'N/A')
            },
            'performance': model_manager.metadata.get('metricas_teste', {}),
            'features': model_manager.feature_names,
            'api': {
                'versao': '2.0.0',
                'cors_enabled': CORS_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Erro ao obter informações: {str(e)}")
        return jsonify({
            'erro': f'Erro ao obter informações: {str(e)}',
            'status': 'ERROR'
        }), 500

# =============================================================================
# 🚨 TRATAMENTO DE ERROS GLOBAIS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'erro': 'Endpoint não encontrado',
        'status': 'NOT_FOUND',
        'endpoints_disponiveis': ['/', '/health', '/predict', '/test', '/test/fraud', '/info'],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'erro': 'Método HTTP não permitido',
        'status': 'METHOD_NOT_ALLOWED',
        'timestamp': datetime.now().isoformat()
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'erro': 'Erro interno do servidor',
        'status': 'INTERNAL_ERROR',
        'timestamp': datetime.now().isoformat()
    }), 500

# =============================================================================
# 🚀 INICIALIZAÇÃO DA APLICAÇÃO
# =============================================================================

def initialize_app():
    """Inicializa a aplicação e carrega os modelos"""
    print("🚀 Iniciando API de Detecção de Fraudes v2.0.0...")
    print("="*60)
    
    # Informações de CORS
    if CORS_AVAILABLE:
        print("✅ CORS habilitado")
    else:
        print("⚠️ CORS desabilitado (flask-cors não instalado)")
    
    # Tentar carregar os modelos
    if model_manager.load_models():
        print("✅ Sistema inicializado com sucesso!")
        print(f"📊 Modelo: {model_manager.metadata.get('modelo_tipo', 'N/A')}")
        print(f"📋 Features: {model_manager.metadata.get('numero_features', 'N/A')}")
        print(f"🎯 Performance (Recall): {model_manager.metadata.get('metricas_teste', {}).get('recall', 'N/A')}")
    else:
        print("⚠️ Sistema inicializado com problemas no modelo")
        print("🔧 Verifique se os arquivos do modelo estão na pasta 'models/'")
    
    print("\n🔗 Endpoints disponíveis:")
    print("  📋 Documentação: http://127.0.0.1:5000")
    print("  🏥 Health Check: http://127.0.0.1:5000/health")
    print("  🔮 Predição: POST http://127.0.0.1:5000/predict")
    print("  🧪 Teste Normal: http://127.0.0.1:5000/test")
    print("  🚨 Teste Fraude: http://127.0.0.1:5000/test/fraud")
    print("  📊 Info Modelo: http://127.0.0.1:5000/info")
    print("\n" + "="*60)

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)