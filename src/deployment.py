"""
Sistema de Deploy para Estimação de Distância - Produção
Inclui API REST, Docker e monitoramento
"""

# ============================================================================
# deployment.py - Sistema de Deploy em Produção
# ============================================================================

import os
import time
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis
import logging

from water_distance_estimation_final import (
    WaterDistancePredictor, 
    ModelConfig,
    ModelEvaluator,
    BiasCalibrator
)

# ============================================================================
# CONFIGURAÇÃO E LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações do ambiente
class DeployConfig:
    MODEL_PATH = os.getenv('MODEL_PATH', './models/final_model.h5')
    CALIBRATION_PATH = os.getenv('CALIBRATION_PATH', './models/calibration.npz')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', 32))
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))
    API_PORT = int(os.getenv('API_PORT', 8000))
    ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'

# ============================================================================
# MÉTRICAS PROMETHEUS
# ============================================================================

# Contadores
prediction_counter = Counter(
    'water_distance_predictions_total',
    'Total number of distance predictions'
)

error_counter = Counter(
    'water_distance_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Histogramas
prediction_latency = Histogram(
    'water_distance_prediction_duration_seconds',
    'Time spent processing prediction'
)

predicted_distance = Histogram(
    'water_distance_predicted_meters',
    'Histogram of predicted distances',
    buckets=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
)

# Gauges
model_confidence = Gauge(
    'water_distance_model_confidence',
    'Current model confidence score'
)

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class PredictionRequest(BaseModel):
    """Modelo de requisição para predição"""
    image_base64: Optional[str] = Field(None, description="Imagem em base64")
    return_confidence: bool = Field(False, description="Retornar intervalo de confiança")
    apply_calibration: bool = Field(True, description="Aplicar calibração")

class PredictionResponse(BaseModel):
    """Modelo de resposta para predição"""
    distance_meters: float = Field(..., description="Distância estimada em metros")
    distance_cm: float = Field(..., description="Distância estimada em centímetros")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Intervalo de confiança")
    processing_time_ms: float = Field(..., description="Tempo de processamento em ms")
    timestamp: str = Field(..., description="Timestamp da predição")
    model_version: str = Field("1.0.0", description="Versão do modelo")

class BatchPredictionRequest(BaseModel):
    """Modelo para predição em lote"""
    images_base64: List[str] = Field(..., description="Lista de imagens em base64")
    apply_calibration: bool = Field(True, description="Aplicar calibração")

class HealthResponse(BaseModel):
    """Modelo de resposta para health check"""
    status: str
    model_loaded: bool
    calibration_loaded: bool
    cache_connected: bool
    uptime_seconds: float

# ============================================================================
# SERVIÇO DE PREDIÇÃO
# ============================================================================

class PredictionService:
    """Serviço principal de predição com cache e otimizações"""
    
    def __init__(self, config: DeployConfig):
        self.config = config
        self.predictor = None
        self.redis_client = None
        self.start_time = time.time()
        
        # Inicializar modelo
        self._load_model()
        
        # Inicializar cache Redis
        if self._init_redis():
            logger.info("Cache Redis conectado")
        else:
            logger.warning("Cache Redis não disponível, continuando sem cache")
    
    def _load_model(self):
        """Carrega modelo e calibração"""
        try:
            self.predictor = WaterDistancePredictor(
                model_path=self.config.MODEL_PATH,
                calibration_path=self.config.CALIBRATION_PATH
            )
            logger.info("Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _init_redis(self) -> bool:
        """Inicializa conexão com Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                decode_responses=True
            )
            self.redis_client.ping()
            return True
        except:
            self.redis_client = None
            return False
    
    def _get_cache_key(self, image_hash: str, calibration: bool) -> str:
        """Gera chave de cache"""
        return f"prediction:{image_hash}:{calibration}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Busca resultado no cache"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except:
            pass
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict):
        """Salva resultado no cache"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key,
                self.config.CACHE_TTL,
                json.dumps(result)
            )
        except:
            pass
    
    def decode_image(self, image_base64: str) -> np.ndarray:
        """Decodifica imagem de base64"""
        import base64
        
        try:
            # Remover header se presente
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            # Decodificar
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Não foi possível decodificar a imagem")
            
            return image
        
        except Exception as e:
            logger.error(f"Erro ao decodificar imagem: {e}")
            raise HTTPException(status_code=400, detail="Imagem inválida")
    
    @prediction_latency.time()
    async def predict(self, 
                     image_base64: str,
                     return_confidence: bool = False,
                     apply_calibration: bool = True) -> Dict:
        """
        Realiza predição com cache e métricas
        
        Args:
            image_base64: Imagem em base64
            return_confidence: Retornar intervalo de confiança
            apply_calibration: Aplicar calibração
            
        Returns:
            Dicionário com resultados
        """
        start_time = time.time()
        
        # Verificar cache
        import hashlib
        image_hash = hashlib.md5(image_base64.encode()).hexdigest()
        cache_key = self._get_cache_key(image_hash, apply_calibration)
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Resultado encontrado no cache: {cache_key}")
            return cached_result
        
        # Decodificar imagem
        image = self.decode_image(image_base64)
        
        # Realizar predição
        try:
            result = self.predictor.predict(image, return_confidence)
            
            # Selecionar distância baseada em calibração
            if apply_calibration and 'distance_calibrated' in result:
                distance_cm = result['distance_calibrated']
            else:
                distance_cm = result['distance_raw']
            
            distance_m = distance_cm / 100.0
            
            # Atualizar métricas
            prediction_counter.inc()
            predicted_distance.observe(distance_m)
            
            # Preparar resposta
            response = {
                'distance_meters': distance_m,
                'distance_cm': distance_cm,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0.0'
            }
            
            if return_confidence and 'confidence_interval' in result:
                response['confidence_interval'] = {
                    'lower_cm': result['confidence_interval']['lower'],
                    'upper_cm': result['confidence_interval']['upper'],
                    'std_cm': result['confidence_interval']['std']
                }
                
                # Calcular confiança baseada no desvio padrão
                confidence = 1.0 / (1.0 + result['confidence_interval']['std'] / 100.0)
                model_confidence.set(confidence)
            
            # Salvar no cache
            self._save_to_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            error_counter.labels(error_type='prediction').inc()
            logger.error(f"Erro na predição: {e}")
            raise HTTPException(status_code=500, detail="Erro no processamento")
    
    async def predict_batch(self,
                          images_base64: List[str],
                          apply_calibration: bool = True) -> List[Dict]:
        """
        Predição em lote otimizada
        
        Args:
            images_base64: Lista de imagens em base64
            apply_calibration: Aplicar calibração
            
        Returns:
            Lista de resultados
        """
        # Limitar tamanho do batch
        if len(images_base64) > self.config.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size máximo é {self.config.MAX_BATCH_SIZE}"
            )
        
        # Decodificar imagens
        images = []
        for img_b64 in images_base64:
            images.append(self.decode_image(img_b64))
        
        # Processar batch
        try:
            results = self.predictor.predict_batch(images)
            
            # Formatar resultados
            formatted_results = []
            for result in results:
                if apply_calibration and 'distance_calibrated' in result:
                    distance_cm = result['distance_calibrated']
                else:
                    distance_cm = result['distance_raw']
                
                formatted_results.append({
                    'distance_meters': distance_cm / 100.0,
                    'distance_cm': distance_cm
                })
            
            # Atualizar métricas
            prediction_counter.inc(len(results))
            for result in formatted_results:
                predicted_distance.observe(result['distance_meters'])
            
            return formatted_results
            
        except Exception as e:
            error_counter.labels(error_type='batch_prediction').inc()
            logger.error(f"Erro na predição em lote: {e}")
            raise HTTPException(status_code=500, detail="Erro no processamento em lote")
    
    def get_health_status(self) -> Dict:
        """Retorna status de saúde do serviço"""
        return {
            'status': 'healthy' if self.predictor else 'unhealthy',
            'model_loaded': self.predictor is not None,
            'calibration_loaded': (self.predictor and 
                                 self.predictor.calibrator.calibration_curve is not None),
            'cache_connected': self.redis_client is not None,
            'uptime_seconds': time.time() - self.start_time
        }

# ============================================================================
# API FASTAPI
# ============================================================================

app = FastAPI(
    title="Water Distance Estimation API",
    description="API para estimação de distância sobre superfícies aquáticas",
    version="1.0.0"
)

# Inicializar serviço
config = DeployConfig()
service = PredictionService(config)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check"""
    return service.get_health_status()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Realiza predição de distância única
    
    Aceita imagem em base64 e retorna distância estimada
    """
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="Imagem é obrigatória")
    
    result = await service.predict(
        request.image_base64,
        request.return_confidence,
        request.apply_calibration
    )
    
    return PredictionResponse(**result)

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Realiza predição em lote
    
    Aceita lista de imagens e retorna lista de distâncias
    """
    results = await service.predict_batch(
        request.images_base64,
        request.apply_calibration
    )
    
    return {
        'predictions': results,
        'count': len(results),
        'timestamp': datetime.now().isoformat()
    }

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...),
                         return_confidence: bool = False,
                         apply_calibration: bool = True):
    """
    Realiza predição com upload de arquivo
    
    Aceita arquivo de imagem e retorna distância estimada
    """
    # Validar tipo de arquivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    
    # Ler arquivo
    contents = await file.read()
    
    # Converter para base64
    import base64
    image_base64 = base64.b64encode(contents).decode('utf-8')
    
    # Realizar predição
    result = await service.predict(
        image_base64,
        return_confidence,
        apply_calibration
    )
    
    return PredictionResponse(**result)

@app.get("/metrics")
async def get_metrics():
    """Endpoint para métricas Prometheus"""
    if config.ENABLE_MONITORING:
        return generate_latest()
    else:
        raise HTTPException(status_code=404, detail="Monitoramento desabilitado")

@app.post("/calibrate")
async def update_calibration(predictions: List[float],
                            ground_truth: List[float]):
    """
    Atualiza calibração do modelo
    
    Requer lista de predições e valores verdadeiros
    """
    try:
        # Criar novo calibrador
        calibrator = BiasCalibrator()
        calibrator.calibrate(
            np.array(predictions),
            np.array(ground_truth)
        )
        
        # Atualizar no serviço
        service.predictor.calibrator = calibrator
        
        # Salvar calibração
        calibrator.save(config.CALIBRATION_PATH)
        
        return {
            'status': 'success',
            'message': 'Calibração atualizada com sucesso'
        }
        
    except Exception as e:
        logger.error(f"Erro ao atualizar calibração: {e}")
        raise HTTPException(status_code=500, detail="Erro ao atualizar calibração")

# ============================================================================
# WORKER ASSÍNCRONO PARA PROCESSAMENTO EM LOTE
# ============================================================================

class BatchProcessor:
    """Processador assíncrono para grandes volumes"""
    
    def __init__(self, service: PredictionService):
        self.service = service
        self.queue = asyncio.Queue()
        self.results = {}
    
    async def process_queue(self):
        """Processa fila de predições"""
        while True:
            try:
                # Aguardar itens na fila
                batch = []
                
                # Coletar batch
                while len(batch) < self.service.config.MAX_BATCH_SIZE:
                    try:
                        item = await asyncio.wait_for(
                            self.queue.get(), 
                            timeout=0.1
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Processar batch
                    images = [item['image'] for item in batch]
                    results = await self.service.predict_batch(images)
                    
                    # Armazenar resultados
                    for item, result in zip(batch, results):
                        self.results[item['id']] = result
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Erro no processador de batch: {e}")
                await asyncio.sleep(1)
    
    async def add_to_queue(self, task_id: str, image_base64: str):
        """Adiciona tarefa à fila"""
        await self.queue.put({
            'id': task_id,
            'image': image_base64
        })
        
        return task_id
    
    def get_result(self, task_id: str) -> Optional[Dict]:
        """Obtém resultado processado"""
        return self.results.pop(task_id, None)

# Inicializar processador de batch
batch_processor = BatchProcessor(service)

@app.on_event("startup")
async def startup_event():
    """Inicia workers assíncronos"""
    asyncio.create_task(batch_processor.process_queue())
    logger.info("API iniciada com sucesso")

@app.post("/predict/async")
async def predict_async(request: PredictionRequest):
    """
    Adiciona predição à fila assíncrona
    
    Retorna ID da tarefa para consulta posterior
    """
    import uuid
    task_id = str(uuid.uuid4())
    
    await batch_processor.add_to_queue(task_id, request.image_base64)
    
    return {
        'task_id': task_id,
        'status': 'queued'
    }

@app.get("/predict/async/{task_id}")
async def get_async_result(task_id: str):
    """
    Consulta resultado de predição assíncrona
    
    Args:
        task_id: ID da tarefa
    """
    result = batch_processor.get_result(task_id)
    
    if result:
        return {
            'task_id': task_id,
            'status': 'completed',
            'result': result
        }
    else:
        return {
            'task_id': task_id,
            'status': 'processing'
        }

# ============================================================================
# DOCKER CONFIGURATION
# ============================================================================

DOCKERFILE_CONTENT = """
# Dockerfile para deploy do sistema de estimação de distância

FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    wget \\
    libglib2.0-0 \\
    libgl1-mesa-glx \\
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar diretórios necessários
RUN mkdir -p models logs checkpoints

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/final_model.h5
ENV CALIBRATION_PATH=/app/models/calibration.npz

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização
CMD ["uvicorn", "deployment:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKER_COMPOSE_CONTENT = """
# docker-compose.yml para deploy completo

version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/final_model.h5
      - CALIBRATION_PATH=/app/models/calibration.npz
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - ENABLE_MONITORING=true
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
"""

REQUIREMENTS_CONTENT = """
# requirements.txt para deploy

# Deep Learning
tensorflow==2.13.0
opencv-python==4.8.0.74
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
Pillow==10.0.0

# API
fastapi==0.100.0
uvicorn[standard]==0.23.1
python-multipart==0.0.6
pydantic==2.0.3

# Cache e Monitoramento
redis==4.6.0
prometheus-client==0.17.1

# Utilitários
python-dotenv==1.0.0
aiofiles==23.1.0
"""

# ============================================================================
# SCRIPT DE INICIALIZAÇÃO
# ============================================================================

def create_deployment_files():
    """Cria arquivos necessários para deploy"""
    
    # Criar Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write(DOCKERFILE_CONTENT)
    
    # Criar docker-compose.yml
    with open('docker-compose.yml', 'w') as f:
        f.write(DOCKER_COMPOSE_CONTENT)
    
    # Criar requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS_CONTENT)
    
    # Criar prometheus.yml
    prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'water-distance-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
"""
    
    with open('prometheus.yml', 'w') as f:
        f.write(prometheus_config)
    
    # Criar estrutura de diretórios
    for dir_path in ['models', 'logs', 'checkpoints', 'grafana/dashboards', 'grafana/datasources']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Arquivos de deployment criados com sucesso")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'setup':
            create_deployment_files()
            print("Arquivos de deployment criados. Execute 'docker-compose up' para iniciar.")
        elif sys.argv[1] == 'dev':
            # Modo desenvolvimento
            uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    else:
        # Modo produção
        uvicorn.run(app, host="0.0.0.0", port=8000)
