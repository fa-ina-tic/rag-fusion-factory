# RAG Fusion Factory

**Intelligent Search Result Fusion with Automated Weight Optimization**

### Trying Vibe Coding to check my Theory : Fine-tuning weights using simple XGBoost on hybrid search(for Convex Combination) would allow us to find optimal weights while making the finetuning process much easier.

RAG Fusion Factory는 여러 검색 엔진의 결과를 지능적으로 융합하여 최적의 검색 성능을 달성하는 시스템입니다. 머신러닝 기반의 자동화된 가중치 최적화와 대조 학습(Contrastive Learning)을 통해 도메인별 최적의 검색 결과 조합을 찾아줍니다.

## 🎯 주요 기능

- **🔍 다중 검색 엔진 통합**: 여러 검색 엔진의 결과를 동시에 활용
- **🤖 자동 가중치 최적화**: XGBoost 기반 머신러닝으로 최적의 융합 가중치 자동 계산
- **📊 스코어 정규화**: 다양한 정규화 방법으로 검색 엔진 간 점수 표준화
- **🎓 대조 학습**: Contrastive Learning을 통한 지능적인 결과 순위 최적화
- **⚙️ 유연한 설정**: YAML 기반 계층적 설정 시스템
- **🚀 고성능 API**: FastAPI 기반 비동기 REST API
- **📈 성능 모니터링**: 실시간 성능 추적 및 메트릭 수집

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Search APIs   │    │  Normalization   │    │  Fusion Model   │
│                 │───▶│     Engine       │───▶│   (XGBoost)     │
│ • Engine A      │    │                  │    │                 │
│ • Engine B      │    │ • Min-Max        │    │ • Weight Calc   │
│ • Engine C      │    │ • Z-Score        │    │ • Score Fusion  │
└─────────────────┘    │ • Quantile       │    └─────────────────┘
                       └──────────────────┘             │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Training Data   │    │ Contrastive      │    │ Ranked Results  │
│                 │───▶│   Learning       │◀───│                 │
│ • Query-Result  │    │                  │    │ • Fused Scores  │
│ • Ground Truth  │    │ • Triplet Loss   │    │ • Confidence    │
│ • Relevance     │    │ • Pair Mining    │    │ • Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/rag-fusion-factory.git
cd rag-fusion-factory

# 의존성 설치
pip install -r requirements.txt
```

### 설정

```bash
# 환경 설정 파일 생성
cp .env.template .env
cp config/user.yaml.template config/user.yaml

# 설정 확인
python3 scripts/config_manager.py summary
```

### 실행

```bash
# 개발 모드로 실행
ENVIRONMENT=development python3 src/main.py

# 또는 프로덕션 모드
ENVIRONMENT=production python3 src/main.py
```

## 📁 프로젝트 구조

```
rag-fusion-factory/
├── src/                          # 소스 코드
│   ├── models/                   # 데이터 모델
│   │   ├── core.py              # 핵심 데이터 클래스
│   │   └── __init__.py
│   ├── services/                 # 비즈니스 로직
│   ├── adapters/                 # 검색 엔진 어댑터
│   ├── api/                      # REST API 엔드포인트
│   ├── config/                   # 설정 관리
│   │   ├── settings.py          # OmegaConf 기반 설정
│   │   └── utils.py             # 설정 유틸리티
│   ├── utils/                    # 공통 유틸리티
│   │   └── logging.py           # 로깅 설정
│   └── main.py                   # 애플리케이션 진입점
├── config/                       # YAML 설정 파일
│   ├── default.yaml             # 기본 설정
│   ├── development.yaml         # 개발 환경 설정
│   ├── production.yaml          # 프로덕션 환경 설정
│   ├── user.yaml.template       # 사용자 설정 템플릿
│   └── README.md                # 설정 가이드
├── scripts/                      # 관리 스크립트
│   └── config_manager.py        # 설정 관리 CLI
├── .kiro/specs/                  # 프로젝트 명세서
│   └── rag-fusion-factory/
│       ├── requirements.md      # 요구사항
│       ├── design.md           # 설계 문서
│       └── tasks.md            # 구현 태스크
├── requirements.txt              # Python 의존성
├── .env.template                # 환경변수 템플릿
└── README.md                    # 프로젝트 문서
```

## ⚙️ 설정 관리

### 환경별 설정

```yaml
# config/development.yaml
api:
  debug: true
  port: 8001

logging:
  level: "DEBUG"
  file_logging: true

model:
  xgboost:
    n_estimators: 50 # 개발용 빠른 학습
```

### 환경변수 오버라이드

```bash
# API 설정 오버라이드
export RAG_API_HOST=localhost
export RAG_API_PORT=8080
export RAG_API_DEBUG=true

# 모델 설정 오버라이드
export RAG_MODEL_CACHE_DIR=/custom/path/models
export RAG_LOG_LEVEL=DEBUG
```

### CLI 도구 사용

```bash
# 설정 검증
python3 scripts/config_manager.py validate

# 설정 요약 보기
python3 scripts/config_manager.py summary

# 설정 편집 가이드 보기
python3 scripts/config_manager.py edit --key api.port --value 8080
python3 scripts/config_manager.py instructions

# 환경 변경
python3 scripts/config_manager.py env --set production
```

## 🔧 핵심 컴포넌트

### 데이터 모델

```python
from src.models import SearchResult, SearchResults, TrainingExample

# 검색 결과
result = SearchResult(
    document_id="doc_123",
    relevance_score=0.85,
    content="검색된 문서 내용",
    metadata={"source": "engine_a"},
    engine_source="elasticsearch"
)

# 학습 데이터
training_example = TrainingExample(
    query="머신러닝 알고리즘",
    engine_results={"engine_a": results_a, "engine_b": results_b},
    ground_truth_labels={"doc_123": 1.0, "doc_456": 0.8}
)
```

### 설정 접근

```python
from src.config.settings import config, get_xgboost_config

# 설정 값 접근
api_host = config.api.host
batch_size = config.training.batch_size

# XGBoost 설정 가져오기
xgb_params = get_xgboost_config()
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 지원

- **이슈 리포팅**: [GitHub Issues](https://github.com/your-username/rag-fusion-factory/issues)
- **설정 가이드**: [config/README.md](config/README.md)

---
