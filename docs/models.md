# 모델 연결

임베딩은 문서를 벡터로 만들고, 리랭커는 검색 후보를 다시 정렬합니다. 생성 모델은 찾은 자료를 읽어 답변합니다.
각 역할을 따로 선택하므로 생성 모델을 바꾸더라도 저장된 문서 벡터를 다시 만들 필요는 없습니다.

| 바꾸려는 모델 | 설정할 곳 | 기본 실행 방식 |
|---|---|---|
| 임베딩 | `models.toml`의 `[embedding]` | CUDA SentenceTransformer |
| 리랭커 | `models.toml`의 `[reranker]` | CUDA CrossEncoder |
| 답변 생성 | 생성 모델 설정 파일의 provider, model, base_url, token_env, mode | 선택적 에이전트 |

기본 검색은 정규화한 dense 벡터와 BM25를 합치고, 리랭킹을 켜면 후보를 다시 정렬합니다.
모델이 제공하는 sparse 벡터는 사용하지 않습니다. 검색 방식은 [검색 가이드](retrieval.md)를 참고합니다.

## 사용할 모델 확인

아래 첫 명령은 선택할 수 있는 프리셋 목록을 보여줍니다. 두 번째 명령은 `models.toml`을 읽어
모델 이름·리비전·CUDA 장치·정밀도 등을 보여줍니다. 생략한 옵션의 기본값도 함께 확인할 수 있습니다.
모델을 로딩하거나 외부 API를 호출하지 않아 GPU 없이 실행할 수 있습니다.

```sh
uv run taco models
uv run taco models --model-config models.toml
```

| 역할 | 계열 | 프리셋 |
|---|---|---|
| 임베딩 | BGE | `bge-m3` |
| 임베딩 | multilingual E5 | `multilingual-e5-base`, `multilingual-e5-large`, `multilingual-e5-large-instruct` |
| 임베딩 | Qwen3 | `qwen3-embedding-0.6b`, `qwen3-embedding-4b`, `qwen3-embedding-8b` |
| 리랭커 | BGE | `bge-reranker-v2-m3`, `bge-reranker-large` |
| 리랭커 | mixedbread | `mxbai-rerank-base-v1` |
| 리랭커 | Qwen3 | `qwen3-reranker-0.6b`, `qwen3-reranker-4b`, `qwen3-reranker-8b` |

프리셋에는 모델 리비전과 필요한 입력 설정이 미리 지정되어 있습니다. 목록의 순서는 추천 순위가 아니며,
모든 프리셋을 실제 가중치로 실행해 품질을 검증한 목록도 아닙니다. GPU 메모리는 모델 크기·정밀도·입력 길이에
따라 달라지므로 모델 카드와 사용할 하드웨어를 기준으로 실행 가능 여부를 확인합니다.

E5는 [모델 카드](https://huggingface.co/intfloat/multilingual-e5-large)에 맞춰 query/document 접두어를 붙입니다.
instruct 변형은 질문에만 작업 지시를 붙입니다. Qwen3 리랭커는 고정한 SentenceTransformers 버전의
네이티브 구현과 모델 저장소의 템플릿을 사용합니다.

BGE의 분류 점수와 Qwen3의 yes/no logit 차이는 같은 확률 척도로 비교할 수 없습니다.
Qwen3는 [모델 카드의 기본 점수 방식](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#usage)을 따르며,
두 리랭커 모두 검색 결과에는 보정하지 않은 점수로 표시합니다.

## 임베딩과 리랭커 설정

예를 들어 임베딩은 BGE, 리랭커는 Qwen3 4B로 조합합니다.

```toml
[embedding]
preset = "bge-m3"
backend = "sentence-transformers"
device = "cuda:0"
precision = "float32"
batch_size = 8

[reranker]
preset = "qwen3-reranker-4b"
backend = "cross-encoder"
device = "cuda:0"
precision = "float16"
batch_size = 4
max_length = 512
```

다른 크기나 계열도 같은 방법으로 지정합니다. `precision`에는 `float32`, `float16`, `bfloat16` 중
GPU와 모델이 지원하는 값을 씁니다. 지정한 모델의 메모리가 부족하면 실행에 실패하며, 더 작은 모델이나 CPU로 자동 교체하지 않습니다.

`device`는 해당 프로세스에서 보이는 CUDA 번호입니다. Docker에 호스트 GPU 한 개를 배정하면
컨테이너에서는 보통 `cuda:0`이 됩니다.

| 역할 | 입력 관련 옵션 |
|---|---|
| 임베딩 | `query_prompt`, `document_prompt`, `max_length` |
| 리랭커 | `prompt`, `max_length` |

프롬프트 옵션을 생략하면 고정한 리비전의 기본값을 사용합니다. 빈 문자열을 지정하면 프롬프트를 끕니다.
API의 문자 수 제한과 모델의 토큰 길이 제한은 별개이며, `rerank`는 질문과 문서를 합친 입력을 기본 512 tokens로 제한합니다.

모델 설정 파일 경로는 `.env`의 `MODEL_CONFIG_PATH`로 정합니다.
다른 조합은 `examples/inference/qwen3.toml`, `e5-bge.toml`, `embedding-only.toml`에서 시작할 수 있습니다.
모델은 기동할 때 로딩하므로 설정을 바꾼 뒤에는 해당 추론 서비스를 다시 생성합니다.
새 가중치를 받거나 기존 GPU 프로세스를 재시작하기 전에는 자원 사용과 다른 서비스에 미칠 영향을 확인합니다.

### 프리셋에 없는 모델

`preset` 대신 `model`과 고정 `revision`을 적습니다.

```toml
[embedding]
model = "your-org/your-native-embedding-model"
revision = "<모델 저장소의 40자리 commit SHA>"
backend = "sentence-transformers"
device = "cuda:0"
precision = "float16"
```

`preset`과 `model`·`revision`을 함께 쓰면 설정 오류가 납니다. 잘못된 프리셋 이름을 넣었을 때도
기본 모델로 바꾸지 않고 오류를 알립니다.

기본 추론 서비스는 SentenceTransformers가 네이티브로 지원하는 텍스트 모델을 실행합니다.
`trust_remote_code`는 켜지 않습니다. 별도 실행 코드가 필요한 모델은 검토한 추론 서버나 어댑터를 통해 연결합니다.
리랭커는 질문·문서 쌍마다 scalar 점수 하나를 반환해야 합니다.
비공개 또는 승인이 필요한 모델은 `HF_TOKEN`과 모델의 이용 조건을 확인합니다.

## 한 GPU에서 실행하거나 나누어 실행하기

기본 Compose는 한 GPU 프로세스에 임베딩과 리랭커를 로딩하고, 두 모델의 추론을 순서대로 실행합니다.
`taco.toml`의 `embedding_url`, `reranker_url`은 모두 `http://inference:8000`입니다.

프로세스나 GPU를 역할별로 나누려면 `.env`에 다음 값을 지정합니다.
`RAG_CONFIG_PATH`의 예제는 공개 데모 소스를 사용합니다. 이미 자신의 소스를 연결했다면 기존 설정 파일에
같은 엔드포인트 구성을 적용하고 그 파일의 경로를 지정합니다.

```dotenv
RAG_CONFIG_PATH=./examples/separate-inference.toml
MODEL_CONFIG_PATH=./models.toml
EMBEDDING_GPU_ID=0
RERANKER_GPU_ID=1
```

자원 사용 승인을 받은 뒤 같은 Compose 파일 조합으로 실행하고 동기화합니다.

```sh
docker compose -f compose.yaml -f compose.inference-split.yaml config --quiet
docker compose -f compose.yaml -f compose.inference-split.yaml up -d --build qdrant inference reranker mcp
docker compose -f compose.yaml -f compose.inference-split.yaml run --rm indexer --config /config/taco.toml sync
docker compose -f compose.yaml -f compose.inference-split.yaml run --rm --no-deps indexer --config /config/taco.toml doctor
```

이때 임베딩 주소는 `http://inference:8000`, 리랭커 주소는 `http://reranker:8000`입니다.
색인은 임베딩만 호출하므로 리랭커 중단만으로 실패하지 않습니다.
검색에서 사용하기로 한 리랭커가 준비되지 않았으면 검색은 오류를 반환합니다.

두 프로세스에 같은 GPU를 배정할 수도 있지만 모델 상주 메모리와 동시 추론의 피크를 함께 계산해야 합니다.
서로 다른 프로세스의 GPU 작업까지 자동으로 순서를 조정하지는 않습니다.

### 리랭커를 쓰지 않는 경우

기본 Compose에서 `embedding-only.toml`을 선택하고 `taco.toml`에 `rerank = false`를 지정합니다.
`INFERENCE_ROLE=all`이어도 설정에 `[reranker]`가 없으면 리랭커는 로딩하지 않습니다.
이 구성에는 분리 실행 override를 사용하지 않습니다. 답변 생성 모델은 어느 구성에서든 별도로 선택합니다.

## 기존 추론 API 연결

이미 실행 중인 추론 서버가 있다면 `embedding_url`, `reranker_url`에 주소를 각각 지정합니다.
인증이 필요하면 `embedding_token_env = "EMBEDDING_API_TOKEN"`,
`reranker_token_env = "RERANKER_API_TOKEN"`을 설정하고 실제 토큰은 `.env`에 넣습니다.
각 주소에 Bearer 헤더로 전달하며 리다이렉트는 따라가지 않습니다.

내장 어댑터는 아래 형식의 응답을 사용합니다.

| API | 필요한 응답 |
|---|---|
| 임베딩 `GET /healthz` | `embedding_space`와 `embedding_fingerprint` |
| 임베딩 `POST /embed` | `vectors`, 실제 실행한 `embedding_space`와 `embedding_fingerprint` |
| 리랭커 `GET /healthz` | `reranker` 정보 |
| 리랭커 `POST /rerank` | `scores`와 실제 실행한 `reranker` 정보 |

서버는 CUDA 장치, 모델, 고정 리비전을 보고해야 합니다. 일반 `/v1/embeddings` 주소만 적는다고
이 형식에 맞춰지는 것은 아니므로, 다른 응답을 쓰는 서버에는 어댑터가 필요합니다.
전체 필드는 `core/contracts.py`, `adapters/inference.py`, `tests/test_inference.py`에서 확인합니다.

### 설정과 실제 모델이 같은지 확인

`taco models --model-config`는 파일에 적힌 설정을 읽습니다. 실제 실행 중인 모델은 CLI의 `doctor`와
Go MCP의 `search`가 받은 추론 응답으로 확인합니다. 설정 조회만으로 가중치가 로딩되거나 GPU에서 실행됐다고 볼 수는 없습니다.

색인과 검색에 같은 임베딩 설정을 쓰는지 비교할 때는 `EmbeddingSpace.fingerprint`를 사용합니다.
모델·리비전·차원·인코딩 설정으로 계산한 값이며 CUDA 장치 번호는 벡터 공간에 영향을 주지 않습니다.
Go는 이 값과 색인 당시의 모델 정보를 함께 확인합니다.

실제 응답의 임베딩 설정이 사전 확인 결과와 다르면 해당 요청을 실패로 처리합니다.
상태 조회 직후 모델이 바뀌어도 다른 모델의 벡터를 기존 색인에 섞지 않습니다.
실행한 모델 정보를 돌려주지 않는 서버도 사용할 수 없습니다.
검색 응답에는 임베딩 공간, 실제 리랭커의 모델·리비전·실행 설정, 점수 종류, 실행 시간과 후보 제한이 남습니다.
리랭커가 실패했을 때 다른 모델이나 cosine 점수로 대신 응답하지 않습니다.

## Claude와 자체 호스팅 모델

답변 생성은 `examples/models/`의 설정 파일을 `taco ask --model-config`로 선택합니다.

```sh
taco --config taco.toml ask '환불 규정은?' --model-config examples/models/qwen.toml
```

이 명령을 실행하는 곳에서 추론·저장소·생성 API에 접근할 수 있어야 합니다.
예제의 `generator`는 Compose 내부의 호스트 이름이므로 로컬 호스트 주소와 구분합니다.

| 설정 파일 | 연결 방식 |
|---|---|
| `qwen.toml`, `gpt-oss.toml`, `llama.toml`, `mistral.toml` | OpenAI 호환 API와 도구 호출 |
| `claude-api.toml` | Anthropic Messages API. 계정에서 사용 가능한 모델 ID와 API 키가 필요합니다. |
| `claude-cli.toml` | 설치된 Claude CLI의 기존 로그인을 사용합니다. CLI는 패키지 이미지에 포함하지 않습니다. |
| `rag-only.toml` | 도구 호출 없이 검색 후 원문을 넘겨 한 번 생성하는 RAG 모드 |

Qwen·gpt-oss·Llama·Mistral 예제의 모델 ID는 다른 크기나 시리즈로 바꿀 수 있습니다.
API가 제공하는 모델 이름과 일치시키고, 모델별 도구 파서와 chat template을 맞춥니다.
같은 API 형식을 제공하는 다른 모델도 연결할 수 있으며 문맥 길이·양자화·라이선스·GPU 요구량은 모델마다 확인합니다.

### Claude CLI를 사용할 때

CLI는 `MCP_AUTH_TOKEN`으로 실행 중인 MCP에 접속합니다. 원격 MCP를 사용하면 로컬에 Qdrant나 문서 DB를 만들지 않습니다.
CLI는 `agent` 모드만 지원하며, Anthropic API 키와 CLI OAuth는 서로 대신 사용할 수 없습니다.

CLI가 도구 호출을 포함한 내부 실행 과정을 관리하므로 HTTP 에이전트와 같은 수준의 모델·검증 정보를 제공하지는 않습니다.
HTTP 에이전트에서는 요청한 모델과 실제 응답한 모델을 `requested_model`, `served_model`로 구분하고,
`trace`, `tool_calls`, `evidence_reads`에서 실행과 읽기 기록을 확인합니다.
읽기 횟수만으로 모델이 내용을 이해했거나 답변이 정확하다고 판단할 수는 없습니다.

### 답변 전에 원문 읽기를 요구할 때

HTTP 에이전트의 생성 모델 설정에 `required_tools = ["get_document"]`를 지정합니다.
기본값은 빈 목록이어서 모델이 도구 사용을 결정합니다. 지정했다면 `get_document`가 성공하고
비어 있지 않은 본문 청크를 반환해야 답변을 완료 처리합니다.

필요한 읽기가 빠졌을 때 보완하는 요청도 `max_steps` 안에서만 실행합니다.
끝까지 충족하지 못하면 `budget_exhausted`와 `missing_required_tools`를 반환합니다.
이 설정은 원문을 읽었는지 확인하는 용도이며 답변 내용의 정확성을 검사하지는 않습니다.
실행 과정을 직접 관리하는 HTTP 에이전트에만 적용하고 Claude CLI와 `rag` 모드에는 적용하지 않습니다.

## 선택적 생성 서버

생성 서버가 이미 있다면 그 주소를 사용합니다. 새 GPU 프로세스를 실행할 때만 `compose.generator.yaml`을 추가합니다.
고정한 vLLM 이미지, 모델 ID, 40자리 고정 리비전, 토큰이 필요합니다.
[vLLM의 OpenAI 호환 서버 문서](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)에서 이미지를 선택하고
[모델별 도구 설정](https://docs.vllm.ai/en/latest/features/tool_calling/)에 따라 파서를 맞춥니다.
Qwen과 gpt-oss가 같은 도구 파서를 사용한다고 가정하지 않습니다.

| 옵션 | 용도 |
|---|---|
| `GENERATOR_TOOL_CALL_PARSER` | agent 모드에서 모델의 도구 호출을 읽을 파서 |
| `GENERATOR_REASONING_PARSER` | thinking 모델의 출력에 맞는 파서. 모델과 vLLM 버전에 맞춰 지정합니다. |
| `GENERATOR_MAX_SEQUENCES` | 동시 처리 수 |
| `GENERATOR_EAGER` | eager 실행 여부 |

도구 호출이 없는 서버는 `GENERATOR_TOOL_CALL_PARSER`를 비우고 클라이언트에 `mode = "rag"`를 지정합니다.
파서를 생략해도 클라이언트 모드가 자동으로 바뀌지는 않습니다.
동시 처리 수와 eager 실행은 속도·VRAM·기동 시간에 영향을 주는 서버 설정입니다.

```sh
docker compose -f compose.yaml -f compose.generator.yaml config --quiet
docker compose -f compose.yaml -f compose.generator.yaml up -d generator
```

생성 서버가 준비되면 다음 명령으로 예제 질문을 보냅니다. 실제 생성 API와 GPU를 사용합니다.
Python CLI가 들어 있는 `indexer`에서 실행하고, 생성 모델 설정은 읽기 전용 볼륨으로 연결합니다.

```sh
docker compose -f compose.yaml -f compose.generator.yaml run --rm --no-deps \
  --volume "$PWD/examples/models:/config/examples/models:ro" \
  indexer --config /config/taco.toml ask '환불 규정은?' \
  --model-config /config/examples/models/qwen.toml
```

기본 GPU 번호는 생성 모델이 1, 임베딩·리랭킹이 0입니다.
GPU가 하나라면 함께 사용할 생성·임베딩·리랭킹 모델의 VRAM과 동시 실행 조건을 확인한 뒤 번호를 직접 지정합니다.
메모리가 부족하면 CPU로 대신 실행하지 않습니다. 모델 ID가 같아도 parser와 chat template이 다르면
에이전트의 동작이 달라질 수 있습니다.

## 임베딩 교체

임베딩 설정이 바뀌면 같은 문장도 다른 벡터로 변환될 수 있습니다.
새 Qdrant 컬렉션과 문서·버전을 저장할 SQLite 파일 경로를 지정해 전체 동기화합니다.
검색 결과를 확인한 뒤 클라이언트를 전환하고 전환 전까지 기존 컬렉션과 SQLite 파일을 보존합니다.

| 변경한 값 | 재색인 |
|---|---|
| 임베딩 모델·리비전·차원 | 새 컬렉션과 SQLite 파일이 필요합니다. 차원이 같아도 다른 모델의 벡터는 섞지 않습니다. |
| 임베딩 backend·정밀도·길이·프롬프트 | 지문에 포함되는 설정이므로 기존 문서 DB를 재사용하지 않습니다. |
| GPU 번호·배치 크기 | 이 변경만으로는 재색인이 필요하지 않습니다. |
| 리랭커·생성 모델 | 벡터 재색인은 필요 없지만 자신의 질문과 원문으로 결과를 다시 확인합니다. |

이전 환경변수 방식에서 옮길 때는 모델 설정을 `models.toml`에 적고 `.env`의 `EMBEDDING_MODEL`,
`EMBEDDING_MODEL_REVISION`, `RERANKER_MODEL`, `RERANKER_MODEL_REVISION`을 제거합니다.
이 값이 남아 있으면 기동할 때 오류로 이전 방법을 안내합니다.
`inference_url`은 이전 임베딩 URL의 별칭으로만 지원하므로 새 설정에는 `embedding_url`, `reranker_url`을 구분해 적습니다.

실행 옵션을 포함하지 않던 이전 지문으로 만든 색인도 최초 전환 때 새 컬렉션과 SQLite 파일이 필요합니다.
기존 설정과 볼륨을 자동으로 덮어쓰거나 삭제하지 않습니다.

## 실제 모델로 검증하기

### 실제 GPU 추론 확인

CUDA가 준비된 Linux 호스트에서 선택한 모델 설정으로 실행합니다.
GPU를 사용하고 캐시에 없는 가중치는 다운로드합니다. 기본 이미지와 같은 Python 3.12를 선택합니다.
고정된 Torch 2.8에는 Python 3.14용 CUDA 패키지가 없습니다.

```sh
RUN_GPU_TESTS=1 MODEL_CONFIG=models.toml \
  uv run --frozen --python 3.12 --extra gpu --group dev pytest -m gpu -q
```

`MODEL_CONFIG`에 다른 프로필 경로를 지정하면 그 모델 조합을 검사합니다.
`INFERENCE_ROLE=embedding` 또는 `reranker`로 한 역할만 확인할 수도 있습니다.
기본 `pytest`는 이 검사를 건너뛰지만, 직접 실행했을 때 CUDA가 없으면 CPU로 바꾸지 않고 실패합니다.

확인하는 항목은 벡터 차원과 정규화, 리랭커 점수, 추론 HTTP 응답, 데모 문서를 Python으로 색인하고 원문을 조회하는 동작입니다.
여기서 Qdrant는 메모리 모드로 사용합니다. 쓰지 않는 역할의 검사는 건너뛰며 생성 모델은 호출하지 않습니다.
기본 `tests/test_go_mcp.py`는 실제 Go HTTP·stdio 서버를 사용하되 추론과 Qdrant HTTP 응답은 테스트 대역을 씁니다.

### Go MCP와 실제 GPU·Qdrant 연결 확인

실행 중인 임베딩·리랭커 서버와 테스트용 Qdrant에 Go MCP를 연결해 확인하는 검사입니다.
먼저 두 서비스를 준비하고, 검사 프로세스에서 접근할 수 있는 URL을 지정합니다.
기본 Compose는 추론과 Qdrant 포트를 호스트에 공개하지 않으므로 같은 Docker 네트워크에서 실행하거나
루프백 연결을 별도로 준비합니다. 테스트를 위해 포트를 인터넷에 공개하지 않습니다.

```sh
RUN_GO_GPU_TESTS=1 \
GPU_TEST_INFERENCE_URL=http://127.0.0.1:8000 \
GPU_TEST_QDRANT_URL=http://127.0.0.1:6333 \
  uv run --frozen --python 3.12 --group dev pytest tests/integration/test_go_gpu.py -q
```

검사 프로세스에는 Go 도구 체인이 필요합니다. 모델 가중치를 다시 로딩하지 않고 추론 API의 CUDA 모델·리비전과
검색 응답을 대조합니다. 리랭커를 별도 실행 중이면 `GPU_TEST_RERANKER_URL`도 지정합니다.
인증 정보는 각각 `GPU_TEST_EMBEDDING_TOKEN`, `GPU_TEST_RERANKER_TOKEN`, `GPU_TEST_QDRANT_TOKEN`으로 전달합니다.

매 실행마다 고유한 테스트 컬렉션과 임시 문서 DB를 만들고, 끝나면 해당 컬렉션만 삭제합니다.
테스트용 Qdrant를 지정하고 컬렉션 생성·삭제 권한을 확인합니다.
HTTP·stdio 양쪽에서 하이브리드·dense·키워드·계획형 검색, 출처 필터, 정확한 버전의 원문 조회,
연결 중 검색 대상 문서 버전의 갱신을 확인합니다. 생성 모델이나 업무별 검색 정확도를 평가하는 검사는 아닙니다.
