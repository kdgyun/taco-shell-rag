# 모델 연결

임베딩, 재순위화, 답변 생성은 서로 다른 역할이다.
생성 모델을 바꾸더라도 색인된 벡터를 다시 만들 필요는 없다.

| 역할 | 기본 경로 | 바꿀 설정 |
|---|---|---|
| 임베딩 | CUDA SentenceTransformer | `models.toml`의 `[embedding]` |
| 재순위화 | CUDA CrossEncoder | `models.toml`의 `[reranker]` |
| 생성 | 선택적 agent 앱 | provider, model, base_url, token_env, mode |

기본 검색은 정규화된 dense 벡터와 선택적 재순위화를 사용한다. sparse·hybrid 검색은 지원하지 않는다.
검색 응답에는 임베딩 공간과 실제 리랭커의 모델·리비전·실행 설정, 점수 종류, 실행 시간과 후보 제한이 담긴다.
리랭커가 실패하면 다른 모델이나 cosine 점수로 조용히 대체하지 않는다.

## 선택 가능한 모델

프리셋 목록과 해석된 설정은 GPU나 외부 API 없이 확인할 수 있다.

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

프리셋은 고정된 모델 리비전과 필요한 입력 설정을 제공한다. 추천 순위나 품질 보증은 아니다.
모든 프리셋을 실제 가중치로 실행해 검증했다는 뜻도 아니다. 크기·정밀도·입력 길이에 따라
필요한 GPU 메모리가 달라지므로 모델 카드와 자신의 하드웨어에서 확인한다.

E5 프리셋은 [모델 카드](https://huggingface.co/intfloat/multilingual-e5-large)에 맞는
query/document 접두어를 포함한다. instruct 변형은 질의에만 작업 지시를 붙인다.
Qwen3 리랭커는 고정된 SentenceTransformers의 네이티브 구현과 모델 저장소의 템플릿을 사용한다.
BGE의 분류 점수와 Qwen3의 yes/no logit 차이는 같은 확률 척도가 아니다.
[Qwen 모델 카드](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#usage)의 기본 점수 방식을 따르며
검색 결과에는 둘 다 보정하지 않은 점수로 표시한다.

## 역할별 설정

예를 들어 임베딩은 BGE, 리랭커는 Qwen3 4B로 조합할 수 있다.

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

같은 방식으로 8B나 다른 계열을 선택한다. `precision`은 `float32`, `float16`, `bfloat16` 중 하나이며
GPU·모델이 지원해야 한다. 메모리가 부족하다고 더 작은 모델이나 CPU로 바꾸지 않는다.
`device`는 프로세스 안에서 보이는 CUDA 번호다. Docker에 호스트 GPU 한 개를 배정하면 보통 `cuda:0`이 된다.

임베딩에는 `query_prompt`, `document_prompt`, `max_length`를 지정할 수 있고
리랭커에는 `prompt`, `max_length`를 지정할 수 있다. 프롬프트를 생략하면 고정 리비전의 기본값을 쓰고
빈 문자열은 프롬프트를 끈다. API의 문자 수 제한과 토큰 길이 제한은 별개다.
`rerank`는 질의와 문서를 합친 입력을 기본 512 tokens로 제한한다.

다른 파일을 쓰려면 `.env`의 `MODEL_CONFIG_PATH`를 바꾼다.
`examples/inference/qwen3.toml`, `e5-bge.toml`, `embedding-only.toml`을 시작점으로 사용할 수 있다.
모델은 기동 시 로딩되므로 설정을 바꾼 뒤 관련 추론 서비스를 다시 생성한다.
새 가중치 다운로드나 기존 GPU 프로세스 재시작 전에는 자원 사용과 영향을 먼저 확인한다.

프리셋에 없는 모델은 `preset` 대신 `model`과 고정 `revision`을 직접 적는다.

```toml
[embedding]
model = "your-org/your-native-embedding-model"
revision = "<모델 저장소의 40자리 commit SHA>"
backend = "sentence-transformers"
device = "cuda:0"
precision = "float16"
```

`preset`과 직접 지정한 `model`·`revision`을 함께 쓰면 오류다. 별칭이 잘못됐다고 기본 모델을 선택하지 않는다.
기본 구현은 SentenceTransformers가 네이티브로 지원하는 텍스트 모델만 대상으로 하며
`trust_remote_code`를 켜지 않는다. 리랭커는 질의·문서 쌍마다 scalar 점수 하나를 반환해야 한다.
다른 실행 코드가 필요한 모델은 별도의 검토된 추론 서버나 어댑터로 연결한다.
비공개·승인형 모델은 `HF_TOKEN`과 모델별 이용 조건을 확인한다.

## 한 프로세스 또는 분리 실행

기본 Compose는 한 GPU 프로세스에서 두 역할을 로딩하고 추론을 직렬화한다.
`taco.toml`의 `embedding_url`, `reranker_url`은 모두 `http://inference:8000`을 가리킨다.

역할마다 별도 프로세스/GPU가 필요하면 `.env`에서 다음을 지정한다.
아래 RAG 설정은 공개 예제 소스용이다. 실제 소스를 쓰고 있다면 자신의 설정 파일에
같은 엔드포인트 구성을 적용하고 그 경로를 지정한다.

```dotenv
RAG_CONFIG_PATH=./examples/separate-inference.toml
MODEL_CONFIG_PATH=./models.toml
EMBEDDING_GPU_ID=0
RERANKER_GPU_ID=1
```

자원 사용을 승인한 뒤 같은 Compose 파일 조합으로 실행·동기화한다.

```sh
docker compose -f compose.yaml -f compose.inference-split.yaml config --quiet
docker compose -f compose.yaml -f compose.inference-split.yaml up -d --build qdrant inference reranker mcp
docker compose -f compose.yaml -f compose.inference-split.yaml run --rm indexer --config /config/taco.toml sync
docker compose -f compose.yaml -f compose.inference-split.yaml exec mcp taco --config /config/taco.toml doctor
```

이 구성에서 `embedding_url`은 `http://inference:8000`, `reranker_url`은 `http://reranker:8000`이다.
색인은 임베딩만 호출하므로 리랭커 프로세스가 중단돼도 그것 때문에 색인이 실패하지 않는다.
검색은 선택한 리랭커가 준비되지 않으면 오류를 반환한다.
두 프로세스에 같은 GPU를 배정할 수도 있지만 모델 상주 메모리와 동시 추론의 피크를 함께 계산해야 한다.
서로 다른 프로세스 사이의 GPU 작업을 자동 직렬화하지 않는다.

리랭킹을 쓰지 않으려면 기본 Compose에서 `embedding-only.toml`을 선택하고 `taco.toml`에
`rerank = false`를 지정한다. `INFERENCE_ROLE=all`이어도 없는 `[reranker]`는 로딩하지 않는다.
분리 실행 override는 이 경우 사용하지 않는다. 생성 모델은 어느 구성에서도 별도 선택 사항이다.

기존 추론 서버를 연결할 때도 두 URL을 따로 지정한다. 내장 HTTP 어댑터의 계약은 다음과 같다.

인증이 필요하면 `embedding_token_env = "EMBEDDING_API_TOKEN"`,
`reranker_token_env = "RERANKER_API_TOKEN"`을 설정하고 실제 값은 `.env`에 넣는다.
토큰은 각 엔드포인트에 Bearer 헤더로 전달되며 리다이렉트는 따라가지 않는다.

- 임베딩: `GET /healthz`의 `embedding_space`, `POST /embed`의 `vectors`와 실행한 `embedding_space`.
- 리랭커: `GET /healthz`의 `reranker`, `POST /rerank`의 `scores`와 실제 실행한 `reranker` 정보.
- CUDA 장치와 모델·고정 리비전을 보고해야 한다. 다른 응답 규격은 어댑터가 필요하며 일반 `/v1/embeddings` 주소를 그대로 넣는다고 호환되지 않는다.

실제 응답 필드는 `core/contracts.py`, `adapters/inference.py`, `tests/test_inference.py`를 참고한다.
임베딩 실행 응답의 공간이 사전 확인한 공간과 다르면 해당 요청을 실패 처리한다.
상태 조회 직후 모델이 교체돼도 다른 공간의 벡터를 기존 색인에 섞지 않는다.
실행 모델 정보가 없는 임베딩 서버도 허용하지 않는다.
MCP의 `doctor`와 `search`는 실행 모델을 확인한다. `taco models --model-config`는 설정만 해석하며
실제 가중치 로딩이나 GPU 실행을 증명하지 않는다.

### 실제 GPU 추론 확인

CUDA가 준비된 Linux 호스트에서 선택한 설정으로 통합 테스트를 실행할 수 있다.
이 명령은 GPU를 사용하며 캐시에 없는 모델 가중치를 다운로드한다.
기본 Docker 이미지와 같은 Python 3.12를 지정한다. 프로젝트가 고정한 Torch 2.8은 Python 3.14용 CUDA 패키지를 제공하지 않는다.

```sh
RUN_GPU_TESTS=1 MODEL_CONFIG=models.toml \
  uv run --frozen --python 3.12 --extra gpu --group dev pytest -m gpu -q
```

`MODEL_CONFIG`를 다른 프로필 경로로 바꾸면 같은 검사로 모델 조합을 확인한다.
`INFERENCE_ROLE=embedding` 또는 `reranker`로 역할 하나만 검사할 수도 있다.
기본 `pytest`에서는 이 검사를 건너뛴다. 명시적으로 실행했는데 CUDA가 없으면 실패하며 CPU로 바꾸지 않는다.

검사 범위는 벡터 차원·정규화, 리랭커 점수, 추론 HTTP 응답과 데모 문서의 MCP 검색·원문 조회다.
검사용 Qdrant는 메모리 모드다. 실제 Qdrant 서버·Docker 연결과 업무별 검색 품질은 별도로 확인한다.
사용하지 않는 역할의 검사는 건너뛰며 생성 모델은 호출하지 않는다.

## Claude와 자체 호스팅 모델

`examples/models/`의 파일을 `taco ask --model-config`로 선택한다.

```sh
taco --config taco.toml ask '환불 규정은?' --model-config examples/models/qwen.toml
```

이 명령을 실행하는 환경에서 추론·저장소·생성 API 주소가 실제로 연결되어야 한다.
컨테이너의 `generator` 호스트 이름을 로컬 호스트 주소로 오해하지 않는다.

- `qwen.toml`, `gpt-oss.toml`, `llama.toml`, `mistral.toml`: OpenAI 호환 API와 도구 호출 예제.
- `claude-api.toml`: Anthropic Messages API용. 계정에서 사용할 모델 ID와 API 키가 필요하다.
- `claude-cli.toml`: 설치된 Claude CLI와 기존 로그인 사용. CLI를 패키지 이미지에 포함하지 않는다.
- `rag-only.toml`: 도구 호출이 없는 모델용. 먼저 검색한 뒤 원문을 넘기는 일반 RAG 모드.

Claude CLI는 `MCP_AUTH_TOKEN`으로 실행 중인 MCP에 접속한다. API 키와 CLI OAuth는 서로 대체하지 않는다.
CLI-managed 실행은 CLI가 내부 루프를 관리하므로 HTTP agent와 같은 수준의 모델·검증 정보를 제공하지 않는다.
이 어댑터는 `agent` 모드만 지원한다. 원격 MCP에 연결할 때 로컬 Qdrant나 색인 장부를 만들지 않는다.
HTTP agent 결과의 `requested_model`, `served_model`, `trace`, `tool_calls`, `evidence_reads`를 구분한다.
읽기 횟수는 모델이 자료를 이해했거나 답변이 정확하다는 증거가 아니다.

HTTP agent에서 원문 확인이 필수인 작업은 생성 모델 설정에
`required_tools = ["get_document"]`를 지정한다. 기본값은 빈 목록으로 모델이 도구 사용을 결정한다.
지정하면 해당 도구가 성공하기 전에는 답변을 완료 처리하지 않는다. `get_document`는 비어 있지 않은
본문 청크를 반환해야 한다. 보완 요청도 `max_steps` 안에서만 실행하며 끝까지 충족하지 못하면
`budget_exhausted`와 `missing_required_tools`를 반환한다. 필수 읽기 확인이지 답변 정확성 보증은 아니다.
이 옵션은 실행 과정을 직접 관리하는 HTTP agent용이며 Claude CLI와 `rag` 모드에는 적용하지 않는다.

Qwen·gpt-oss·Llama·Mistral의 예시 모델 ID는 다른 크기나 시리즈로 바꿀 수 있다.
서버가 노출하는 모델 이름과 일치시켜야 하며 모델별 도구 파서와 chat template도 확인한다.
다른 모델도 같은 API 계약을 제공하면 연결할 수 있다.
모델마다 문맥 길이, 양자화, 라이선스와 GPU 요구량이 다르다.

## 선택적 생성 서버

기존 생성 서버가 있다면 그 주소를 사용한다. 새 GPU 프로세스가 필요할 때만 `compose.generator.yaml`을 추가한다.
이 파일은 고정 vLLM 이미지, 모델 ID, 40자리 고정 리비전과 토큰을 요구한다.
사용할 이미지는 [vLLM 문서](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)에서 선택한다.
Qwen과 gpt-oss의 도구 파서는 같다고 가정하지 않는다. [모델별 도구 설정](https://docs.vllm.ai/en/latest/features/tool_calling/)을 확인한다.

`GENERATOR_TOOL_CALL_PARSER`는 agent 모드에 맞춰 선택한다. 도구 없는 서버는 이 값을 비우고
클라이언트의 `mode = "rag"`를 명시한다. 파서를 생략했다고 클라이언트 모드를 자동 전환하지 않는다.
thinking 모델은 `GENERATOR_REASONING_PARSER`도 모델과 vLLM 버전에 맞춰 지정한다.
동시 처리 수는 `GENERATOR_MAX_SEQUENCES`, eager 실행은 `GENERATOR_EAGER`로 선택한다.
이는 속도·VRAM·기동 시간에 영향을 주는 서버 옵션이며 모델 자체를 교체하는 설정은 아니다.

```sh
docker compose -f compose.yaml -f compose.generator.yaml config --quiet
docker compose -f compose.yaml -f compose.generator.yaml up -d generator
```

생성 서버가 준비되면 예제 설정으로 호출한다. 아래 명령은 실제 생성 API와 GPU를 사용한다.

```sh
docker compose -f compose.yaml -f compose.generator.yaml exec mcp \
  taco --config /config/taco.toml ask '환불 규정은?' \
  --model-config /config/examples/models/qwen.toml
```

기존 MCP 컨테이너를 먼저 띄웠다면 새 모델 설정 마운트를 적용하도록
같은 Compose 파일 조합으로 `up -d mcp`도 실행한다.

기본 생성 GPU 번호는 1, 임베딩·리랭킹은 0이다.
GPU가 하나라면 두 모델의 VRAM과 동시 실행 조건을 확인한 뒤 명시적으로 설정한다.
메모리 부족을 CPU 폴백으로 숨기지 않는다. 모델 ID가 같아도 parser나 chat template이 다르면 agent 동작이 달라질 수 있다.

## 임베딩 교체

임베딩 모델과 고정 리비전을 바꾸면 새 컬렉션과 새 색인 장부 경로를 지정한 뒤 전체 동기화를 실행한다.
기존 컬렉션과 장부는 보존한다. 새 검색 결과를 검증한 후 클라이언트를 전환한다.
차원만 같다고 같은 벡터 공간인 것은 아니다. 모델·리비전·차원뿐 아니라 임베딩 backend·정밀도·길이·프롬프트
설정도 지문에 포함한다. 이 값이 달라지면 기존 장부를 재사용하지 않는다.
GPU 번호와 배치 크기만 바꾸는 것은 재색인 사유가 아니다.
리랭커나 생성 모델만 바꾸면 벡터 재색인은 필요 없지만 자신의 질의와 원문으로 결과를 다시 검증한다.

이전 환경변수 방식에서 옮길 때는 모델 설정을 `models.toml`로 이동하고 `.env`의
`EMBEDDING_MODEL`, `EMBEDDING_MODEL_REVISION`, `RERANKER_MODEL`, `RERANKER_MODEL_REVISION`을 제거한다.
이 값이 남아 있으면 무시하지 않고 기동 오류로 안내한다. `inference_url`은 이전 임베딩 URL 별칭으로만 지원하며
새 설정은 `embedding_url`, `reranker_url`로 구분한다.
이전의 실행 옵션 없는 지문으로 만든 색인은 최초 전환 때 새 컬렉션·장부가 필요하다.
기존 설정·볼륨을 자동으로 덮어쓰거나 지우지 않는다.
