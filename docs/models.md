# 모델 연결

임베딩, 재순위화, 답변 생성은 서로 다른 역할이다.
생성 모델을 바꾸더라도 색인된 벡터를 다시 만들 필요는 없다.

| 역할 | 기본 경로 | 바꿀 설정 |
|---|---|---|
| 임베딩 | CUDA SentenceTransformer | `EMBEDDING_MODEL`, `EMBEDDING_MODEL_REVISION` |
| 재순위화 | CUDA CrossEncoder | `RERANKER_MODEL`, `RERANKER_MODEL_REVISION` |
| 생성 | 선택적 agent 앱 | provider, model, base_url, token_env, mode |

기본 검색은 정규화된 dense 벡터와 선택적 재순위화를 사용한다. sparse·hybrid 검색을 구현했다고 주장하지 않는다.
검색 응답에는 모델·리비전·차원, 실제 사용한 점수 종류, 실행 시간과 후보 제한이 담긴다.

## Claude와 자체 호스팅 모델

`examples/models/`의 파일을 `taco ask --model-config`로 선택한다.

```sh
taco --config taco.toml ask '환불 규정은?' --model-config examples/models/qwen.toml
```

이 명령을 실행하는 환경에서 추론·저장소·생성 API 주소가 실제로 연결되어야 한다.
컨테이너의 `generator` 호스트 이름을 로컬 호스트 주소로 오해하지 않는다.

- `qwen.toml`, `gpt-oss.toml`: OpenAI 호환 Chat Completions와 도구 호출을 사용하는 예제.
- `claude-api.toml`: Anthropic Messages API용. 계정에서 사용할 모델 ID와 API 키가 필요하다.
- `claude-cli.toml`: 설치된 Claude CLI와 기존 로그인 사용. CLI를 패키지 이미지에 포함하지 않는다.
- `rag-only.toml`: 도구 호출이 없는 모델용. 먼저 검색한 뒤 원문을 넘기는 일반 RAG 모드.

Claude CLI는 `MCP_AUTH_TOKEN`으로 실행 중인 MCP에 접속한다. API 키와 CLI OAuth는 서로 대체하지 않는다.
CLI-managed 실행은 CLI가 내부 루프를 관리하므로 HTTP agent와 같은 수준의 모델·검증 정보를 제공하지 않는다.
이 어댑터는 `agent` 모드만 지원한다. 원격 MCP에 연결할 때 로컬 Qdrant나 색인 장부를 만들지 않는다.
HTTP agent 결과의 `requested_model`, `served_model`, `trace`, `tool_calls`, `evidence_reads`를 구분한다.
읽기 횟수는 모델이 자료를 이해했거나 답변이 정확하다는 증거가 아니다.

Qwen, gpt-oss 외의 모델도 같은 API 계약을 제공하면 연결할 수 있다.
모든 모델의 가중치를 실행해 검증했다는 뜻은 아니다. [모델 후보 글](https://www.premai.io/blog/best-open-source-llms-for-rag-in-2026-10-models-ranked-by-retrieval-accuracy/)의 순위나 성능 수치를 이 프로젝트의 측정값으로 사용하지 않는다.
모델마다 문맥 길이, 양자화, 라이선스와 GPU 요구량이 다르다.

## 선택적 생성 서버

기존 생성 서버가 있다면 그 주소를 사용한다. 새 GPU 프로세스가 필요할 때만 `compose.generator.yaml`을 추가한다.
이 파일은 고정 vLLM 이미지, 모델 ID, 리비전, 토큰과 모델에 맞는 도구 파서를 요구한다.
사용할 이미지는 [vLLM 문서](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)에서 선택한다.
Qwen과 gpt-oss의 도구 파서는 같다고 가정하지 않는다. [모델별 도구 설정](https://docs.vllm.ai/en/latest/features/tool_calling/)을 확인한다.

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
차원만 같다고 같은 벡터 공간인 것은 아니다. 장부는 모델·리비전·차원 변경을 감지해 재사용을 거부한다.

기본 GPU 구현은 SentenceTransformers가 네이티브로 지원하는 모델을 대상으로 하며 `trust_remote_code`를 켜지 않는다.
쿼리 접두어·프롬프트·정규화가 필요한 모델은 해당 모델 카드와 `encode_query`·`encode_document` 설정을 확인한다.
모델 이름만 바꿨다고 모든 임베딩 방식이 같은 조건으로 동작하는 것은 아니다.
별도 실행 코드가 필요한 모델은 다른 추론 어댑터로 연결한다.
재순위화는 현재 입력을 512 tokens로 제한한다. 모델 교체 후 자신의 질의와 원문으로 순위를 검증한다.
