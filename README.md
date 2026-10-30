# taco-shell-rag

문서 폴더와 REST API의 자료를 연결하는 GPU 전용 RAG 스타터 킷입니다.
Claude·Codex 같은 MCP 클라이언트에서 연결한 자료를 검색할 수 있습니다.
검색 결과에는 근거가 된 원문과 출처, 문서 버전이 함께 담깁니다.

데이터 소스와 임베딩·리랭킹·생성 모델을 각각 설정할 수 있습니다.
검색 MCP만 사용한다면 생성 모델이나 Claude 계정은 필요하지 않습니다.

## 시작

Linux, NVIDIA GPU, Docker Compose와 NVIDIA Container Toolkit이 필요합니다.
임베딩과 리랭킹은 GPU에서 실행합니다. 드라이버와 컨테이너 준비 방법은 [설치 가이드](docs/install.md)에 있습니다.

기본 설정은 [예제 환불 문서](examples/knowledge/refunds.md) 한 개를 색인합니다.
저장소 루트에서 아래 명령을 실행하면 이 문서로 첫 검색을 확인할 수 있습니다.

```sh
python3 scripts/bootstrap.py --check
python3 scripts/bootstrap.py --write-env
docker compose config --quiet
docker compose up -d --build qdrant inference mcp
docker compose run --rm indexer --config /config/taco.toml sync
docker compose run --rm --no-deps indexer --config /config/taco.toml search 'refund'
```

첫 실행에는 이미지와 모델 가중치를 내려받는 시간이 걸립니다.
검색 결과에서 `document_key`, `version`, 원문과 출처를 확인합니다.
MCP 클라이언트는 `http://127.0.0.1:8080/mcp`에 연결하며, 인증에는 생성된 `.env`의 `MCP_AUTH_TOKEN`을 사용합니다.
원격 GPU 호스트에 연결하는 방법도 설치 가이드에 정리했습니다.

Codex나 Claude와 함께 설치하려면 `taco-install` 스킬을 사용합니다.
사용할 GPU·포트와 데이터 접근 범위를 확인한 뒤 설치 순서를 안내합니다.

## 데이터 연결과 설정

데모 검색이 동작하면 자신의 문서나 API를 연결합니다. 설정은 다음 파일에서 나누어 관리합니다.

| 바꾸려는 항목 | 설정 파일 |
|---|---|
| 비밀값, GPU 번호, 포트와 설정 파일 경로 | `.env` — `.env.example`의 한·영 주석 참고 |
| 임베딩·리랭커 모델과 실행 옵션 | `models.toml` |
| 데이터 소스, 저장소, 추론 API 주소와 검색 옵션 | `taco.toml` |
| 생성 모델과 에이전트 실행 방식 | `examples/models/*.toml` |

문서 폴더를 연결할 때는 `taco.toml`의 소스 경로와 Compose의 읽기 전용 마운트를 함께 바꿉니다.
REST API 연결 예제와 새 커넥터 작성 방법은 [데이터 연결 가이드](docs/connectors.md)에서 설명합니다.
검색 범위는 설정에 등록해 색인한 소스입니다.

## 모델 선택

기본 모델은 BGE-M3와 BGE-reranker-v2-m3입니다.
임베딩과 리랭커를 한 GPU 프로세스에서 실행하거나 각각 나누어 실행할 수 있습니다.

| 역할 | 선택 가능한 모델·연결 방식 |
|---|---|
| 임베딩 | BGE-M3, multilingual E5, Qwen3 0.6B·4B·8B 프리셋 |
| 리랭킹 | BGE, mixedbread, Qwen3 0.6B·4B·8B 프리셋 |
| 생성 | Claude CLI·API, OpenAI 호환 API로 호스팅한 Qwen·gpt-oss·Llama·Mistral 등 |

프리셋과 현재 모델 설정은 다음 명령으로 확인합니다.

```sh
uv run taco models
uv run taco models --model-config models.toml
```

프리셋에 없는 모델도 호환 백엔드와 고정 리비전을 지정해 연결합니다.
필요한 GPU 메모리와 실행 조건은 모델마다 다르므로, 선택한 모델의 실행 여부와 검색 품질은 사용할 환경에서 확인해야 합니다.
도구 호출을 지원하지 않는 생성 모델은 `rag` 모드로 사용합니다.
역할별 설정과 교체 절차는 [모델 연결 가이드](docs/models.md)에 정리했습니다.

## 구성

Go 서버는 MCP와 검색을, Python 서비스는 문서 수집·색인과 GPU 추론을 맡습니다.
벡터는 Qdrant에 저장합니다. 원문, 검색에 사용할 문서 버전, BM25 키워드 색인은 SQLite에서 관리합니다.

```text
폴더 / REST API → Python indexer → GPU embedder → Qdrant
                       └───────→ SQLite: 원문·버전·BM25

MCP 클라이언트 → Go MCP → Qdrant + BM25 → GPU reranker → 근거 청크
선택적 agent ──→ 같은 Go MCP + Claude / 자체 호스팅 LLM
```

기본 검색은 벡터 검색과 BM25 키워드 검색의 순위를 RRF로 결합합니다.
리랭킹을 켜면 결합한 후보를 GPU 리랭커로 다시 정렬합니다.
MCP는 HTTP와 stdio를 지원하며, 아래 도구는 모두 읽기 전용입니다.

| 도구 | 용도 |
|---|---|
| `search` | 질문과 관련된 근거를 하이브리드 검색으로 찾습니다. |
| `keyword_search` | 티켓 번호·버전·경로처럼 정확한 단어를 찾습니다. 모델을 호출하지 않습니다. |
| `deep_search` | 질문의 식별자와 출처를 나누어 탐색합니다. 계획은 규칙 기반입니다. |
| `get_document` | 문서 키와 버전을 지정해 검색 결과 앞뒤의 원문을 구간별로 읽습니다. |
| `status` | 소스별 체크포인트, 마지막 색인 성공과 오류를 확인합니다. |

검색 결과에는 출처·버전 외에 자료를 관측한 시점과 조회 범위, 읽지 못한 부분도 남깁니다.
답변은 이 결과를 읽는 MCP 클라이언트나 에이전트가 작성합니다.
도구별 한도와 검색 방식은 [검색 가이드](docs/retrieval.md)를 참고합니다.

Go의 서버 진입점은 `cmd/`, 검색 구현은 `internal/retriever/`에 있습니다.
Python 코드는 `src/taco_shell_rag/` 아래의 `indexer/`, `embedder/`, `reranker/`, `agent/`로 나뉩니다.
조직별 용어·업무 정책과 자동화는 코어 바깥에서 연결합니다.

## 동기화와 접근 범위

문서 수집이 실패하거나 원본 목록을 끝까지 읽지 못하면 현재 검색에 사용하는 문서를 유지합니다.
체크포인트는 새 문서를 검색에 반영하고 필요한 데이터 정리까지 마친 뒤 갱신합니다.
검색 결과가 없거나 자료 조회가 실패했다는 이유로 해당 정책이 없다고 판단해서는 안 됩니다.

같은 배포 인스턴스에 접속한 사용자는 연결된 모든 소스를 읽을 수 있습니다.
자료별 접근 권한을 구분해야 한다면 인스턴스를 분리해 운영합니다.

## 개발과 검증

소스 개발에는 Go 1.25 이상이 필요합니다. Docker로 빌드하면 Go 바이너리도 함께 준비합니다.
로컬 개발 환경의 빌드와 검사는 다음 명령으로 실행합니다.

```sh
uv sync --group dev
make build
make test
make check
```

기본 계약 테스트는 실제 Go HTTP·stdio 서버와 Python 색인기를 연결하고, 추론에는 테스트 대역을 사용합니다.
실제 모델을 사용한 GPU 실행과 검색 품질은 별도로 검증합니다.
GPU 추론 서버에 연결해 통합 테스트를 실행할 때는 [모델 연결 가이드](docs/models.md#go-mcp와-실제-gpuqdrant-연결-확인)에서 실행 조건을 확인합니다.

## 가이드

| 하려는 일 | 문서 |
|---|---|
| GPU 호스트 준비와 첫 검색 | [설치](docs/install.md) |
| 임베딩·리랭커·생성 모델 선택 | [모델 연결](docs/models.md) |
| 폴더·REST API 연결과 새 커넥터 작성 | [데이터 연결](docs/connectors.md) |
| 하이브리드·키워드·계획형 검색과 한도 설정 | [검색](docs/retrieval.md) |
| 색인 갱신 순서와 장애 처리 이해 | [구조와 동작](docs/architecture.md) |
| 로컬 개발과 변경 검증 | [개발 가이드](CONTRIBUTING.md) |

## 라이선스

Copyright 2026 kdgyun. [Apache-2.0](LICENSE) 라이선스를 적용합니다.
모델 가중치와 외부 데이터에는 각각 별도의 이용 조건이 적용됩니다.
