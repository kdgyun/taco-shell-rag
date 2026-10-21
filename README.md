# taco-shell-rag

원하는 문서와 API를 연결하는 GPU 전용 RAG 스타터 킷.
검색 결과에 원문, 출처, 관측 시점과 조회 제한을 함께 반환한다.
검색 MCP만 연결하거나 별도로 고른 생성 모델로 agent를 실행한다.

## 구성

폴더와 REST API에서 읽은 문서를 공통 규격으로 색인한다.
Go 서버가 MCP와 검색을 맡고 Python 서비스가 수집·색인과 GPU 추론을 맡는다.
Qdrant는 벡터, SQLite는 게시 문서와 BM25 키워드 색인을 저장한다.
두 추론 역할은 모델·API 주소가 독립적이며 한 프로세스 또는 별도 GPU 프로세스로 실행한다.
MCP 도구는 `search`, `keyword_search`, `deep_search`, `get_document`, `status`이며 모두 읽기 전용이다.

```text
폴더 / REST API → Python indexer → GPU embedder → Qdrant
                       └───────→ SQLite: 게시 장부·BM25

MCP 클라이언트 → Go MCP → Qdrant + BM25 → GPU reranker → 근거 청크
선택적 agent ──→ 같은 Go MCP + Claude / 자체 호스팅 LLM
```

Go의 `cmd/`는 서버 진입점, `internal/retriever/`는 하이브리드·키워드·계획형 검색이다.
Python의 `src/taco_shell_rag/` 아래에 `indexer/`, `embedder/`, `reranker/`, `agent/`를 구분한다.
검색 MCP는 생성 모델이나 Claude 계정 없이도 사용한다.
회사별 용어, 업무 정책과 자동화는 기본 코어에 넣지 않는다.
[커넥터 작성 가이드](docs/connectors.md)에 소스 규격과 확장 위치를 정리했다.

## 시작

[설치 가이드](docs/install.md)를 먼저 읽는다. Linux NVIDIA GPU와 Container Toolkit이 필요하다.
임베딩·리랭킹은 GPU 전용이며 CPU 폴백을 제공하지 않는다.

```sh
python3 scripts/bootstrap.py --check
python3 scripts/bootstrap.py --write-env
docker compose config --quiet
docker compose up -d --build qdrant inference mcp
docker compose run --rm indexer --config /config/taco.toml sync
```

Codex와 Claude에서 `taco-install` 스킬을 요청하면 환경 점검과 연결 순서를 안내한다.
포트, GPU, 데이터 범위와 권한을 확인한 뒤 실제 실행을 진행한다.
`.env.example`은 비밀값·경로·GPU 배정, `models.toml`은 역할별 모델 선택,
`taco.toml`은 소스·검색·추론 API 연결을 설명한다.

기본 예제는 공개된 데모 문서 한 개다. 먼저 이 문서로 검색과 원문 조회를 확인한 뒤
자신의 소스를 추가한다. MCP를 연결했다고 로컬 저장소 전체나 외부 API를 임의로 탐색하는 것은 아니다.
연결한 소스만 색인하며 실시간 API 호출이나 파일 수정 도구는 제공하지 않는다.

## 모델 선택

[모델 연결 가이드](docs/models.md)에서 각 역할을 따로 선택한다.

- 임베딩: BGE-m3, multilingual E5, Qwen3 0.6B·4B·8B 프리셋.
- 리랭커: BGE, mixedbread, Qwen3 0.6B·4B·8B 프리셋.
- 생성: Claude CLI·API 또는 Qwen·gpt-oss·Llama·Mistral 등의 자체 호스팅 API.

```sh
uv run taco models
uv run taco models --model-config models.toml
```

프리셋 밖의 모델도 호환 backend와 고정 리비전을 직접 설정한다.
프리셋이 모든 GPU에서의 실행이나 품질을 보증하지는 않는다.
도구 호출이 없는 모델은 명시적인 `rag` 모드를 사용한다.
검색 결과의 출처·버전·실행 시간과 agent의 도구 호출 결과를 확인할 수 있다.

## 데이터와 검증

실패한 수집은 기존 게시 문서를 지우지 않는다.
체크포인트는 게시와 필요한 정리가 끝난 뒤 갱신한다.
조회하지 못한 자료를 없는 정책으로 판단하지 않는다.
접속자는 해당 배포의 모든 소스를 읽을 수 있으므로 권한이 다른 자료는 인스턴스를 분리한다.

```sh
uv sync --group dev
make build
make test
make check
```

소스 개발에는 Go 1.25 이상이 필요하다. 기본 Docker 빌드는 Go 바이너리를 함께 준비한다.
계약 테스트는 실제 Go HTTP·stdio 서버와 Python 색인기를 연결하며 GPU와 외부 모델은 대역을 사용한다.
API 계약을 검증한 것과 모델 가중치·GPU 실행·검색 품질을 검증한 것은 별개다.

## 가이드

| 하려는 일 | 문서 |
|---|---|
| GPU 호스트 준비와 첫 검색 | [설치](docs/install.md) |
| 임베딩·리랭커·생성 모델 선택 | [모델 연결](docs/models.md) |
| 폴더·REST API 연결과 새 커넥터 작성 | [데이터 연결](docs/connectors.md) |
| 하이브리드·키워드·계획형 검색과 한도 설정 | [검색](docs/retrieval.md) |
| 구성 요소와 장애 시 게시 흐름 이해 | [구조와 동작](docs/architecture.md) |
| 로컬 개발과 변경 검증 | [개발 가이드](CONTRIBUTING.md) |

## 라이선스

Copyright 2026 kdgyun. [Apache-2.0](LICENSE)을 적용한다.
모델 가중치와 외부 데이터의 이용 조건은 이 코드의 라이선스와 별개다.
