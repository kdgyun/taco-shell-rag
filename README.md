# taco-shell-rag

원하는 문서와 API를 연결하는 GPU 전용 RAG 스타터 킷.
검색 결과에 원문, 출처, 관측 시점과 조회 제한을 함께 반환한다.
답변 생성은 연결한 MCP 클라이언트가 담당한다.

## 구성

폴더와 REST API에서 읽은 문서를 공통 규격으로 색인한다.
GPU 추론 서비스가 임베딩과 재순위화를 맡고 Qdrant가 벡터를 저장한다.
MCP의 `search`, `get_document`, `status`는 읽기 전용이다.

```text
폴더 / REST API → 색인 코어 → GPU 임베딩 → Qdrant
                               검색·재순위화 ← MCP 클라이언트
                                                  또는
                                 선택적 agent → Claude / 자체 호스팅 LLM
```

연결 구현은 `adapters/`, 공통 계약과 검색·색인은 `core/`, 답변 생성은 `agent/`에 둔다.
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
docker compose up -d --build
docker compose run --rm indexer --config /config/taco.toml sync
```

Codex와 Claude에서 `taco-install` 스킬을 요청하면 환경 점검과 연결 순서를 안내한다.
포트, GPU, 데이터 범위와 권한을 확인한 뒤 실제 실행을 진행한다.
`.env.example`은 비밀값·모델·GPU 설정, `taco.toml`은 소스와 검색 설정을 설명한다.

## 모델 선택

[모델 연결 가이드](docs/models.md)에서 Claude API, Claude CLI, OpenAI 호환 서버를 선택한다.
Qwen·gpt-oss 예제를 제공하며 모델 이름을 코어에 고정하지 않는다.
도구 호출이 없는 모델은 명시적인 `rag` 모드를 사용한다.
검색 결과의 출처·버전·실행 시간과 agent의 도구 호출 결과를 확인할 수 있다.

## 데이터와 검증

실패한 수집은 기존 게시 문서를 지우지 않는다.
체크포인트는 게시와 필요한 정리가 끝난 뒤 갱신한다.
조회하지 못한 자료를 없는 정책으로 판단하지 않는다.
접속자는 해당 배포의 모든 소스를 읽을 수 있으므로 권한이 다른 자료는 인스턴스를 분리한다.

```sh
uv sync --group dev
uv run pytest
uv run ruff check src tests scripts
```

계약 테스트에는 실제 GPU나 외부 모델을 사용하지 않는다.
API 계약을 검증한 것과 모델 가중치·GPU 실행·검색 품질을 검증한 것은 별개다.
