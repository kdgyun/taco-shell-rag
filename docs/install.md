# 설치

## 실행 환경

실제 추론은 Linux의 NVIDIA GPU에서 실행한다. NVIDIA 드라이버, Docker Compose,
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)이 필요하다.
GPU가 없는 호스트에서 CPU로 대체 실행하지 않는다. GPU 메모리 요구량은 선택한 모델과 입력 길이에 따라 달라진다.

Mac은 MCP 클라이언트나 계약 테스트용 개발 환경으로 사용한다. CUDA 실행 호스트는 아니다.
이미 다른 서비스가 사용하는 GPU라면 메모리와 GPU 번호를 확인한 뒤 실행한다.
이 가이드의 명령은 현재 저장소에만 적용한다. 시스템 패키지 설치나 Docker 재시작은 별도 승인 대상이다.

## 설정 준비

저장소 루트에서 실행한다.

```sh
python3 scripts/bootstrap.py --check
python3 scripts/bootstrap.py --write-env
```

첫 명령은 호스트 점검만 수행한다. 두 번째 명령은 토큰이 들어간 `.env`를 생성하며 기존 파일은 덮어쓰지 않는다.
`.env.example`의 한·영 주석을 보고 GPU 번호와 포트를 확인한다. `.env`를 셸에서 `source`하지 않는다.
토큰이나 외부 API 키를 채팅에 붙여 넣거나 Git에 추가하지 않는다.

문서 경로와 데이터 연결은 `taco.toml`에 설정한다. 기본 예제는 공개된 데모 문서뿐이다.
자신의 디렉터리를 연결하려면 `sources.path`와 Compose의 읽기 전용 마운트를 함께 바꾼다.
REST 연결은 `examples/rest-source.toml`을 참고한다. 원본 시스템에 접근해도 되는 자료만 연결한다.

## 기동과 확인

처음에는 이미지와 모델 가중치를 다운로드한다. 네트워크·디스크·GPU 사용을 승인받은 뒤 진행한다.

```sh
docker compose config --quiet
docker compose build
docker compose up -d qdrant inference mcp
docker compose exec mcp taco --config /config/taco.toml doctor
docker compose run --rm indexer --config /config/taco.toml sync
docker compose exec mcp taco --config /config/taco.toml search 'refund'
docker compose exec mcp taco --config /config/taco.toml status
```

추론 서비스가 준비되기 전에는 모델 다운로드나 로딩을 기다린다.
검색 결과에서 `document_key`, `version`, `index`, 원문과 출처를 확인한다.
MCP 클라이언트에서 같은 `document_key`와 `version`으로 `get_document`도 호출한다.
이 단계까지 성공해야 연결과 색인을 함께 확인한 것이다. 상태 점검 성공만으로 검색 품질을 보장하지 않는다.

주기적 동기화가 필요하면 `docker compose --profile sync up -d indexer`를 실행한다.
한 프로세스가 수집과 색인을 순서대로 수행하며, 다른 동기화 프로세스와 겹치면 쓰기 잠금을 획득하지 못한 쪽이 실패한다.
실패한 수집은 이전 게시 버전을 유지한다. 삭제는 완전한 원본 목록을 읽은 뒤 반영한다.
추론·저장소 서비스가 잠시 중단되면 오류 종류를 기록하고 다음 주기에 다시 시도한다.
실제 도구 호출이나 단발성 `sync` 실패는 성공으로 처리하지 않는다.

## MCP 클라이언트

주소는 `http://127.0.0.1:8080/mcp`이며 `Authorization: Bearer <private token>` 헤더가 필요하다.
Codex에서는 HTTP MCP 서버의 `bearer_token_env_var`를 `MCP_AUTH_TOKEN`으로 지정한다.
Claude Code에서는 HTTP MCP 서버와 인증 헤더를 설정한다. 실제 토큰은 클라이언트의 비밀 설정에 저장한다.

원격 GPU 호스트라면 로컬에서 `ssh -L 8080:127.0.0.1:8080 <gpu-host>`로 터널을 연 뒤 같은 주소에 연결한다.
기본 Compose는 Qdrant와 추론 API를 호스트에 공개하지 않는다.
컨테이너 내부에서 `0.0.0.0`으로 바인딩해도 MCP의 Host·Origin 검사는 유지한다.
SSH 터널 대신 전용 도메인의 역방향 프록시를 사용한다면 `taco.toml`의 `allowed_hosts`와
필요한 `allowed_origins`를 명시적으로 설정한다. 허용 대상을 넓히기 전에 TLS와 접근 권한을 확인한다.
Qdrant 이미지는 [v1.19.0](https://github.com/qdrant/qdrant/releases/tag/v1.19.0)으로 고정했다.
저장소나 클라이언트 버전을 바꿀 때는 양쪽의 호환 범위와 실제 API 동작을 함께 확인한다.
도구는 `search`, `get_document`, `status` 세 가지이며 쓰기 도구나 셸을 제공하지 않는다.

현재 인증 단위는 배포 인스턴스 전체다. 접속자는 그 인스턴스의 모든 연결 자료를 읽을 수 있다.
서로 다른 접근 권한이 필요한 데이터는 별도 인스턴스로 분리한다. 원본 시스템의 문서별 ACL을 자동 복제하지 않는다.

## 문제가 생겼을 때

- CUDA를 찾지 못하면 호스트 드라이버와 컨테이너 GPU 접근부터 확인한다. CPU로 전환하지 않는다.
- 모델 로딩에 실패하면 고정 리비전, 디스크, VRAM과 모델별 실행 요구사항을 확인한다.
- 인증 오류는 서버와 클라이언트의 토큰 설정을 대조한다. 토큰 자체를 로그에 출력하지 않는다.
- 색인 오류는 `status`의 마지막 시도·성공 시각, 체크포인트, 오류와 정리 대기 수를 확인한다.
- 임베딩 모델이나 분할 설정을 바꾼 경우 `docs/models.md`의 재색인 절차를 따른다.

볼륨 삭제나 전체 초기화는 기본 복구 방법이 아니다. 원인을 확인하기 전에는 `down -v`를 실행하지 않는다.
외부 응답 본문에는 비밀값이 섞일 수 있어 공개 오류에는 예외 종류만 남긴다.
이 정보만으로 원인을 좁힐 수 없으면 해당 서비스의 접근 제한된 로그를 확인한다.

## 개발 환경

```sh
uv sync --group dev
uv run pytest
uv run ruff check src tests scripts
```

테스트는 가짜 추론과 로컬 Qdrant를 사용한다. 외부 계정이나 실제 GPU 가중치 실행 결과로 해석하지 않는다.
