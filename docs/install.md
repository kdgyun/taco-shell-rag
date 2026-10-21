# 설치

## 실행 환경

실제 추론은 Linux의 NVIDIA GPU에서 실행한다. NVIDIA 드라이버, Docker Compose·Buildx,
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)이 필요하다.
GPU가 없는 호스트에서 CPU로 대체 실행하지 않는다. GPU 메모리 요구량은 선택한 모델과 입력 길이에 따라 달라진다.

MCP·검색 서버는 Go, 색인·GPU 추론은 Python으로 실행한다.
소스 개발에는 Go 1.25 이상과 Python 3.11~3.13이 필요하다. Docker 빌드가 Go 바이너리를 준비하며
Python 기본 이미지는 3.12를 사용한다.
소스에서 직접 설치할 때도 `uv sync --python 3.12`처럼 버전을 지정한다.
Python 3.14에는 현재 고정한 Torch 2.8 CUDA 패키지가 없어 설치할 수 없다.

Mac은 MCP 클라이언트나 계약 테스트용 개발 환경으로 사용한다. CUDA 실행 호스트는 아니다.
이미 다른 서비스가 사용하는 GPU라면 메모리와 GPU 번호를 확인한 뒤 실행한다.
이 가이드의 명령은 현재 저장소에만 적용한다. 시스템 패키지 설치나 Docker 재시작은 별도 승인 대상이다.

### 드라이버와 커널

호스트에서 다음 두 명령부터 확인한다.

```sh
uname -r
nvidia-smi
```

새 클라우드 이미지의 커널이 드라이버보다 앞서 나가면 NVIDIA 모듈의 DKMS 빌드가 실패할 수 있다.
드라이버 설치가 끝났다는 메시지만 보지 말고 `nvidia-smi`가 GPU를 반환하는지 확인한다.
실패하면 DKMS 빌드 로그와 [NVIDIA 지원 환경](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/introduction.html)을 대조한다.
GCP의 Ubuntu 24.04에서는 최신 커널을 무조건 올리기보다 `linux-gcp-lts-24.04` 같은
LTS 커널과 호환 드라이버 조합을 먼저 검토한다.
이 프로젝트는 기존 서버의 커널을 지우는 명령을 자동 실행하지 않는다.
다른 서비스가 쓰는 드라이버를 재설치하는 명령도 자동 실행하지 않는다.
커널 변경이 필요하면 새 커널의 부팅을 확인하기 전까지 기존 부팅 커널을 보존한다.

클라우드 GPU 이미지에는 드라이버와 Container Toolkit이 이미 설치된 경우가 있다.
`nvidia-smi`와 `nvidia-ctk --version`으로 기존 설치부터 확인한다.
이미지가 패키지 버전을 잠가 두었다면 잠금을 강제로 풀기보다 지원 범위와 Docker 런타임 설정을 먼저 확인한다.
정상 동작하는 도구를 새로 설치할 필요는 없다.

## 설정 준비

저장소 루트에서 실행한다.

```sh
python3 scripts/bootstrap.py --check
python3 scripts/bootstrap.py --write-env
```

첫 명령은 호스트 점검만 수행한다. 두 번째 명령은 토큰이 들어간 `.env`를 생성하며 기존 파일은 덮어쓰지 않는다.
`.env.example`의 한·영 주석을 보고 GPU 번호·포트·설정 경로를 확인한다. `.env`를 셸에서 `source`하지 않는다.
토큰이나 외부 API 키를 채팅에 붙여 넣거나 Git에 추가하지 않는다.

문서 경로와 데이터 연결은 `taco.toml`에 설정한다. 기본 예제는 공개된 데모 문서뿐이다.
자신의 디렉터리를 연결하려면 `sources.path`와 Compose의 읽기 전용 마운트를 함께 바꾼다.
REST 연결은 `examples/rest-source.toml`을 참고한다. 원본 시스템에 접근해도 되는 자료만 연결한다.

모델은 `models.toml`에서 임베딩·리랭커별로 고른다. [모델 연결](models.md)에 프리셋, 사용자 모델,
역할별 분리 실행과 교체 절차를 정리했다. 모델 설정과 GPU 자원 배정은 구분한다.

| 파일 | 설정할 내용 |
|---|---|
| `.env` | 비밀값, GPU 번호, 포트와 설정 파일 경로 |
| `models.toml` | 임베딩·리랭커 모델, 고정 리비전과 실행 옵션 |
| `taco.toml` | 소스, 컬렉션·장부, API 주소와 검색 옵션 |
| `examples/models/*.toml` | 선택적 생성 모델의 provider, 주소와 agent/rag 모드 |

처음에는 기본 모델과 데모 소스로 시작한다. 생성 모델이나 외부 API 키는 필수가 아니다.
자체 문서 연결과 모델 변경을 동시에 진행하지 않으면 실패한 위치를 구분하기 쉽다.

### 컨테이너의 파일 권한

기본 MCP·indexer 이미지는 root가 아닌 UID `10001`로 실행한다. 호스트 계정에서 읽히는 파일도
컨테이너에서는 권한이 없을 수 있다. `:ro` 마운트는 쓰기를 막을 뿐 읽기 권한을 부여하지 않는다.

- `taco.toml`, `models.toml`과 연결 문서는 UID `10001`이 읽을 수 있어야 하고 문서 디렉터리는 탐색 권한도 필요하다.
- 기본 `state` 볼륨은 이미지가 준비한다. 별도의 호스트 디렉터리를 연결한다면 해당 UID의 쓰기 권한을 확인한다.
- 공개 예제를 제한적인 `umask`로 압축 해제했다면 기본 예제 파일만 아래처럼 조정할 수 있다.

```sh
chmod 0644 taco.toml models.toml examples/knowledge/refunds.md
chmod 0755 examples/knowledge
```

실제 비공개 문서에는 이 예제를 일괄 적용하지 않는다. 필요한 UID에만 읽기·탐색 권한을 주는
소유권 설정이나 ACL을 사용한다. `.env`는 bootstrap이 만든 `0600`을 유지한다.
컨테이너는 비밀 파일을 직접 마운트하지 않고 Compose가 전달한 환경변수를 사용한다.
권한 오류를 피하려고 컨테이너를 root로 실행하거나 `chmod -R 777`을 적용하지 않는다.

Dockerfile은 BuildKit 캐시 마운트를 사용한다. `docker buildx version`이 실패하면
Docker 설치에 맞는 Buildx 플러그인을 준비한다. 다운로드 캐시는 실행 이미지와 분리하며
소스만 바꾼 빌드는 기존 의존성 레이어를 재사용한다.

## 기동과 확인

처음에는 이미지와 모델 가중치를 다운로드한다. 네트워크·디스크·GPU 사용을 승인받은 뒤 진행한다.

```sh
docker compose config --quiet
docker compose build mcp indexer inference
docker compose up -d qdrant inference mcp
docker compose run --rm --no-deps indexer models --model-config /config/models.toml
docker compose run --rm --no-deps indexer --config /config/taco.toml doctor
docker compose run --rm indexer --config /config/taco.toml sync
docker compose run --rm --no-deps indexer --config /config/taco.toml search 'refund'
docker compose run --rm --no-deps indexer --config /config/taco.toml search 'refund' --mode keyword
docker compose run --rm --no-deps indexer --config /config/taco.toml status
```

추론 서비스가 준비되기 전에는 모델 다운로드나 로딩을 기다린다.
검색 결과에서 `document_key`, `version`, `index`, 원문과 출처를 확인한다.
MCP 클라이언트에서 같은 `document_key`와 `version`으로 `get_document`도 호출한다.
이 단계까지 성공해야 연결과 색인을 함께 확인한 것이다. 상태 점검 성공만으로 검색 품질을 보장하지 않는다.

주기적 동기화가 필요하면 `docker compose --profile sync up -d indexer`를 실행한다.
한 프로세스가 수집과 색인을 순서대로 수행하며 다른 동기화 프로세스와 겹치면 쓰기 잠금을 획득하지 못한 쪽이 실패한다.
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
도구는 `search`, `keyword_search`, `deep_search`, `get_document`, `status`이며 쓰기 도구나 셸을 제공하지 않는다.
[검색 가이드](retrieval.md)에 도구별 범위와 한도를 정리했다.

Go stdio 서버를 직접 연결하려면 저장소에서 `make build` 후 `bin/mcp-stdio --config taco.toml`을 사용한다.
stdio에는 HTTP 접속 토큰이 필요하지 않지만 모델·Qdrant에 인증이 있으면 설정한 자격 증명은 전달해야 한다.
`uv run taco serve --transport stdio`도 같은 Go 바이너리로 전환한다.
wheel만 설치한 환경은 Go 바이너리를 따로 준비해 PATH에 두거나 `MCP_STDIO_BIN`, `MCP_HTTP_BIN`으로 지정한다.
HTTP·stdio 모두 같은 Go 검색 구현과 게시 장부를 사용한다.
네이티브 실행은 `.env`를 자동으로 읽지 않는다. 필요한 자격 증명은 실행 환경의 비밀 설정으로 전달한다.

현재 인증 단위는 배포 인스턴스 전체다. 접속자는 그 인스턴스의 모든 연결 자료를 읽을 수 있다.
서로 다른 접근 권한이 필요한 데이터는 별도 인스턴스로 분리한다. 원본 시스템의 문서별 ACL을 자동 복제하지 않는다.

## 문제가 생겼을 때

먼저 실행 상태와 최근 로그를 확인한다. 아래 명령은 컨테이너를 재시작하거나 데이터를 지우지 않는다.

```sh
docker compose ps
docker compose logs --tail 100 inference mcp
docker compose run --rm --no-deps indexer --config /config/taco.toml status
```

분리 실행 중이면 같은 Compose 파일 조합을 사용하고 `reranker` 로그도 확인한다.
외부 시스템 오류가 포함된 로그를 공유할 때는 본문과 인증 정보를 먼저 가린다.

- CUDA를 찾지 못하면 호스트 드라이버와 컨테이너 GPU 접근부터 확인한다. CPU로 전환하지 않는다.
- `PermissionError`로 기동에 실패하면 위 파일 권한 절에 따라 설정·문서 읽기와 상태 디렉터리 쓰기 권한을 확인한다.
- 모델 로딩에 실패하면 고정 리비전, 디스크, VRAM과 모델별 실행 요구사항을 확인한다.
- 인증 오류는 서버와 클라이언트의 토큰 설정을 대조한다. 토큰 자체를 로그에 출력하지 않는다.
- 색인 오류는 `status`의 마지막 시도·성공 시각, 체크포인트, 오류와 정리 대기 수를 확인한다.
- 키워드 색인이 없거나 모델 정보가 장부에 없으면 indexer에서 `sync`를 한 번 실행한다. 같은 모델이라면 기존 벡터는 유지한다.
- 임베딩 모델이나 분할 설정을 바꾼 경우 `docs/models.md`의 재색인 절차를 따른다.
- 실행 중 모델 정보가 사전 확인과 다르면 임베딩 응답을 거부한다. 로드밸런서 뒤의 모든 추론 서버가 같은 설정인지 먼저 확인한다.
- `taco models --model-config models.toml`은 설정 검사다. 실제 실행 모델은 컨테이너 `doctor`와 검색 응답에서 별도로 확인한다.

볼륨 삭제나 전체 초기화는 기본 복구 방법이 아니다. 원인을 확인하기 전에는 `down -v`를 실행하지 않는다.
외부 응답 본문에는 비밀값이 섞일 수 있어 공개 오류에는 예외 종류만 남긴다.
이 정보만으로 원인을 좁힐 수 없으면 해당 서비스의 접근 제한된 로그를 확인한다.

## 개발 환경

```sh
uv sync --group dev --python 3.12
make build
make test
make check
```

테스트는 실제 Go 서버·Python 색인기와 가짜 추론·로컬 Qdrant를 연결한다.
Go가 없으면 교차 언어 통합 테스트는 건너뛰므로 전체 계약 검증을 마친 것으로 보지 않는다.
외부 계정이나 실제 GPU 가중치 실행 결과로 해석하지 않는다.
실제 CUDA 검사는 [모델 연결 가이드](models.md#실제-gpu-추론-확인)에 별도로 설명했다.
