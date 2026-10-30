# 설치

Linux GPU 호스트에서 서비스를 실행하고, 예제 문서를 검색한 뒤 MCP 클라이언트를 연결하는 순서로 진행합니다.
첫 설치에서는 기본 모델과 데모 문서를 사용합니다. 이 상태에서 검색까지 확인한 다음 자신의 데이터나 다른 모델로 바꾸면
문제가 생겼을 때 어느 변경이 원인인지 찾기 쉽습니다. 생성 모델과 외부 API 키는 첫 검색에 필요하지 않습니다.

## 실행 환경

임베딩과 리랭킹은 Linux의 NVIDIA GPU에서 실행합니다. 호스트에 NVIDIA 드라이버,
Docker Compose·Buildx, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)을 준비합니다.
GPU가 없는 환경에서 CPU 추론으로 대신 실행하는 기능은 없습니다. Mac은 MCP 클라이언트나 계약 테스트용으로 사용합니다.

이미 다른 서비스가 사용하는 GPU라면 GPU 번호와 남은 메모리부터 확인합니다.
필요한 메모리는 모델과 입력 길이에 따라 다릅니다. 이 가이드의 명령은 현재 저장소를 대상으로 하며,
시스템 패키지를 설치하거나 Docker를 재시작하는 작업은 별도 승인을 받아 진행합니다.

### 드라이버 확인

호스트에서 아래 명령을 실행합니다. 설치 완료 메시지뿐 아니라 `nvidia-smi`가 실제 GPU를 반환하는지 확인합니다.

```sh
uname -r
nvidia-smi
```

드라이버가 동작하지 않으면 DKMS 빌드 로그와 [NVIDIA 지원 환경](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/introduction.html)을
대조합니다. 설치한 드라이버가 새 클라우드 이미지의 커널을 지원하지 않아 모듈 빌드에 실패하는 경우가 있습니다.
배포판이 지원하는 LTS 커널과 호환 드라이버 조합을 먼저 살펴봅니다.

GPU 이미지에 드라이버와 Toolkit이 이미 포함되어 있다면 `nvidia-ctk --version`도 확인해 기존 설치를 사용합니다.
패키지 버전이 잠겨 있을 때는 강제로 잠금을 풀기보다 이미지의 지원 범위와 Docker 런타임 설정을 확인합니다.
정상 동작하는 패키지를 다시 설치할 필요는 없습니다.

설치 과정에서 기존 커널을 삭제하거나 다른 서비스가 쓰는 드라이버를 자동 재설치하지 않습니다.
커널 변경이 필요해도 새 커널로 부팅되는 것을 확인할 때까지 기존 부팅 커널을 남겨 둡니다.

## 설정 파일 준비

저장소 루트에서 다음 명령을 실행합니다.

```sh
python3 scripts/bootstrap.py --check
python3 scripts/bootstrap.py --write-env
```

`--check`는 호스트 환경을 점검합니다. `--write-env`는 접속 토큰이 들어 있는 `.env`를 만들고,
이미 파일이 있으면 덮어쓰지 않습니다. GPU 번호와 포트는 `.env.example`의 한·영 주석을 보고 맞춥니다.

| 파일 | 정할 내용 |
|---|---|
| `.env` | 비밀값, GPU 번호, 사용할 포트, 설정 파일 경로 |
| `models.toml` | 임베딩·리랭커 모델, 모델 리비전과 실행 옵션 |
| `taco.toml` | 수집할 소스, Qdrant 컬렉션, 문서 정보를 저장할 SQLite 파일, API 주소와 검색 옵션 |
| `examples/models/*.toml` | 답변 생성 모델의 provider, 주소, agent/rag 모드 |

모델 종류와 GPU 배정은 따로 설정합니다. 모델을 바꾸거나 역할별로 나누어 실행할 때는
[모델 연결 가이드](models.md)를 참고합니다.

`.env`는 셸에서 `source`하지 않습니다. 토큰과 외부 API 키는 채팅이나 Git에 넣지 않고 로컬의 비밀 설정으로 관리합니다.
Compose는 이 파일에서 읽은 환경변수를 컨테이너에 전달하며, 비밀 파일 자체를 컨테이너에 마운트하지 않습니다.
파일 권한은 bootstrap이 설정한 `0600`을 유지합니다.

기본 `taco.toml`은 공개 데모 문서만 읽습니다. 자신의 폴더를 연결할 때는 `sources.path`와 Compose의
읽기 전용 마운트를 함께 바꿉니다. REST API는 `examples/rest-source.toml`에서 시작합니다.
원본 시스템에서 읽을 권한이 있는 자료만 연결합니다.

### 컨테이너의 파일 권한

기본 MCP·indexer 컨테이너는 UID `10001`로 실행합니다. 호스트 사용자에게 읽기 권한이 있어도 이 UID는
파일을 읽지 못할 수 있습니다. `:ro` 마운트는 쓰기를 막는 옵션이며 읽기 권한을 새로 부여하지는 않습니다.

| 대상 | 필요한 권한 |
|---|---|
| `taco.toml`, `models.toml`, 연결할 문서 | UID `10001`의 읽기 권한 |
| 문서가 들어 있는 디렉터리 | UID `10001`의 탐색 권한 |
| 상태를 저장할 디렉터리 | UID `10001`의 쓰기 권한 |

기본 `state` 볼륨은 이미지가 준비합니다. 호스트 디렉터리로 바꿨다면 쓰기 권한을 직접 맞춥니다.
제한적인 `umask`로 공개 예제를 압축 해제했을 때는 예제 파일에 한해 아래처럼 조정할 수 있습니다.

```sh
chmod 0644 taco.toml models.toml examples/knowledge/refunds.md
chmod 0755 examples/knowledge
```

비공개 문서에는 위 권한을 일괄 적용하지 않고, 소유권이나 ACL로 필요한 UID에만 읽기·탐색을 허용합니다.
권한 문제를 피하려고 컨테이너를 root로 실행하거나 `chmod -R 777`을 사용하지 않습니다.

## 서비스 실행과 첫 검색

첫 실행에서는 이미지와 모델 가중치를 다운로드합니다. 네트워크·디스크·GPU 사용 승인을 받은 뒤 실행합니다.

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

추론 서비스는 모델 다운로드와 로딩을 마친 뒤 사용할 수 있습니다. CUDA 패키지 다운로드가 끝나도
설치와 이미지 저장이 이어질 수 있으므로 로그의 현재 단계와 디스크 여유를 함께 봅니다.

Dockerfile은 BuildKit 캐시를 사용합니다. `docker buildx version`이 실패하면 Docker 설치에 맞는 Buildx 플러그인을
준비합니다. 다운로드 캐시는 실행 이미지와 별도이며 소스만 바꿔 빌드할 때는 기존 의존성 레이어를 재사용합니다.
캐시, 실행 이미지, 모델 가중치가 각각 디스크 공간을 차지하는 점도 고려합니다.

검색이 성공하면 결과의 `document_key`, `version`, `index`, 원문과 출처를 봅니다.
이후 [MCP 클라이언트를 연결](#mcp-클라이언트-연결)하고 같은 문서 키와 버전으로 `get_document`를 호출해 원문도 읽어 봅니다.
검색과 원문 조회까지 확인해야 색인한 자료를 실제로 사용할 수 있는지 알 수 있습니다.
상태 점검이 성공했다는 것만으로 검색 품질까지 확인된 것은 아닙니다.

Qdrant 이미지는 [v1.19.0](https://github.com/qdrant/qdrant/releases/tag/v1.19.0)으로 고정되어 있습니다.
저장소나 클라이언트 버전을 변경할 때는 호환 범위와 실제 API 호출도 함께 확인합니다.

### 주기적으로 동기화하기

`docker compose --profile sync up -d indexer`를 실행하면 한 프로세스가 수집과 색인을 순서대로 반복합니다.
다른 동기화 프로세스와 겹칠 경우에는 쓰기 잠금을 얻지 못한 쪽이 실패합니다.

수집에 실패하면 기존에 검색에 사용하던 버전을 유지하고, 문서 삭제는 원본의 전체 목록을 읽은 뒤 반영합니다.
추론 서버나 저장소가 일시적으로 중단되면 오류 종류를 기록하고 다음 주기에 다시 시도합니다.
도구 호출이나 한 번만 실행한 `sync`가 실패한 결과를 성공으로 바꾸지는 않습니다.

## MCP 클라이언트 연결

HTTP 주소는 `http://127.0.0.1:8080/mcp`입니다. 요청에 `Authorization: Bearer <private token>` 헤더를 넣습니다.
실제 토큰은 클라이언트의 비밀 설정에 저장합니다.

| 클라이언트 | 설정 |
|---|---|
| Codex | HTTP MCP 서버의 `bearer_token_env_var`를 `MCP_AUTH_TOKEN`으로 지정합니다. |
| Claude Code | HTTP MCP 서버 주소와 인증 헤더를 설정합니다. |

원격 GPU 호스트는 로컬에서 `ssh -L 8080:127.0.0.1:8080 <gpu-host>`로 터널을 연 뒤 같은 주소에 연결합니다.
기본 Compose는 Qdrant와 추론 API를 호스트에 공개하지 않습니다. 컨테이너가 `0.0.0.0`에서 수신하더라도
MCP의 Host·Origin 검사는 그대로 적용됩니다.

전용 도메인의 역방향 프록시를 사용할 때는 TLS와 접근 권한을 확인한 뒤 `taco.toml`의 `allowed_hosts`,
필요한 `allowed_origins`를 지정합니다. 도구는 `search`, `keyword_search`, `deep_search`, `get_document`,
`status`이며 파일 수정이나 셸 실행 기능은 없습니다. 도구별 사용 범위는 [검색 가이드](retrieval.md)를 참고합니다.

### stdio로 연결하기

소스에서 `make build`를 실행한 뒤 `bin/mcp-stdio --config taco.toml`을 클라이언트에 등록합니다.
`uv run taco serve --transport stdio`도 같은 Go 바이너리를 실행합니다.
HTTP와 stdio는 같은 검색 코드와 문서 DB를 사용합니다.

stdio에는 HTTP 접속 토큰이 필요하지 않습니다. 다만 추론 서버와 Qdrant가 인증을 요구한다면 그 인증 정보는
실행 환경에 전달해야 합니다. 네이티브 실행은 `.env`를 자동으로 읽지 않습니다.
wheel만 설치했다면 Go 바이너리를 따로 준비해 PATH에 넣거나 `MCP_STDIO_BIN`, `MCP_HTTP_BIN`으로 경로를 지정합니다.

### 연결한 자료의 접근 권한

한 인스턴스에 접속한 사용자는 그곳에 연결된 모든 소스를 읽을 수 있습니다.
원본 시스템의 문서별 ACL을 자동으로 복제하지 않으므로 접근 권한이 다른 데이터는 별도 인스턴스로 나눕니다.

## 문제가 생겼을 때

먼저 컨테이너 상태, 최근 로그, 마지막 색인 결과를 봅니다. 아래 명령은 재시작이나 데이터 삭제 없이 상태를 조회합니다.

```sh
docker compose ps
docker compose logs --tail 100 inference mcp
docker compose run --rm --no-deps indexer --config /config/taco.toml status
```

분리 실행을 사용한다면 기동할 때와 같은 Compose 파일 조합으로 조회하고 `reranker` 로그도 확인합니다.
외부 응답 본문이나 인증 정보가 들어 있는 로그는 가린 뒤 공유합니다.

| 증상 | 확인할 곳 |
|---|---|
| CUDA를 찾지 못함 | 호스트 드라이버와 컨테이너의 GPU 접근을 확인합니다. CPU로 전환하지 않습니다. |
| `PermissionError`로 기동 실패 | 설정·문서 읽기 권한과 상태 디렉터리 쓰기 권한을 확인합니다. |
| 모델 로딩 실패 | 고정 리비전, 디스크, VRAM, 모델별 실행 요구사항을 봅니다. |
| 인증 오류 | 서버와 클라이언트의 토큰 설정을 대조합니다. 토큰 값은 로그에 출력하지 않습니다. |
| 색인 실패 | `status`에서 마지막 시도·성공 시각, 체크포인트, 오류, 정리 대기 수를 봅니다. |
| 키워드 색인이나 문서 DB의 모델 정보가 없음 | indexer에서 `sync`를 한 번 실행합니다. 같은 모델이면 기존 벡터를 유지합니다. |
| 임베딩 모델·분할 설정 변경 후 문제 | [모델 교체 절차](models.md#임베딩-교체)에 따라 재색인 여부를 확인합니다. |
| 사전 확인과 실제 실행 모델 정보가 다름 | 임베딩 응답을 거부하는 경우입니다. 로드밸런서 뒤의 추론 서버들이 같은 설정인지 봅니다. |

`taco models --model-config models.toml`은 설정 파일을 읽는 명령입니다. 실제 컨테이너에서 사용한 모델은
`doctor`와 검색 응답에 나온 정보로 확인합니다.

원인을 확인하기 전에는 `down -v`로 볼륨을 삭제하지 않습니다. 공개 오류에는 비밀값을 노출하지 않도록 예외 종류만 남깁니다.
그 정보만으로 원인을 찾기 어렵다면 해당 서비스의 접근이 제한된 로그를 확인합니다.

## 소스에서 개발하기

MCP와 검색 서버는 Go, 색인과 GPU 추론은 Python으로 작성되어 있습니다.
코드를 수정하거나 직접 빌드하려면 Go 1.25 이상과 Python 3.11~3.13을 준비합니다. 아래 예제는 Python 3.12를 선택합니다.
Docker 빌드는 Go 바이너리를 함께 준비하고 Python 기본 이미지도 3.12를 사용합니다.
직접 설치할 때도 `uv sync --python 3.12`처럼 버전을 지정합니다.
고정된 Torch 2.8 CUDA 패키지는 Python 3.14용이 없어 현재 구성으로 설치할 수 없습니다.

```sh
uv sync --group dev --python 3.12
make build
make test
make check
```

기본 테스트는 실제 Go 서버·Python 색인기와 가짜 추론·로컬 Qdrant를 연결합니다.
Go가 없으면 두 언어를 연결하는 테스트를 건너뛰므로 전체 검사를 마친 것으로 볼 수 없습니다.
이 검사는 외부 계정이나 실제 모델 가중치를 사용한 GPU 실행과는 별개입니다.
CUDA에서 직접 확인하는 방법은 [모델 연결 가이드](models.md#실제-gpu-추론-확인)에 있습니다.
