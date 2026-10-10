# 데이터 연결과 확장

기본 연결은 폴더와 REST API다. 회사별 SDK나 인증 방식을 코어에 넣지 않고
새 어댑터를 추가한다. 자동 플러그인 탐색이나 임의 코드 다운로드는 제공하지 않는다.

| 위치 | 책임 |
|---|---|
| `core/contracts.py` | 문서·스냅샷·청크와 소스·추론·저장소의 공통 규격 |
| `core/service.py` | 게시 순서, 체크포인트, 검색 결과 검증 |
| `adapters/sources.py` | 폴더·REST 수집과 입력 제한 |
| `adapters/inference.py`, `adapters/qdrant.py` | 추론 HTTP와 벡터 저장소 연결 |
| `config.py` | 설정 검증과 어댑터 조립 |
| `agent/` | 검색과 분리된 선택형 답변 생성 |

## 기본 연결

폴더는 UTF-8 `.md`·`.txt`만 읽는다. 숨김 경로와 심볼릭 링크는 건너뛴다.
기본 제한은 문서 1,000개, 파일당 2 MB, 스냅샷 합계 20 MB다.
제한을 넘기면 일부 문서만 성공한 것으로 처리하지 않고 수집에 실패한다.
대규모 디렉터리는 먼저 의미 있는 소스 단위로 나눈다.

REST 예제는 고정 HTTPS 주소에 GET 요청을 보내며 아래 응답을 기대한다.

```json
{
  "items": [{"id": "policy-1", "text": "Policy text", "title": "Policy"}],
  "next_cursor": null
}
```

다음 페이지가 있으면 `next_cursor`에 문자열을 넣는다. 같은 주소에 `cursor` 쿼리를 붙여
요청하며 응답에 담긴 URL이나 리다이렉트는 따라가지 않는다.
기본 제한은 100페이지·문서 1,000개·응답당 4 MB·스냅샷 합계 20 MB다.
이 제한은 운영자의 실수나 예기치 않은 응답으로 한 번의 수집이 무한히 커지는 것을 막는다.

`examples/rest-source.toml`의 필드 매핑과 환경변수 이름을 수정한다.
필드 값은 문자열이어야 하며 `id`는 비어 있으면 안 된다.
REST API가 이 응답 형식이나 Bearer 인증을 제공하지 않으면 별도 소스 어댑터를 작성한다.

## 새 소스의 계약

소스는 고유한 `id`와 `scan(checkpoint)`를 제공한다.
반환값은 `Snapshot(documents, checkpoint, complete)`다.

- `Document.id`는 원본의 안정된 식별자다. 제목이나 조회 순서를 식별자로 쓰지 않는다.
- `Document.source_id`는 소스의 `id`와 같아야 한다. 소스 ID 변경은 새 소스로 취급한다.
- `text`에는 검색할 본문, 나머지 필드에는 제목·출처 URL·리비전·원본 갱신 시각을 담는다.
- `updated_at`을 알 수 없으면 비워 둔다. 조회 시각을 원본 수정 시각으로 꾸미지 않는다.
- 권한 오류·페이지 누락·타임아웃은 예외를 발생시키거나 `complete=False`로 반환한다.
- 실패를 빈 문서 목록으로 바꾸지 않는다. 완전한 빈 스냅샷은 기존 문서를 삭제할 수 있다.

현재 계약에서 `complete=True`는 해당 소스의 **전체 목록**을 뜻한다.
변경 문서만 반환하는 증분 API를 그대로 연결하면 안 된다.
그런 API는 어댑터 안에서 전체 목록을 복원하거나 별도의 증분 게시 계약을 먼저 구현해야 한다.
기본 소스도 매번 전체 목록을 읽는다. 체크포인트만 보고 수집을 생략하지 않는다.
변하지 않은 문서는 버전 비교 후 임베딩을 생략한다.

체크포인트는 게시와 정리가 끝난 뒤 코어가 저장한다. 어댑터에서 미리 갱신하지 않는다.
새 벡터를 저장하는 것과 검색에 새 버전을 게시하는 것도 별개다.
실패한 벡터 쓰기는 게시하지 않으며 이전 버전 정리는 재시도할 수 있게 장부에 남긴다.

새 어댑터를 만든 뒤 `config.py`의 `SourceConfig`와 조립 함수에 명시적으로 등록한다.
코어의 게시·검색 로직을 수정할 필요는 없다. 설정 파일에는 비밀값 대신 환경변수 이름을 적는다.
컨테이너의 데이터 마운트와 네트워크 권한도 함께 확인한다.

다음은 전체 목록을 읽는 어댑터의 최소 형태다. `client`의 구현과 인증은 어댑터가 맡는다.

```python
from taco_shell_rag.core.contracts import Document, Snapshot


class PolicySource:
    id = "policies"

    def __init__(self, client):
        self.client = client

    def scan(self, checkpoint):
        rows, cursor = self.client.read_all_pages()
        documents = [
            Document(
                id=row["id"],
                source_id=self.id,
                text=row["text"],
                title=row.get("title", ""),
                url=row.get("url", ""),
                updated_at=row.get("updated_at", ""),
            )
            for row in rows
        ]
        return Snapshot(documents, checkpoint=cursor, complete=True)
```

`read_all_pages()`는 설명용 인터페이스다. 페이지 일부만 받았거나 읽기 권한을 잃었을 때
정상 목록을 반환하지 않도록 구현하고 그 동작을 가짜 응답으로 검사한다.
이 코드를 붙여 넣는 것만으로 새 `type`이 설정에 등록되지는 않는다.

## 교체 후 확인

테스트에는 외부 계정 대신 가짜 응답을 사용한다.
정상 목록 외에도 중복 ID, 빈 전체 목록, 페이지 누락, 부분 쓰기 실패와 재시도를 검사한다.
`tests/test_sources.py`, `tests/test_publication_failures.py`가 기본 예제다.

추론·벡터 저장소를 바꿀 때도 같은 방식으로 공통 프로토콜을 구현하고 조립 위치만 바꾼다.
임베딩 모델이나 벡터 공간을 바꾸는 작업은 새 컬렉션·장부로 재색인한다.
문서별 접근 제어가 필요한 자료는 현재의 인스턴스 공용 인증 범위에 섞지 않는다.
