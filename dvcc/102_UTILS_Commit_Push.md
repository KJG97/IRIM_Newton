# 03 — 커밋 메시지 규칙 (Conventional Commits)

**Conventional Commits**를 따른다. 메시지만 보고 **변경 종류**와 **범위(Scope)**를 바로 알 수 있게 쓴다.

---

## 1. 구조

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

---

## 2. 타입 (Type)

| 타입 | 의미 | 예시 |
|------|------|------|
| `feat` | 기능 추가 | `feat: add habilis brain inference logic` |
| `fix` | 버그 수정 | `fix: resolve ik solver convergence issue` |
| `docs` | 문서 수정 | `docs: update prd for gripper control` |
| `style` | 포맷팅만 (로직 변경 없음) | `style: linting via black` |
| `refactor` | 리팩터링 (기능 동일) | `refactor: optimize point cloud processing` |
| `test` | 테스트 추가/수정 | `test: add unit test for quaternion utility` |
| `chore` | 빌드·패키지 등 | `chore: update torch version in requirements` |

---

## 3. 작성 규칙

- 제목과 본문 사이 **한 줄 띄우기**
- 제목 **50자 이내**, **첫 글자 대문자**, **끝에 마침표 없음**
- 제목은 **명령문** (Fix, Add — Fixed, Added 아님)
- 본문은 **무엇을·왜** 위주 (어떻게는 코드에)
- 가능하면 **scope 명시** (예: `feat(isaac-sim): ...`)

---

## 4. 에이전트: 커밋 + push 절차

사용자가 **"커밋해 줘"** / **"커밋 + push 해 줘"**라고 하면 아래 순서로 끝까지 수행한다.

1. **변경 확인** — `git status`, `git diff --stat`으로 변경 파일 파악
2. **메시지 작성** — 위 §1~§3 규격·규칙에 맞춰 제목(+ 본문), 타입·scope 적용
3. **스테이징** — 관련 파일만 `git add` (지정 없으면 변경분만; 불필요 파일 제외)
4. **커밋** — `git commit -m "<제목>" -m "<본문>"` 또는 본문 여러 줄이면 `-m` 반복 / `git commit -F <파일>`
5. **확인** — `git log -1 --oneline` 또는 `git show --stat`으로 커밋 검증
6. **push**
   - `git remote -v`로 원격 확인 후, 추적 중이면 `git push`, 아니면 `git push -u origin <현재브랜치>`
   - **실패 시**: 원격이 앞서 있으면 에러 전달 + `git pull --rebase` / `git pull` 후 재시도 안내; 인증 실패는 사용자에게 맡김
   - **성공 시**: `git log -1 --oneline`과 함께 "커밋 및 push 완료" 요약

**요청 예시**  
- "지금 변경 사항 DDVC 규칙으로 커밋 메시지 작성해 줘. 그걸로 **커밋하고 push까지** 해 줘."  
- "**커밋 + push**까지 에이전트가 다 해 줘."

---

## 5. 팀 협업

연구실 단위에서는 **타입 목록만이라도 통일**하면 실험·버그 수정 이력 추적이 빨라진다.

---

## 6. 이미 dev/newton에서 개발한 뒤 → 본인 계정(fork)에 올리기

`origin`이 원본(isaac-sim/IsaacLab)인 상태에서, **지금 로컬의 dev/newton(이미 개발한 내용)** 을 **본인 GitHub 계정**에 올리는 절차.

### 6.1 한 번만 할 것

1. **GitHub에 본인 계정용 저장소 만들기**  
   - **Fork**: https://github.com/isaac-sim/IsaacLab → **Fork** (나중에 원본에 PR 보낼 때 유리).  
   - **새 저장소**: GitHub **New repository** → 이름 예: `IsaacLab` (원본에 PR 안 보낼 거면 이것만으로 충분).

2. **원본을 upstream으로 남기고, 본인 저장소를 push 대상(origin)으로 추가**  
   (이미 `origin` = isaac-sim 이므로, 원본 이름만 바꾸고 본인 repo를 origin으로 추가)

   ```bash
   git remote rename origin upstream
   git remote add origin git@github.com:본인아이디/IsaacLab.git
   git remote -v   # origin=본인 fork, upstream=isaac-sim 확인
   ```

   SSH 대신 HTTPS 쓰면: `https://github.com/본인아이디/IsaacLab.git`

### 6.2 지금 브랜치 올리기

- **커밋 안 한 변경이 있으면** 먼저 커밋할지 결정. (§4 규칙으로 커밋 후 push 가능.)
- **이미 커밋된 dev/newton** 그대로 올리기:

  ```bash
  git push -u origin dev/newton
  ```

이후에는 `git push` / `git pull` 시 기본이 **본인 fork(origin)** 이므로, 일상적으로는 `git push`만 해도 본인 계정에 반영된다.  
원본(isaac-sim) 쪽 업데이트는 `git fetch upstream` 후 `git merge upstream/dev/newton` (또는 rebase)로 따로 가져오면 된다.
