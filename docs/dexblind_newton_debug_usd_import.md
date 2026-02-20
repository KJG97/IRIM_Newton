# Newton USD 모델 로드 경로 & 디버깅 위치

## 1. Isaac Lab 쪽 호출 경로 (우리 코드)

```
ManagerBasedEnv
  → scene.clone_environments(force_simple_replicate=True)
     → cloner.newton_replicate(...)   [source/isaaclab/isaaclab/cloner/cloner_utils.py]
```

**파일:** `source/isaaclab/isaaclab/cloner/cloner_utils.py`  
**함수:** `newton_replicate()` (대략 316~414행)

### 여기서 하는 일

1. **프로토타입 로드 (env_0 한 번만)**
   ```python
   for src_path in sources:  # e.g. ["/World/envs/env_0"]
       p = ModelBuilder(up_axis=up_axis)
       p.add_usd(stage, root_path=src_path, load_visual_shapes=True)  # ← Newton 라이브러리 호출
       protos[src_path] = p
   ```

2. **각 env 위치에 world 추가**
   ```python
   for col, env_id in enumerate(env_ids.tolist()):
       ...
       builder.add_world(
           protos[sources[row]],
           xform=wp.transform(positions[col].tolist(), quaternions[col].tolist()),  # ← positions 여기 들어감
       )
   ```

3. **빌더 등록**
   ```python
   NewtonManager.set_builder(builder)
   ```

---

## 2. 디버깅할 위치 (우선 순서)

### A. Isaac Lab – `newton_replicate()` 안 (cloner_utils.py)

- **확인할 것**
  - `positions`가 실제로 grid 간격인지  
    → `positions[col].tolist()` 직전에 `print(col, positions[col])` 또는 로그로 `_default_env_origins` 확인.
  - `sources`가 `["/World/envs/env_0"]` 하나만 오는지  
    → `force_simple_replicate` 분기일 때 맞는지 확인.

- **추가로 찍어볼 것**
  - 프로토타입 `p`에 body/geom이 몇 개인지, 이름이 뭔지  
    → `p.body_key`, `p.body_count` (또는 Newton API에 맞는 필드) 출력해서  
    “env 루트 1개 + table 1개 + robot 링크들”처럼 계층이 하나의 world로 묶여 있는지, 아니면 table이 따로 떠 있는지 확인.

**추천 브레이크/로그 위치**

- `source/isaaclab/isaaclab/cloner/cloner_utils.py`  
  - `p.add_usd(stage, root_path=src_path, ...)` 직후  
  - `builder.add_world(protos[sources[row]], xform=...)` 직전 (해당 루프 안)

**디버깅 코드 사용법**

- 같은 파일 상단에 `DEBUG_NEWTON_REPLICATE = True` 로 두면 다음이 로그로 출력됩니다.
  - `sources`, `env_ids`, `positions`, `quaternions`, `mapping` (add_world 루프 진입 전)
  - 프로토타입별 `body_count`, `shape_count`, `body_key`, `shape_key` 샘플 (`add_usd` 직후)
  - 매 `add_world` 호출 시 `col`, `env_id`, `position`, `quat`, `source`
- 로그는 `logging.getLogger("isaaclab.cloner.newton_replicate")` 로 남으므로,  
  `logging.basicConfig(level=logging.INFO)` 또는 해당 로거 레벨을 INFO로 설정하면 콘솔에서 확인할 수 있습니다.

---

### B. Newton 라이브러리 (외부 패키지)

- **패키지:**  
  `newton @ git+https://github.com/newton-physics/newton.git@beta-0.2.1`  
  (설치는 `source/isaaclab_newton/setup.py` 등에서 사용)

- **확인할 것**
  1. **`ModelBuilder.add_usd(stage, root_path=...)`**
     - `root_path="/World/envs/env_0"`일 때:
       - 이 Xform 아래의 **자식들(테이블, Robot)**을 어떻게 넣는지  
         (하나의 “world” body 아래에 자식으로 넣는지, 아니면 body를 평탄하게 잡고 월드 좌표로 넣는지).
     - static / collision-only geom(테이블)이:
       - “env_0 루트의 자식”으로 들어가는지,  
       - 별도 body로 들어가고 위치만 월드로 세팅되는지.

  2. **`ModelBuilder.add_world(proto_builder, xform=...)`**
     - `xform`이 **루트 body 하나**에만 적용되는지,  
       아니면 “이 world에 속한 모든 body/geom”에 적용되는지.
     - 테이블이 루트의 자식이 아니면, `xform`을 줘도 테이블 위치가 안 바뀌어서  
       num_env≥2일 때 Newton 시각화에서만 어긋나 보일 수 있음.

- **찾아볼 위치 (Newton 쪽 저장소 클론 후)**
  - `ModelBuilder`의 `add_usd`, `add_world` 구현  
    (Python 바인딩 + C++/실구현이 있다면, “USD에서 body/geom 만들 때 부모 관계와 좌표를 어떻게 세팅하는지” 위주로 보면 됨).

---

## 3. Newton 문서 참고 (USD 연동)

- [Newton – Custom attributes (USD)](https://newton-physics.github.io/newton/latest/concepts/custom_attributes.html)  
  → USD에서 모델/state 쪽 속성 어떻게 넘기는지.
- [Newton – Worlds](https://newton-physics.github.io/newton/latest/concepts/worlds.html)  
  → `add_world`와 “world” 개념이 문서에 나오면, xform이 어떤 단위에 적용되는지 확인.
- [Newton – Articulations](https://newton-physics.github.io/newton/latest/concepts/articulations.html)  
  → ArticulationView / root transform 등; 테이블은 articulation이 아니지만, “루트 기준 이동”이 어떻게 되는지 참고 가능.

---

## 4. 요약

| 단계 | 위치 | 확인할 내용 |
|------|------|-------------|
| 1 | `cloner_utils.newton_replicate()` | `positions`, `sources`, 프로토타입 body/geom 개수·이름 |
| 2 | Newton `ModelBuilder.add_usd(root_path=...)` | env_0 아래 테이블/Robot이 한 “world” 계층으로 들어가는지, static은 어떻게 처리하는지 |
| 3 | Newton `ModelBuilder.add_world(proto, xform=...)` | xform이 “world 전체(테이블 포함)”에 적용되는지, 루트만 옮겨지는지 |

**정리:**  
- “모델을 불러오는 방법”은 **Isaac Lab**에서는 `cloner_utils.newton_replicate()` → `ModelBuilder.add_usd` / `add_world` 호출까지가 전부이고,  
- **실제로 USD를 파싱해서 body/geom을 만드는 부분**은 **Newton 라이브러리** (`newton-physics/newton` 저장소)의 `add_usd` / `add_world` 구현을 봐야 한다.  
- 디버깅은 **먼저 cloner_utils에서 positions·프로토타입 구조 확인** → **이상하면 Newton 쪽 add_usd/add_world 동작**을 보면 된다.
