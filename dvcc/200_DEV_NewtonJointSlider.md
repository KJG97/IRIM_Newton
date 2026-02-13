# 200_DEV_NewtonJointSlider — Newton 관절 슬라이더 제어

## 1. 개요

- **목표**: `zero_agent.py` 기반으로 현재 로봇의 Joint state를 파싱하고, joint limit을 반영한 **Joint Slider**를 제공하여 원하는 관절을 직접 움직일 수 있게 한다.
- **방식**: **PySide/PyQt GUI를 시뮬레이터와 같은 프로세스**에서 실행. **Active(driver) 관절만** 슬라이더로 표시(degree), mimic/passive 제외. TCP 없이 매 step에서 목표(°→rad)를 읽어 `env.step(actions)`에 반영.
- **실행 예시**:
  ```bash
  ./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-NoLeft-v0 --visualizer newton
  ```

---

## 2. 가능성 검토

### 2.1 기술적 타당성

| 항목 | 내용 | 판단 |
|------|------|------|
| Joint state 파싱 | `Articulation.find_joints(".*")` → `(mask, names, indices)` / `robot.data.joint_pos` (Warp → `wp.to_torch`) | ✅ 기존 API로 가능 |
| Joint limit 획득 | `robot.data.joint_pos_limits_lower`, `joint_pos_limits_upper` (shape: `(num_envs, num_joints)`), `wp.to_torch`로 호스트에서 사용 | ✅ articulation_data에 이미 제공 |
| 액션 적용 | `AllexEnv._apply_action()`은 `current + scale * actions`로 target 계산 후 `set_joint_position_target(target_driver, joint_ids=...)` 호출 | ✅ 외부에서 받은 목표치를 actions로 주입하면 됨 |
| zero_agent 활용 | `actions = torch.zeros(...)` 대신 **외부에서 수신한 목표 joint position**(또는 오프셋)을 actions로 사용 | ✅ 구조 변경 최소 |
| PySide 외부 GUI | 시뮬레이터와 **별도 프로세스**로 실행 → Omniverse/Newton 런타임과 충돌 없음 | ✅ 권장 |
| 원격 제어 | ZMQ 또는 TCP socket으로 Sim(서버) ↔ GUI(클라이언트) 통신 | ✅ 일반적인 패턴으로 구현 가능 |

**결론**: **진행 가능**. 기존 Newton Articulation API와 `zero_agent` 루프만 확장하면 되고, GUI는 완전히 분리된 프로세스로 두면 된다.

### 2.2 제약·고려사항

- **Mimic joint**: `AllexEnvNoLeftCfg.use_newton_equality_for_mimic=True` 이면 driver만 target 설정하고 mimic는 equality로 따라감. 슬라이더는 **action space와 동일하게 driver 관절만** 두거나, 전체 DOF를 보여주되 실제 제어는 driver만 해도 됨.
- **첫 프레임**: Newton 모델이 빌드된 뒤에만 `find_joints` / `joint_pos_limits_*` 가 유효. `_ensure_joint_dof_idx()` 호출 후에 파싱해야 함.
- **단위**: 시뮬 내부는 **rad**. GUI 슬라이더는 rad 또는 degree 선택 가능하게 두는 것이 사용성에 유리.

---

## 3. zero_agent 활용 방안

- **진입점**: `scripts/environments/zero_agent.py` 와 동일한 구조로,
  - `parse_env_cfg` → `gym.make(task, cfg=env_cfg)` → `env.reset()` 후
  - `is_simulation_running` 루프 안에서 **한 번 step을 진행한 뒤** `env.unwrapped` 에서 `robot` 접근.
- **필요 확장**:
  1. **첫 step 이후** `env.unwrapped.robot`으로 joint names, indices, limits 수집.
  2. **통신 서버 시작** (예: ZMQ REP 소켓 bind 또는 TCP listen).
  3. 매 step에서:
     - (선택) 클라이언트로부터 **목표 joint position** 수신 (또는 “현재 유지” 플래그).
     - 수신한 값을 `actions`로 변환해 `env.step(actions)` 에 전달.
     - (선택) 현재 `joint_pos`를 클라이언트에 전송해 GUI 슬라이더 동기화.

- **actions 의미**: `AllexEnv`는 `target = current + scale * actions` 이므로,
  - **옵션 A**: GUI에서 “목표 절대 위치”를 보내면, 시뮬 쪽에서 `actions = (target - current) / scale` 로 변환.
  - **옵션 B**: GUI에서 “오프셋”만 보내고 `actions`로 그대로 사용.

---

## 4. Joint state / limit 파싱 (구현 참고)

- **관절 이름·인덱스**  
  `robot.find_joints(".*")` → `(mask, names, indices)`.  
  NoLeft 환경에서는 `env.unwrapped` 에서 한 번 step 후 `_ensure_joint_dof_idx()` 가 호출된 상태와 동일한 순서로 `names` / `indices` 사용 가능.

- **현재 위치**  
  `joint_pos = wp.to_torch(robot.data.joint_pos)[env_id, indices]` (해당 env 한정이면 `env_id=0`).

- **Limit**  
  - `lower = wp.to_torch(robot.data.joint_pos_limits_lower)[env_id, indices]`  
  - `upper = wp.to_torch(robot.data.joint_pos_limits_upper)[env_id, indices]`  
  → 슬라이더 min/max 또는 QDoubleSpinBox range로 사용.

- **Driver-only 제어**  
  NoLeft에서 equality 사용 시, `_apply_action`과 동일하게 mimic 인덱스를 제외한 driver만 목표로 설정하면 됨. GUI에서 “전체 31 DOF”를 보여주되, 서버에서 driver만 target으로 변환해도 됨.

---

## 5. 같은 프로세스 PySide GUI (TCP 없음)

- **단일 프로세스**
  - `isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-NoLeft-v0 --visualizer newton` 한 번만 실행.
  - 스크립트가 `QApplication` 생성 후 env 생성·reset·첫 step → joint spec 수집 → `JointSliderWindow` 생성·표시.
  - 메인 루프: `qt_app.processEvents()` → 창에서 `get_target_positions()` → `actions = (target - current) / scale` → `env.step(actions)`. 창을 닫으면 루프 종료.
  - **Active(driver) 관절만** 표시; mimic/passive 제외 (`ALLEX_MIMIC_SPEC` 기준). 관절별 **현재(°)** + **목표 QSlider (degree)** + limit 반영. 통신 없이 같은 프로세스에서 위젯 값 직접 읽음.


---

## 6. 실행 방법 요약

```bash
./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-NoLeft-v0 --visualizer newton
```

- 한 프로세스에서 시뮬 + PySide 창 실행. 창이 뜨면 Active 관절만 degree 슬라이더로 목표를 넣고, 매 step에서 그 값(°→rad)이 적용됨. 창을 닫으면 시뮬 종료.
- 의존성: `PySide6` (또는 `PyQt6`). 없으면 `pip install PySide6`.

---

## 7. 진행 단계 (작업 breakdown)

1. **Phase 1 — 시뮬 쪽 (zero_agent 확장)** ✅
   - [x] `scripts/environments/joint_slider_agent.py` 추가: zero_agent와 동일한 env 생성/reset/step 루프.
   - [x] 첫 step 이후 `robot.find_joints(".*")`, `joint_pos_limits_lower/upper` 로 names·indices·limits 수집.
   - [x] TCP 제거. 같은 프로세스 PySide 창: 첫 step 후 JointSliderWindow 생성·표시, 매 step `processEvents()` 후 창에서 목표 읽어 `env.step(actions)` (scale=0.5, clamp ±1).

2. **Phase 2 — 프로토콜**
   - 해당 없음 (같은 프로세스, TCP/원격 없음).

3. **Phase 3 — PySide GUI** ✅
   - [x] `joint_slider_agent.py` 내부에 `JointSliderWindow`: PySide6/PyQt6, **Active(driver) 관절만** QSlider (degree) + limit 반영.
   - [x] 현재(°) + 목표 슬라이더(°). 매 step에서 목표 읽어 적용. 창 닫으면 루프 종료.
   - [ ] (선택) "Sync from sim" 버튼.

4. **Phase 4 — 테스트·문서**
   - [ ] Isaac-Allex-Direct-NoLeft-v0 한 env로 슬라이더 동작 확인.
   - [ ] README 또는 본 문서에 실행 방법·옵션 정리.

---

## 8. 참고 코드 위치

- Env: `source/isaaclab_tasks/isaaclab_tasks/direct/allex/allex_env.py` — `_ensure_joint_dof_idx`, `_apply_action`, `robot.set_joint_position_target`.
- Config: `source/isaaclab_tasks/isaaclab_tasks/direct/allex/allex_env_cfg.py` — `AllexEnvNoLeftCfg`, `use_newton_equality_for_mimic`, action_space 31.
- Robot: `source/isaaclab_assets/isaaclab_assets/robots/allex.py` — `ALLEX_NO_LEFT_CFG`, driver/mimic joint 목록.
- Joint limits: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py` — `joint_pos_limits_lower`, `joint_pos_limits_upper`.
- Agent 스크립트: `scripts/environments/zero_agent.py` — env 생성, reset, step 루프.

---

이 문서를 기준으로 **zero_agent 기반 + joint limit 반영 + 같은 프로세스 PySide 슬라이더 제어**가 구현되어 있다.
