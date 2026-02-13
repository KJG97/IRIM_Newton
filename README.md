<div align="center">

# Isaac Lab (Newton + ALLEX)

**Isaac Lab Newton 물리 엔진 기반 ALLEX 휴머노이드 Direct RL 환경 및 Joint Slider 제어**

[Overview](#-overview) •
[프로젝트 구조](#-프로젝트-구조-변경된-부분만) •
[변경 내용](#-변경-내용-요약) •
[실행 예시](#-실행-예시)

</div>

---

## 📖 Overview

이 저장소는 **Isaac Lab**의 **Newton 물리 엔진** 브랜치를 기반으로, **ALLEX 휴머노이드**용 커스텀 환경과 도구를 추가·수정한 포크입니다.

### 핵심 변경 사항

- **Direct RL 환경 (ALLEX)**  
  Newton joint equality로 mimic 관절 구속, driver 관절만 액션으로 제어
- **Joint Slider Agent**  
  PySide GUI로 Active(driver) 관절을 degree 슬라이더로 제어, 시뮬과 같은 프로세스에서 연동
- **Mimic 구속**  
  `equality_constraints`로 MuJoCo joint equality 주입, 액추에이터는 damping만 사용

---

## 📁 프로젝트 구조 (변경된 부분만)

아래는 **이 포크에서 추가·수정한 경로만** 나열한 트리입니다.

```
IsaacLab/
├── dvcc/
│   ├── 101_UTILS_Refactoring.md
│   ├── 102_UTILS_Commit_Push.md
│   └── 200_DEV_NewtonJointSlider.md
│
├── scripts/
│   └── environments/
│       └── joint_slider_agent.py
│
└── source/
    ├── isaaclab/
    │   └── isaaclab/
    │       ├── cloner/
    │       │   └── cloner_utils.py
    │       ├── scene/
    │       │   └── interactive_scene_cfg.py
    │       └── sim/
    │           └── _impl/
    │               ├── newton_manager.py
    │               └── newton_manager_cfg.py
    │
    ├── isaaclab_assets/
    │   ├── isaaclab_assets/
    │   │   └── robots/
    │   │       └── allex.py
    │   └── allex_usd/
    │       ├── ALLEX_newton.usd
    │       ├── ALLEX_newton_no_left.usd
    │       └── allex_model_mjcf_250903/
    │           └── allex_contact_sensor.xml
    │
    └── isaaclab_tasks/
        └── isaaclab_tasks/
            └── direct/
                └── allex/
                    ├── allex_env.py
                    └── allex_env_cfg.py
```

---

## 📋 변경 내용 요약

| 경로 | 역할 |
|------|------|
| **dvcc/** | 개발·운영 유틸 문서 (리팩터링, 커밋 규칙, Newton Joint Slider 기획) |
| **scripts/environments/joint_slider_agent.py** | PySide GUI로 Active(driver) 관절만 degree 슬라이더 제어. mimic는 equality 구속, 같은 프로세스에서 `env.step(actions)` 연동 |
| **isaaclab/cloner/cloner_utils.py** | `newton_replicate(..., equality_constraints=...)` 로 mimic joint용 MuJoCo joint equality 주입 |
| **isaaclab/scene/interactive_scene_cfg.py** | 클론 시 `newton_replicate_kwargs`(예: `equality_constraints`) 전달 지원 |
| **isaaclab/sim/_impl/newton_manager.py, _cfg** | Newton 시뮬레이션 스텝·substep·CUDA Graph 설정 |
| **isaaclab_assets/robots/allex.py** | ALLEX / ALLEX_NO_LEFT articulation 설정. mimic 구간 `ImplicitActuatorCfg` (stiffness=0, damping만) |
| **isaaclab_assets/allex_usd/** | ALLEX Newton용 USD·MJCF (equality 정의: `allex_contact_sensor.xml`) |
| **isaaclab_tasks/direct/allex/** | Direct RL env: `_apply_action`에서 driver만 target 설정, mimic는 Newton equality에 위임 (`use_newton_equality_for_mimic`) |

---

## 🚀 실행 예시

### Joint Slider (ALLEX No-Left)

```bash
./isaaclab.sh -p scripts/environments/joint_slider_agent.py --task Isaac-Allex-Direct-NoLeft-v0 --visualizer newton
```

- Active(driver) 관절만 슬라이더로 표시되며, mimic 관절은 equality로 자동 구속됩니다.

---

## 📚 References

- [Isaac Lab – Newton Physics Integration](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html)
