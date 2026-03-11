# Table + Hammer 환경 셋업 트러블슈팅

**날짜:** 2026-03-09
**환경:** IRIM_Newton (Isaac Lab Newton fork), Ubuntu 24.04, RTX 5080, micromamba

---

## 요약

Table + Hammer 물리 시뮬레이션(Newton/MuJoCo)을 실행하기 위해 발생한 문제들은 **전부 설치/환경 이슈**였다. 물리 엔진 자체에는 문제가 없었으며, `coacd`를 설치하여 collision mesh를 올바르게 생성하면 hammer가 table 위에 정상적으로 안착한다.

---

## 발생한 문제 및 해결

### 1. VS Code 터미널에서 `conda activate` 오류

- **증상:** 터미널 열 때 `source /home/hancheol/.local/bin/activate isaaclab` → `No such file or directory`
- **원인:** VS Code Python 확장이 micromamba 환경을 conda로 인식하고 자동 활성화 시도, 하지만 conda가 없음
- **해결:**
  ```json
  // ~/.config/Code/User/settings.json
  "python.condaPath": "/home/hancheol/.local/bin/micromamba",
  "python.defaultInterpreterPath": "/home/hancheol/micromamba/envs/isaaclab/bin/python"
  ```
  + `~/.local/bin/activate` shim 스크립트 생성:
  ```bash
  eval "$(/home/hancheol/.local/bin/micromamba shell activate "$1" --shell bash 2>/dev/null)"
  ```

### 2. `--visualizer newton` 인식 불가

- **증상:** `unrecognized arguments: --visualizer newton`
- **원인:** Python 환경에 `~/IsaacLab`(구 버전)의 `isaaclab` 패키지가 설치되어 있었음. `--visualizer` 인자는 IRIM_Newton fork에만 존재.
- **해결:**
  ```bash
  pip install --no-build-isolation --no-deps -e ~/IRIM_Newton/source/isaaclab
  ```

### 3. `No module named 'newton'`

- **증상:** `from newton import ... → ModuleNotFoundError`
- **원인:** `isaaclab_newton` 패키지와 그 의존성(`newton`, `mujoco`, `mujoco-warp`)이 미설치
- **해결:**
  ```bash
  pip install --no-deps --no-build-isolation -e ~/IRIM_Newton/source/isaaclab_newton
  pip install --find-links https://py.mujoco.org/ \
    "mujoco>=3.4.0.dev839962392" \
    "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp.git@e9a67538f2c14486121635074c5a5fd6ca55fa83" \
    "newton @ git+https://github.com/newton-physics/newton.git@beta-0.2.1"
  ```

### 4. `No module named 'pxr'`

- **증상:** `import isaaclab.sim` 시 `ModuleNotFoundError: No module named 'pxr'`
- **원인:** IRIM_Newton의 `AppLauncher`가 standalone 모드(SimulationApp 미생성)로 동작 → Omniverse의 `pxr` 라이브러리 미로딩
- **해결:** 환경변수로 Omniverse 모드 강제:
  ```bash
  LAUNCH_OV_APP=1 ./isaaclab.sh -p <script> --visualizer newton
  ```

### 5. Newton visualizer: `pyglet >= 2.0 required`

- **증상:** `OpenGLRenderer requires pyglet (version >= 2.0)`
- **원인:** pyglet 1.5 설치됨
- **해결:**
  ```bash
  pip install 'pyglet>=2.1.6'
  ```

### 6. Newton visualizer: `No module named 'PySide6'`

- **증상:** Newton visualizer 창 렌더링 실패
- **원인:** PySide6 미설치
- **해결:**
  ```bash
  pip install PySide6
  ```

### 7. Qt xcb 플랫폼 플러그인 크래시

- **증상:** `Could not load the Qt platform plugin "xcb"` → 프로세스 중단
- **원인:** `libxcb-cursor0` 시스템 라이브러리 미설치
- **해결:**
  ```bash
  sudo apt install -y libxcb-cursor0
  ```

### 8. Hammer가 table에 함몰/진동/드리프트

- **증상:** Hammer가 table 표면 아래로 관통(z≈0.869, table top=0.885), 좌우로 드리프트, 상하 진동 후 table에서 낙하
- **원인:** `coacd` 미설치 → collision mesh 생성 시 `coacd` 실패 → `convex_hull` 실패 → **bounding box fallback**. Bounding box는 실제 hammer 형상보다 훨씬 크고, 중심이 시각 메시와 어긋남
- **해결:**
  ```bash
  pip install coacd
  ```
- **결과:** coacd가 hammer를 6개의 convex hull로 분해 → z≈0.898(table 위)에 안정적으로 안착, 속도 0, 드리프트 없음

---

## 최종 실행 방법

```bash
# micromamba 환경 활성화
micromamba activate isaaclab

# Table + Hammer 실행 (gravity only)
LAUNCH_OV_APP=1 ./isaaclab.sh -p scripts/environments/table_hammer_test.py --visualizer newton

# 외력 테스트 (10N, 1초 간격)
LAUNCH_OV_APP=1 ./isaaclab.sh -p scripts/environments/table_hammer_test.py --visualizer newton --apply_force 10.0

# 강한 외력 (50N, 0.5초 간격)
LAUNCH_OV_APP=1 ./isaaclab.sh -p scripts/environments/table_hammer_test.py --visualizer newton --apply_force 50.0 --force_interval 0.5
```

---

## 결론

모든 문제는 **설치/환경 구성 이슈**였다. Newton 물리 엔진 자체는 정상이며, `coacd`를 포함한 의존성을 올바르게 설치하면 collision mesh가 정확하게 생성되어 hammer-table 접촉이 안정적으로 동작한다.
