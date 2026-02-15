# source/isaaclab_assets/isaaclab_assets — Change log

In this fork only **robots/allex.py** was added and later modified. It defines two articulation configs: **ALLEX_CFG** (full body, 60 DOF, `allex_test.usd`) and **ALLEX_NO_LEFT_CFG** (31 DOF, `ALLEX_newton_no_left.usd`), with mimic joints configured so they do not fight Newton/MuJoCo equality constraints.

---

## 1. robots/allex.py

### 1.1 USD paths (lines 16–18)

- **Initial (commit 297caefa):**  
  `_ALLEX_USD_PATH` pointed to `allex_usd/ALLEX_newton.usd`, `_ALLEX_NO_LEFT_USD_PATH` to `allex_usd/ALLEX_newton_no_left.usd`.
- **Modified (commit b9e4f579):**  
  **Line 17:** `_ALLEX_USD_PATH` changed from `"allex_usd" / "ALLEX_newton.usd"` to `"allex_usd" / "allex_test.usd"`.
- **Reason:** Full-body env uses `allex_test.usd` (same structure as ALLEX_newton but with mesh scale/instanceable fixes and meter-unit geometry for Newton). No-Left path is unchanged.

### 1.2 ALLEX_CFG — articulation_props (lines 28–31)

- **Original:** `enabled_self_collisions=True`.
- **Modified (commit b9e4f579):** **Line 29:** `enabled_self_collisions=False`.
- **Reason:** Reduce risk of self-collision issues with the full 60-DOF robot in early testing; can be turned back on if needed.

### 1.3 ALLEX_CFG — init_state pos (lines 102–104)

- **Original:** `pos=(0.15, 0.0, 0.3)`.
- **Modified (commit b9e4f579):** **Line 103:** `pos=(0.0, 0.0, 1.0)`.
- **Reason:** Align full-body spawn height and XY with No-Left and with teleop/slider usage (e.g. 1 m above ground).

### 1.4 ALLEX_CFG — body actuator effort_limit_sim (lines 108–111)

- **Original:** `effort_limit_sim=10000.0`.
- **Modified (commit b9e4f579):** **Line 109:** `effort_limit_sim=300.0`.
- **Reason:** More realistic limit for body (waist) actuators; 10000 was overly high.

### 1.5 ALLEX_CFG — passive (mimic) actuator (lines 115–127)

- **Original:**  
  `effort_limit_sim=0.1`, `velocity_limit_sim=0.1`, `stiffness=1.0`, `damping=0.1`.
- **Modified (commit be87bb0f, then kept in b9e4f579):**  
  **Lines 120–125:**  
  `effort_limit_sim=1.0`, `velocity_limit_sim=0.0`, `stiffness=0.0`, `damping=1.0`.
- **Reason:** Mimic joints are driven by Newton/MuJoCo equality only. The actuator must not apply a position target (stiffness=0) so it does not fight the constraint; damping only helps numerical stability. effort/velocity limits are set to minimal non-zero where required by the stack.

### 1.6 ALLEX_NO_LEFT_CFG — articulation_props (lines 268–271)

- **Modified (commit be87bb0f31e):** **Line 269:** `enabled_self_collisions` from `False` to `True`.
- **Reason:** No-Left has fewer links; enabling self-collisions is acceptable and useful for contact-rich tasks.

### 1.7 ALLEX_NO_LEFT_CFG — init_state pos (lines 308–311)

- **Original:** `pos=(0.0, 0.0, 0.0)`.
- **Modified (commit be87bb0f):** **Line 309:** `pos=(0.0, 0.0, 1.0)`.
- **Reason:** Spawn robot 1 m above ground so it does not start inside the ground plane.

### 1.8 ALLEX_NO_LEFT_CFG — mimic actuator (lines 346–361)

- **Original:** Comment described setting poly(driver) target from env with high stiffness/damping; `stiffness=1000.0`, `damping=100.0`, `effort_limit_sim=1000.0`, `velocity_limit_sim=10.0`.
- **Modified (commit be87bb0f):**  
  - Comment: “Mimic joints: MuJoCo equality가 자세 강제. 액추에이터는 damping만 두어 constraint와 견제하지 않음.”
  - **Lines 357–360:** `stiffness=0.0`, `damping=1.0` (position target 없음; equality가 강제. damping만 사용).
- **Modified again (commit b9e4f579):**  
  **Lines 357–360:** `effort_limit_sim=1.0`, `velocity_limit_sim=0.0`, `stiffness=0.0`, `damping=10.0`.
- **Reason:** Same as full-body: mimic motion is enforced only by Newton equality; actuator uses stiffness=0 and damping so it does not fight the constraint. Damping=10 for No-Left gives a bit more stability; effort/velocity set to minimal where required.

---

## 2. Summary table

| Config | Location (approx) | Change |
|--------|-------------------|--------|
| ALLEX_CFG | Line 17 | USD path: `ALLEX_newton.usd` → `allex_test.usd` |
| ALLEX_CFG | Line 29 | `enabled_self_collisions` True → False |
| ALLEX_CFG | Line 103 | `pos` (0.15, 0, 0.3) → (0, 0, 1.0) |
| ALLEX_CFG | Line 109 | body `effort_limit_sim` 10000 → 300 |
| ALLEX_CFG | Lines 120–125 | passive actuator: stiffness=0, damping=1, effort=1, velocity=0 |
| ALLEX_NO_LEFT_CFG | Line 269 | `enabled_self_collisions` False → True |
| ALLEX_NO_LEFT_CFG | Line 309 | `pos` (0,0,0) → (0,0,1.0) |
| ALLEX_NO_LEFT_CFG | Lines 346–360 | mimic: comment + stiffness=0, damping=10, effort=1, velocity=0 |

No other files under `source/isaaclab_assets/isaaclab_assets/` were added or modified in this fork.
