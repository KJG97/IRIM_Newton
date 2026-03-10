# Hammer Velocity Divergence 및 Clamping 해결

**날짜:** 2026-03-10
**환경:** IRIM_Newton (Isaac Lab Newton fork), Newton/MuJoCo backend

---

## 문제

Newton(MuJoCo) 물리 시뮬레이션에서 외력(force + torque)이 가해질 때, **속도가 무한대로 발산(NaN)**하여 시뮬레이션이 붕괴하는 현상이 발생한다.

- 특히 **회전 속도(angular velocity)**가 선형 속도보다 훨씬 빠르게 발산
- 외력 + 외부 토크를 동시에 가하면 발산이 가속됨
- 접촉(contact) 상태에서 토크가 가해지면 solver가 수렴하지 못하고 (`solver didn't converge!`) 속도가 폭주
- 속도가 한 번 임계값을 넘으면 다음 timestep에서 더 큰 접촉력이 생겨 **양성 피드백 루프** → NaN

---

## 원인 분석

Newton solver는 implicit integrator + iterative solver를 사용한다. 외부 교란이 크면:

1. 한 timestep에서 속도가 크게 증가
2. 다음 timestep에서 큰 속도로 인한 penetration 발생
3. Solver가 penetration을 해소하려고 큰 접촉력 생성
4. 큰 접촉력이 다시 속도를 증가시킴 (양성 피드백)
5. Solver iteration 내에서 수렴 실패 → NaN

이는 solver iteration 수를 늘려도 근본적으로 해결되지 않으며, **속도 자체를 물리적으로 합리적인 범위로 제한**하는 것이 효과적이다.

---

## 해결: `_pre_physics_step`에서 Velocity Clamping

물리 step 이전(`_pre_physics_step`)에 매 step마다 선형/각속도를 확인하고, 임계값을 초과하면 방향을 유지한 채 크기만 제한한다.

### 핵심 코드

```python
def _pre_physics_step(self, actions: torch.Tensor):
    hammer = self.scene.articulations["hammer"]

    # --- Velocity Clamping ---
    MAX_LIN_VEL = 10.0   # m/s
    MAX_ANG_VEL = 20.0   # rad/s

    lin_vel = wp.to_torch(hammer.data.root_lin_vel_w)
    ang_vel = wp.to_torch(hammer.data.root_ang_vel_w)
    lin_speed = lin_vel.norm(dim=-1, keepdim=True)
    ang_speed = ang_vel.norm(dim=-1, keepdim=True)

    need_clamp = (lin_speed > MAX_LIN_VEL).any() or (ang_speed > MAX_ANG_VEL).any()
    if need_clamp:
        # 방향 유지, 크기만 제한 (v * max / |v|, |v| >= max 보장)
        clamped_lin = lin_vel * (MAX_LIN_VEL / lin_speed.clamp(min=MAX_LIN_VEL))
        clamped_ang = ang_vel * (MAX_ANG_VEL / ang_speed.clamp(min=MAX_ANG_VEL))

        # root state 재구성: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        root_pos = wp.to_torch(hammer.data.root_pos_w)
        root_quat = wp.to_torch(hammer.data.root_quat_w)
        new_state = torch.cat([root_pos, root_quat, clamped_lin, clamped_ang], dim=-1)
        hammer.write_root_state_to_sim(new_state)
```

### 동작 원리

1. **매 physics step 직전** 현재 속도 읽기
2. 선형 또는 각속도가 임계값 초과 시 **방향 보존 clamping** 적용
3. Clamped 속도를 포함한 전체 root state를 시뮬레이션에 다시 쓰기
4. 이후 physics step은 제한된 속도에서 시작 → 양성 피드백 루프 차단

### 왜 `_pre_physics_step`인가?

- **physics step 이전**에 속도를 제한해야 solver 입력이 안정적
- `_post_physics_step`이나 reward 계산 시점에서는 이미 solver가 발산한 후
- `_pre_physics_step`은 action 적용과 같은 시점으로, 물리 엔진에 상태를 주입하기에 적합

---

## 검증: 비교 영상

두 단계로 구성된 테스트로 clamping 효과를 검증:

| 단계 | 시간 | 외력 | 목적 |
|------|------|------|------|
| Normal | 0-3초 | 30N 수직 lift/drop, 토크 없음 | 일상적 조작 시뮬레이션 |
| Extreme (clamp ON) | 3-8초 | 20N 랜덤 3D + 10Nm 랜덤 토크 | 안정성 검증 |
| Extreme (clamp OFF) | 3-6초 | 20N 랜덤 3D + 10Nm 랜덤 토크 | 발산 유도 (3초면 충분) |

### 결과

| | Normal 단계 | Extreme 단계 |
|---|---|---|
| **Clamping OFF** (`hammer_no_clamp.mp4`) | 정상 lift/drop (z≈1.07) | **즉시 NaN 발산** |
| **Clamping ON** (`hammer_with_clamp.mp4`) | 정상 lift/drop (z≈1.07) | 격렬하지만 안정 (vel≈10 m/s) |

- Normal 단계에서는 **두 영상이 동일하게** 보임 (clamping이 정상 동작에 영향 없음)
- Extreme 단계에서 clamping 유무에 따라 **명확한 차이** 발생

### 녹화 스크립트

```bash
LAUNCH_OV_APP=1 ./isaaclab.sh -p scripts/environments/table_hammer_record.py --visualizer newton --output_dir dvcc/
```

---

## 적용 시 고려사항

### 임계값 선정

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| `MAX_LIN_VEL` | 10.0 m/s | 로봇 조작 환경에서 물체가 10 m/s 이상 이동할 일 없음 |
| `MAX_ANG_VEL` | 20.0 rad/s | ~3 rev/s, 일반적인 manipulation 범위 초과 |

환경에 따라 조정 가능. 너무 낮으면 정상 동작을 방해하고, 너무 높으면 발산 방지 효과 감소.

### 제한사항

- `write_root_state_to_sim`은 **root body**의 속도만 변경 (articulated body의 joint velocity는 별도 처리 필요)
- NaN이 한 번 발생하면 `write_root_state_to_sim`으로도 복구 불가 → clamping은 **예방적** 조치
- Clamping은 에너지 보존을 깨뜨리므로 물리적 정확성이 중요한 시나리오에서는 주의

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `scripts/environments/table_hammer_record.py` | 비교 영상 녹화 스크립트 (2단계 테스트) |
| `scripts/environments/table_hammer_test.py` | 인터랙티브 테스트 환경 (clamping 상시 적용) |
| `dvcc/hammer_no_clamp.mp4` | Clamping OFF 영상 |
| `dvcc/hammer_with_clamp.mp4` | Clamping ON 영상 |
