# allex_test.usd vs ALLEX_newton_no_left.usd 메시 스케일 검토

## 비교 결과 (현재 상태)

| 항목 | allex_test.usd | no_left.usd | 비고 |
|------|----------------|-------------|------|
| **metersPerUnit** | 1.0 | 1.0 | 동일 (미터) |
| **메시 extent** | (-231,-75,-77) ~ (76.6,107,75) | 동일 | 같은 지오메트리 |
| **로컬 스케일 ≠ (1,1,1) 인 prim 수** | 0 | 0 | 둘 다 전부 1.0 |
| **world_scale (메시까지)** | (1,1,1) | (1,1,1) | 동일 |

- no_left는 루트가 `/allex_contact_sensor`, allex_test는 `/ALLEX` 로 경로만 다름.
- **스케일/단위/지오메트리**는 두 파일 모두 동일하므로, **USD만 보면** 메시를 불러올 때 크기 차이가 나는 이유는 없음.

---

## “USD만 봐서는 알 수 없는” 이유

크기 차이가 **런타임(Newton/시각화)** 에서만 보인다면, 원인은 USD 바깥에 있을 수 있음.

1. **Newton 로더**
   - `newton/utils/import_usd.py` 는 `usd.get_scale(prim)` 으로 **로컬 스케일**을 읽어 `add_shape_mesh(..., scale=scale)` 등에 넘김.  
     → USD에 scale (1,1,1)이면 동일하게 로드되어야 함.
   - 다만 `cloner_utils.py` 주석: *“Newton's add_usd() does not apply USD xform scale when loading **mesh geometry**”*  
     → 메시 **버텍스**에는 부모 xform scale이 적용되지 않고, **shape 크기**만 scale을 쓰는 구현일 수 있음.  
     그러면 “지오메트리는 mm, scale 0.001로 m로 보이게” 하던 USD는 scale을 (1,1,1)로 바꾼 뒤 **버텍스가 mm 그대로**면 1000배 크게 보일 수 있음.

2. **스테이지 구성 차이**
   - AllexEnvCfg는 `allex_test.usd` 를, AllexEnvNoLeftCfg는 `ALLEX_newton_no_left.usd` 를 참조.
   - 같은 `newton_replicate(stage, sources=[...], ...)` 라도, **stage에 들어 있는 템플릿 prim**이 어느 USD에서 왔는지에 따라 로드되는 메시/스케일이 달라짐.
   - USD 파일 내용은 같아도, **어떤 prim_path / source 경로로** add_usd 되느냐에 따라 트리 구조가 다르면 scale이 다른 prim을 읽을 수 있음.

3. **정리**
   - **USD만** 열어서 보면 두 파일은 동일한 스케일·단위·지오메트리.
   - 그래도 시뮬/뷰어에서 크기 차이가 난다면, **로더가 어떤 prim을 어떻게 쓰는지**, **버텍스에 scale이 적용되는지** 를 런타임에서 확인해야 함.

---

## 이전에 있던 차이 (이미 수정됨)

- allex_test의 `.../visuals`, `.../collisions` 아래 메시들은 **instanceable** 이었고, 프로토타입 안에 **scale (0.001, 0.001, 0.001)** 이 있어 1000배 크게 보이는 문제가 있었음.
- `scripts/utils/fix_allex_test_usd_scales.py` 로 **instanceable 해제 + 해당 스케일을 (1,1,1)로 변경** 해 둔 상태임.

---

## 런타임에서 확인하는 방법

1. **Newton 로드 시 scale 로그**
   - `newton_replicate` 호출 직전/직후에, 해당 source 경로의 stage prim들을 순회해 `UsdGeom.Xformable(prim).GetLocalTransformation()` 또는 `get_scale(prim)` 을 출력.
   - no_left용 env와 allex_test용 env에서 **같은 상대 경로** (예: `Waist_Base/visuals/ALLEX_Base`) 의 scale이 둘 다 (1,1,1)인지 확인.

2. **shape_scale / 메시 크기**
   - Newton `add_usd` 반환값이나 builder 내부에 `path_shape_scale` 이 있으면, 동일 링크명에 대해 allex_test vs no_left 로드 시 값이 같은지 비교.

3. **메시 단위 가정**
   - extent가 약 300~400 이면 **mm** 가정 시 0.3~0.4 m로 정상 크기.
   - no_left가 “정상 크기”로 보인다면, no_left 메시는 이미 **m 단위로 베이크**됐거나, 로더가 해당 stage에서만 단위 보정을 할 가능성도 있음.
   - allex_test는 **버텍스가 mm 그대로**일 수 있으므로, **스케일을 USD에서 (1,1,1)로만 맞춰서는 부족**하고, **메시 지오메트리를 m 단위로 베이크**하는 방법 검토:  
     `scripts/tools/bake_scale_into_mesh_usd.py` (no_left용으로 있음) 를 allex_test에 적용해 보는 것.

---

## 그래도 크기 차이가 날 때 확인할 것

1. **실제로 쓰는 파일이 맞는지**  
   - allex.py 의 `_ALLEX_USD_PATH` / no_left용 경로가 각각 allex_test.usd, ALLEX_newton_no_left.usd 를 가리키는지 확인.

2. **스폰/클론 시 스케일**  
   - Newton replicate 나 클로너가 **위치/스케일**을 곱하는지, 두 태스크에서 동일한지 확인.

3. **다시 비교해 보기**  
   - `python scripts/utils/compare_allex_test_no_left_scale.py`  
   - 끝부분 요약에서 metersPerUnit, extent, “비(1,1,1) 로컬 스케일 개수” 가 위 표와 같으면 USD 쪽은 동일한 상태.

4. **스케일이 다시 꼈을 때**  
   - XML 등에서 USD를 다시 만들었다면, `fix_allex_test_usd_scales.py` 를 다시 실행해 0.001 스케일과 instanceable을 정리.

5. **메시 단위 통일**  
   - no_left가 “정상 크기”인데 allex_test만 크다면, allex_test 메시가 **mm**이고 로더가 scale을 버텍스에 안 곱할 수 있음.  
   - `scripts/tools/bake_scale_into_mesh_usd.py` 는 **scale이 (0.001,0.001,0.001)인 prim**을 찾아 그 하위 메시에 베이크함.  
     지금은 allex_test 스케일을 전부 (1,1,1)로 바꿔 둔 상태라서, 이 스크립트만으로는 동작하지 않음.  
     allex_test를 m 단위로 쓰려면 “모든 메시 버텍스에 0.001 곱하기” 같은 **전역 베이크** 스크립트를 별도로 쓰거나,  
     no_left를 만들 때 썼던 파이프라인(베이크 후 export)을 allex_test에도 적용하는 식으로 맞추는 게 필요함.
