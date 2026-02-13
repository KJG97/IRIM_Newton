# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """
    ADR 난이도에 따라 초기값과 최종값 사이를 보간하는 함수.
    
    'data'의 임의로 중첩된 리스트/튜플 구조를 처리하며, 리프 노드에서 스칼라(int/float) 값을 보간합니다.
    """
    # 디바이스에서 난이도 비율 스칼라 가져오기
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        # 초기 단계에서는 난이도 비율이 0에 가까워 리소스 낭비이므로 변경 없음
        return mdp.modify_env_param.NO_CHANGE

    # 초기값/최종값을 텐서로 변환 (재귀 과정에서 분해됨)
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


# def _recurse(iv_elem, fv_elem, data_elem, frac):
#     # 시퀀스인 경우: 각 요소를 재귀적으로 처리하여 동일한 타입으로 재구성
#     if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
#         # 참고: 초기값과 최종값 요소가 data와 동일한 구조를 가진다고 가정
#         return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
#     # 리프 스칼라인 경우: 보간 수행
#     new_val = frac * (fv_elem - iv_elem) + iv_elem
#     if isinstance(data_elem, int):
#         return int(new_val.item())
#     else:
#         # float 또는 기타 숫자형으로 변환
#         return new_val.item()
def _recurse(iv_elem, fv_elem, data_elem, frac):
    # 시퀀스인 경우: 각 요소를 재귀적으로 처리하여 동일한 타입으로 재구성
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        # 참고: 초기값과 최종값 요소가 data와 동일한 구조를 가진다고 가정
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    # 리프 스칼라인 경우: 보간 수행
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val)  # .item() 제거 - new_val은 이미 Python 숫자 타입
    else:
        # float 또는 기타 숫자형으로 변환
        return float(new_val)  # .item() 제거하고 float()로 명시적 변환


def step_based_interpolate_fn(
    env: ManagerBasedRLEnv,
    env_id,
    data,
    initial_value,
    final_value,
    start_step: int,
    end_step: int,
    num_steps_per_env: int | None = None,
):
    """학습 스텝 기반 초기값에서 최종값으로의 보간 함수.

    학습 스텝 수에 따라 initial_value에서 final_value로 점진적으로 보간합니다.
    start_step과 end_step 사이에서 선형 보간을 수행합니다.

    `num_steps_per_env`가 제공되면, `start_step`과 `end_step`은 iteration 번호로 해석되며
    다음 공식으로 step 번호로 변환됩니다: step = iteration × (num_steps_per_env × num_envs)

    Args:
        env: 환경 객체
        env_id: 환경 ID (사용되지 않음, API 호환성을 위해 유지)
        data: 대상 속성의 현재 값
        initial_value: start_step 이전에 사용할 초기값
        final_value: end_step 이후에 사용할 최종값
        start_step: 보간이 시작되는 step 번호 (num_steps_per_env가 제공되면 iteration 번호)
        end_step: 보간이 완료되는 step 번호 (num_steps_per_env가 제공되면 iteration 번호)
        num_steps_per_env: iteration당 환경당 step 수 (RSL-RL용). 제공되면
            start_step과 end_step은 iteration 번호로 처리됩니다. 기본값: None

    Returns:
        initial_value와 final_value 사이의 보간된 값, 또는 start_step 이전이면 NO_CHANGE
    """
    current_step = env.common_step_counter

    # num_steps_per_env가 제공되면 iteration을 step으로 변환
    if num_steps_per_env is not None:
        nstep = num_steps_per_env # iteration당 총 step 수
        start_step_actual = start_step * nstep
        end_step_actual = end_step * nstep
    else:
        start_step_actual = start_step
        end_step_actual = end_step

    # start_step 이전: 초기값 유지 (불필요한 업데이트 방지를 위해 NO_CHANGE 반환)
    if current_step < start_step_actual:
        return mdp.modify_env_param.NO_CHANGE

    # end_step 이후: 최종값 사용
    if current_step >= end_step_actual:
        frac = 1.0
    else:
        # start_step과 end_step 사이에서 선형 보간
        frac = float(current_step - start_step_actual) / float(end_step_actual - start_step_actual)
        frac = max(0.0, min(1.0, frac))  # [0, 1] 범위로 제한

    # 텐서로 변환하고 동일한 재귀 로직을 사용하여 보간
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


def step_based_switch_fn(
    env: ManagerBasedRLEnv,
    env_id,
    data,
    final_value,
    start_step: int,
    num_steps_per_env: int | None = None,
):
    """특정 iteration 이후에 값을 즉시 전환하는 함수 (보간 없음).
    
    Blind Grasp을 위해 특권 정보를 완전히 제거할 때 사용합니다.
    예: 특정 iteration 이후에 observation function을 제로 패딩 함수로 전환.
    
    Args:
        env: 환경 객체
        env_id: 환경 ID (사용되지 않음, API 호환성을 위해 유지)
        data: 대상 속성의 현재 값
        final_value: start_step 이후에 사용할 최종값 (함수 객체 등)
        start_step: 전환이 일어나는 step 번호 (num_steps_per_env가 제공되면 iteration 번호)
        num_steps_per_env: iteration당 환경당 step 수 (RSL-RL용). 제공되면
            start_step은 iteration 번호로 처리됩니다. 기본값: None
    
    Returns:
        start_step 이후: final_value
        start_step 이전: NO_CHANGE
    """
    current_step = env.common_step_counter
    
    # num_steps_per_env가 제공되면 iteration을 step으로 변환
    if num_steps_per_env is not None:
        nstep = num_steps_per_env  # iteration당 총 step 수
        start_step_actual = start_step * nstep
    else:
        start_step_actual = start_step
    
    # start_step 이후: final_value로 전환
    if current_step >= start_step_actual:
        return final_value
    else:
        # start_step 이전: 변경 없음
        return mdp.modify_env_param.NO_CHANGE


class DifficultyScheduler(ManagerTermBase):
    """커리큘럼 학습을 위한 적응형 난이도 스케줄러.

    환경별 난이도 수준을 추적하고 작업 성능에 따라 조정합니다. 위치/방향 오차가 주어진 허용 오차보다
    낮아지면 난이도가 증가하고, 그렇지 않으면 감소합니다 (`promotion_only`가 설정되지 않은 경우).
    환경 전체의 정규화된 평균 난이도는 커리큘럼 보간에 사용하기 위해 `difficulty_frac`로 노출됩니다.

    Args:
        cfg: 스케줄러 매개변수를 지정하는 설정 객체
        env: 매니저 기반 RL 환경

    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        pos_tol: float = 0.1,
        rot_tol: float | None = None,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
    ):
        # asset: Articulation = env.scene[asset_cfg.name]
        # object: RigidObject = env.scene[object_cfg.name]
        # command = env.command_manager.get_command("object_pose")
        # des_pos_w, des_quat_w = combine_frame_transforms(
        #     asset.data.root_pos_w[env_ids], asset.data.root_quat_w[env_ids], command[env_ids, :3], command[env_ids, 3:7]
        # )
        # pos_err, rot_err = compute_pose_error(
        #     des_pos_w, des_quat_w, object.data.root_pos_w[env_ids], object.data.root_pos_w[env_ids]
        # )
        # pos_dist = torch.norm(pos_err, dim=1)
        # rot_dist = torch.norm(rot_err, dim=1)
        # move_up = (pos_dist < pos_tol) & (rot_dist < rot_tol) if rot_tol else pos_dist < pos_tol
        # demot = self.current_adr_difficulties[env_ids] if promotion_only else self.current_adr_difficulties[env_ids] - 1
        # self.current_adr_difficulties[env_ids] = torch.where(
        #     move_up,
        #     self.current_adr_difficulties[env_ids] + 1,
        #     demot,
        # ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac


class modify_term_cfg_with_logging(mdp.modify_term_cfg):
    """로깅을 지원하는 modify_term_cfg 래퍼 클래스.

    이 클래스는 mdp.modify_term_cfg를 상속받아서 __call__ 메서드를 오버라이드하여
    현재 값을 반환하도록 합니다. 이를 통해 wandb 등에서 커리큘럼 값의 변화를 추적할 수 있습니다.

    사용법:
        .. code-block:: python

            hammer_static_friction_range_curriculum = CurrTerm(
                func=mdp.modify_term_cfg_with_logging,
                params={
                    "address": "events.hammer_physics_material.params.static_friction_range",
                    "modify_fn": mdp.step_based_interpolate_fn,
                    "modify_params": {...},
                },
            )
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        address: str,
        modify_fn: callable,
        modify_params: dict | None = None,
    ):
        """매니저 term 설정을 수정하고 로깅을 위해 현재 값을 반환합니다.

        이 메서드는 부모 클래스의 __call__을 오버라이드하여 현재 값을 반환하도록 합니다.
        이를 통해 커리큘럼 매니저가 이 값을 로그에 기록할 수 있습니다.

        Args:
            env: 매니저 기반 RL 환경
            env_ids: 환경 ID 시퀀스
            address: 수정할 속성의 주소 (점으로 구분된 경로)
            modify_fn: 값을 수정하는 함수
            modify_params: modify_fn에 전달할 추가 매개변수

        Returns:
            수정된 값 (또는 수정되지 않은 경우 현재 값)
        """
        # getter와 setter 함수가 아직 컴파일되지 않았다면 컴파일
        if not self._get_fn:
            self._get_fn, self._set_fn = self._process_accessors(self._env, self._address)

        # None 타입 해결
        modify_params = {} if modify_params is None else modify_params

        # 대상 속성의 현재 값 가져오기
        data = self._get_fn()
        # 제공된 함수를 사용하여 값 수정
        new_val = modify_fn(self._env, env_ids, data, **modify_params)
        # 수정된 값을 대상 속성에 다시 설정
        # 참고: modify_fn이 NO_CHANGE 신호를 반환하면 self.set_fn을 호출하지 않음
        if new_val is not self.NO_CHANGE:
            self._set_fn(new_val)
            # 로깅을 위해 새 값 반환
            # 함수 객체인 경우: 로깅용 숫자 값 반환 (함수 이름 기반)
            if callable(new_val):
                # 함수 이름을 기반으로 로깅 값 결정
                func_name = getattr(new_val, "__name__", "unknown")
                return 1.0 if "zero" in func_name.lower() else 0.0
            # 리스트/튜플인 경우: max 절대값 반환
            if isinstance(new_val, (list, tuple)) and len(new_val) > 0:
                # 모든 요소가 숫자인지 확인
                if all(isinstance(x, (int, float)) for x in new_val):
                    # max 절대값 반환 (대칭 범위의 경우 양수 값, 비대칭 범위의 경우 max 값)
                    return max(abs(x) for x in new_val)
            return new_val
        else:
            # 로깅을 위해 현재 값 반환 (변경 사항 없음)
            # 함수 객체인 경우: 로깅용 숫자 값 반환
            if callable(data):
                # 함수 이름을 기반으로 로깅 값 결정
                func_name = getattr(data, "__name__", "unknown")
                return 1.0 if "zero" in func_name.lower() else 0.0
            # 리스트/튜플인 경우: max 절대값 반환
            if isinstance(data, (list, tuple)) and len(data) > 0:
                # 모든 요소가 숫자인지 확인
                if all(isinstance(x, (int, float)) for x in data):
                    # max 절대값 반환
                    return max(abs(x) for x in data)
            return data
