# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # ADR: 자동/적응형 도메인 랜덤화 (Automatic/Adaptive Domain Randomization)
    # 난이도 스케줄러: 학습 진행에 따라 자동으로 난이도를 조절
    adr = CurrTerm(
        func=mdp.DifficultyScheduler, params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10}
    )
