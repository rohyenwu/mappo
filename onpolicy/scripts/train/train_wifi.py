#!/usr/bin/env python
"""
WiFi 다중링크 공존 환경 MAPPO 학습 스크립트.

실행 예시
---------
python train_wifi.py \
    --env_name WiFi \
    --algorithm_name mappo \
    --experiment_name test \
    --num_mld_a 2 \
    --num_mld_b 2 \
    --num_sld 2 \
    --num_env_steps 2000000 \
    --episode_length 200 \
    --n_rollout_threads 1 \
    --hidden_size 64 \
    --use_centralized_V \
    --seed 1
"""
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config
from onpolicy.envs.wifi.WiFi_Env import WiFiEnv
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


# ──────────────────────────────────────────────────────────────────────────────
# 환경 생성 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def make_env(all_args, seed_offset: int):
    """단일 WiFiEnv 팩토리 함수 반환."""
    def get_env_fn(rank: int):
        def init_env():
            env = WiFiEnv(
                num_mld_a=all_args.num_mld_a,
                num_mld_b=all_args.num_mld_b,
                num_sld_per_link=all_args.num_sld,
                use_ind_reward=all_args.use_ind_reward,
                use_w_in_astar=all_args.use_w_in_astar,
            )
            env.seed(seed_offset + rank * 1000)
            return env
        return init_env
    return get_env_fn


def make_train_env(all_args):
    fns = [make_env(all_args, all_args.seed)(i)
           for i in range(all_args.n_rollout_threads)]
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv(fns)
    return ShareSubprocVecEnv(fns)


def make_eval_env(all_args):
    fns = [make_env(all_args, all_args.seed * 50000)(i)
           for i in range(all_args.n_eval_rollout_threads)]
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv(fns)
    return ShareSubprocVecEnv(fns)


# ──────────────────────────────────────────────────────────────────────────────
# WiFi 전용 인수 추가
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(args, parser):
    parser.add_argument(
        '--num_mld_a', type=int, default=2,
        help="MLD-A STA 수  (링크: 2.4 + 5 GHz)",
    )
    parser.add_argument(
        '--num_mld_b', type=int, default=2,
        help="MLD-B STA 수  (링크: 2.4 + 5 + 6 GHz)",
    )
    parser.add_argument(
        '--num_sld', type=int, default=2,
        help="링크당 SLD STA 수  (CSMA/CA)",
    )
    parser.add_argument(
        '--use_ind_reward', action='store_true', default=False,
        help="개인 보상(r_ind) 사용 여부. 미지정 시 전체 보상(r_global)만 사용.",
    )
    parser.add_argument(
        '--use_w_in_astar', action='store_true', default=False,
        help="A* 계산에 W(대기 슬롯) 포함 여부. 미지정 시 h, retry만 사용.",
    )
    return parser.parse_known_args(args)[0]


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    parser   = get_config()
    all_args = parse_args(args, parser)

    # ── 알고리즘 설정 ──────────────────────────────────────────────────────
    if all_args.algorithm_name == "rmappo":
        print("rmappo: recurrent policy ON")
        all_args.use_recurrent_policy      = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("mappo: recurrent policy OFF")
        all_args.use_recurrent_policy      = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("ippo: centralized V OFF")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError(f"Unknown algorithm: {all_args.algorithm_name}")

    # ── CUDA 설정 ──────────────────────────────────────────────────────────
    if all_args.cuda and torch.cuda.is_available():
        print("GPU 사용")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark    = False
            torch.backends.cudnn.deterministic = True
    else:
        print("CPU 사용")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # ── 실행 디렉터리 ──────────────────────────────────────────────────────
    base_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
        / "results"
        / all_args.env_name
        / "wifi"
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not base_dir.exists():
        os.makedirs(str(base_dir))

    if all_args.use_wandb:
        import wandb
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
            group="wifi",
            dir=str(base_dir),
            job_type="training",
            reinit=True,
        )
        run_dir = Path(wandb.run.dir)
    else:
        exst = [
            int(str(f.name).split('run')[1])
            for f in base_dir.iterdir()
            if str(f.name).startswith('run')
        ] if base_dir.exists() else []
        curr_run = f"run{max(exst) + 1}" if exst else "run1"
        run_dir  = base_dir / curr_run
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-wifi-{all_args.experiment_name}"
        f"@{getattr(all_args, 'user_name', 'user')}"
    )

    # ── 시드 고정 ──────────────────────────────────────────────────────────
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # ── 환경 생성 ──────────────────────────────────────────────────────────
    envs      = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    num_agents = len(envs.observation_space)

    config = {
        "all_args":  all_args,
        "envs":      envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device":    device,
        "run_dir":   run_dir,
    }

    # ── Runner 실행 ────────────────────────────────────────────────────────
    from onpolicy.runner.shared.wifi_runner import WiFiRunner as Runner
    runner = Runner(config)
    runner.run()

    # ── 정리 ───────────────────────────────────────────────────────────────
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir) + '/summary.json'
        )
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
