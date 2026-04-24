#!/usr/bin/env python
"""
WiFi v2 동기 TXOP 환경 MAPPO 학습 스크립트.

실행 예시
---------
python train_wifi_v2.py \
    --env_name WiFi_v2 \
    --algorithm_name mappo \
    --experiment_name test \
    --num_mld 3 \
    --num_sld 3 \
    --round_length 50 \
    --num_env_steps 2000000 \
    --episode_length 50 \
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
from onpolicy.envs.wifi_v2.wifi_env import WiFiEnvV2
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def make_env(all_args, seed_offset: int):
    def get_env_fn(rank: int):
        def init_env():
            env = WiFiEnvV2(
                num_mld=all_args.num_mld,
                num_sld=all_args.num_sld,
                round_length=all_args.round_length,
                mu_range=(all_args.mu_min, all_args.mu_max),
                eta=all_args.eta,
                zeta=all_args.zeta,
                r_sld=all_args.r_sld,
                c_idle=all_args.c_idle,
                theta_scale=all_args.theta_scale,
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


def parse_args(args, parser):
    # WiFi v2 전용 인수
    parser.add_argument('--num_mld', type=int, default=3,
                        help="MLD STA 수 (2.4 + 5 GHz)")
    parser.add_argument('--num_sld', type=int, default=3,
                        help="SLD STA 수 (2.4GHz only)")
    parser.add_argument('--round_length', type=int, default=50,
                        help="라운드 길이 T (TXOP 수)")
    parser.add_argument('--mu_min', type=float, default=0.01,
                        help="MLD 도착률 최솟값")
    parser.add_argument('--mu_max', type=float, default=0.1,
                        help="MLD 도착률 최댓값")
    parser.add_argument('--eta', type=float, default=1.0,
                        help="SLD 미달 시 과점유 MLD 페널티 크기")
    parser.add_argument('--zeta', type=float, default=1.0,
                        help="SLD 달성 시 양보 MLD 보상 크기")
    parser.add_argument('--r_sld', type=float, default=0.3,
                        help="2.4GHz link에서 SLD 성공 시 r_global")
    parser.add_argument('--c_idle', type=float, default=0.3,
                        help="idle TXOP에 대한 r_global penalty")
    parser.add_argument('--theta_scale', type=float, default=1.0,
                        help="Scale factor for the SLD protection threshold theta")
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # episode_length = round_length (1 episode = 1 round)
    all_args.episode_length = all_args.round_length

    # 알고리즘 설정
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError(f"Unknown algorithm: {all_args.algorithm_name}")

    # CUDA 설정
    if all_args.cuda and torch.cuda.is_available():
        print("GPU 사용")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("CPU 사용")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    repo_root = Path(__file__).resolve().parents[3]

    # 실행 디렉터리
    base_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
        / "results" / all_args.env_name / all_args.algorithm_name
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
        run_dir = base_dir / curr_run
        os.makedirs(str(run_dir))

    model_save_dir = (
        repo_root
        / "model"
        / all_args.env_name
        / all_args.algorithm_name
        / f"{all_args.experiment_name}_seed{all_args.seed}"
    )
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model checkpoints will be saved to: {model_save_dir}")

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-wifi-v2-{all_args.experiment_name}"
    )

    # 시드 고정
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 환경 생성
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    num_agents = len(envs.observation_space)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "save_dir": model_save_dir,
    }

    # Runner 실행
    from onpolicy.runner.shared.wifi_v2_runner import WiFiV2Runner as Runner
    runner = Runner(config)
    runner.run()

    # 정리
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
