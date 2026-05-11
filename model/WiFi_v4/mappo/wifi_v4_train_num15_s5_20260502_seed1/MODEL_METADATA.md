# WiFi_v3 Model Metadata

Saved at: 2026-05-02 09:00:49 UTC

## Run
- env_name: WiFi_v4
- algorithm_name: mappo
- experiment_name: wifi_v4_train_num15_s5_20260502
- seed: 1
- num_mld: 15
- num_sld: 5
- round_length: 500
- rounds_per_update: 4
- rollout_length: 2000

## Local Observation (Actor Input)
- shared_load: current MLD shared demand normalized by `round_length * num_links`
- n_sld_norm: normalized SLD count on the current link
- shared_fulfillment: current shared fulfillment of the MLD
- prev_action_self: this agent's previous action (`0=skip`, `1=transmit`)
- prev_action_peer: paired same-MLD agent's previous action
- link_onehot: link identity (`[1,0]` for 2.4 GHz, `[0,1]` for 5 GHz)

## Shared Observation (Critic Input)
- flattened local observations of all agents on the same link
- current average SLD success ratio on 2.4 GHz (`0` on 5 GHz critic input)
- same-link shared fulfillment vector
- link one-hot

## Reward
### Global Reward
- MLD success by top-urgency agent: `+1`
- MLD success by non-top agent: `urgency(success_agent)`
- SLD success: `0`
- collision: `-1`
- idle: `-c_idle`

### Local Reward
- shared queue already satisfied and transmit: `-2`
- shared queue already satisfied and skip: `0`
- no urgent agent on the link and transmit: `-2`
- no urgent agent on the link and skip: `0`
- top-urgency agent transmit: `+1`
- top-urgency agent skip: `-(1 + urgency)`
- non-top agent transmit: `-2`
- non-top agent skip: `+1`

### Sparse Reward
- applied only on the final step of the round
- only affects 2.4 GHz agents
- if SLD throughput is below threshold: penalize above-average 2.4 GHz participation
- otherwise: reward above-average skipping / yielding on 2.4 GHz

## Allocation Metric
- expected share: each MLD's round demand divided by total round demand
- actual share: each MLD's round success count divided by total round success
- allocation/match: `1 - mean(abs(expected_share - actual_share))`
