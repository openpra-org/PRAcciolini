#!/bin/bash
set -a
source .env
set +a

mkdir -p /tmp/xla_cache && chown 1000:1000 /tmp/xla_cache

#python3 tests/grammar/canopy/model/test_construct_subgraph.py
#python3 pracciolini/grammar/canopy/module/broadcast_sampler.py
python3 pracciolini/translator/opsamef_canopy/tf_module.py
