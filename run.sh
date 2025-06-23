#!/bin/bash
set -a
source .env
set +a

mkdir -p ./xla/cache && chown -R 1000:1000 ./xla

#python3 tests/grammar/canopy/model/test_construct_subgraph.py
#python3 pracciolini/grammar/canopy/module/broadcast_sampler.py
#python3 pracciolini/translator/opsamef_canopy/tf_module.py
time python3 pracciolini/translator/opsamef_canopy/tf_mod.py
time python3 pracciolini/translator/opsamef_canopy/tf_mod.py
