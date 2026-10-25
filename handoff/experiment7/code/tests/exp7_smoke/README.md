# Experiment 7 CPU smoke test (no GPU, no real data, ~3 minutes)

Synthetic 12-block model; `tests/exp6_smoke/shim` is a numpy stand-in for the torch calls on this
code path. The `inputs` stage builds Experiment 3/5/6 outputs with the REAL scripts (run_oracle,
run_entropy, run_agreement, rebaseline_exp3, run_layer_oracle), then each Exp 7 subcommand runs
against them. Accuracies are meaningless; it checks plumbing, caching/resume, the must-hold checks
(expected 0.000 pp and 0 images) and the gate/verdict logic.

    W=/tmp/exp7_smoke; mkdir -p $W
    export PYTHONPATH=$PWD/tests/exp6_smoke/shim
    H="$PYTHON tests/exp7_smoke/harness7.py lane_harm vision_lora_merge $W"
    $H inputs
    for st in "replay four" "checks core" "gate core" "controls core" "wide core"; do
      $H $st > "$W/${st// /_}.log" 2>&1 && echo "OK $st" || echo "FAIL $st"; done
    grep -h "checks vs" $W/*.log          # every line must end in PASS
    unset PYTHONPATH

The synthetic gate usually says FAIL, so `method`/`entropy` refuse to run; that is the expected,
pre-registered behaviour.
