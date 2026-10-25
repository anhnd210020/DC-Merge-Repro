#!/usr/bin/env bash
# Exp 7 scheduler v3. Default threads (the verified setting). One job in lane setup at a time,
# at most 5 jobs, starts only with >= 6 GB free GPU memory, OOM -> requeue, any other failure -> no new starts.
source /workspace/ducanh/exp7_env.sh
cd /workspace/ducanh/DC-Merge
C1=dtd-eurosat-gtsrb-svhn; C2=eurosat-gtsrb-mnist-svhn; C3=eurosat-gtsrb-sun397-svhn
C4=stanford_cars-eurosat-gtsrb-mnist; C5=stanford_cars-eurosat-gtsrb-resisc45
C6=stanford_cars-eurosat-mnist-resisc45; C7=stanford_cars-eurosat-mnist-sun397
JOBS=("controls $C3" "controls $C1" "controls $C2"
      "method $C3" "method $C2" "method $C1"
      "entropy $C3" "entropy $C2" "entropy $C1"
      "controls $C4" "controls $C5" "controls $C6" "controls $C7"
      "wide $C3" "wide $C1" "wide $C2")
MAXJOBS=5; NEED=6000; SETUP_MAX=1200; stop=0; running=()
ts(){ date -u +%H:%M:%S; }
free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
logf(){ set -- $1; echo "$OUT-$1-$2.log"; }
reap(){
  local still=() r pid job start t0 rc
  for r in "${running[@]}"; do
    IFS='|' read -r pid job start t0 <<<"$r"
    if kill -0 "$pid" 2>/dev/null; then still+=("$r"); continue; fi
    wait "$pid"; rc=$?
    if [ $rc -eq 0 ]; then echo "$(ts) DONE $job"
    elif tail -n +$((start+1)) "$(logf "$job")" | grep -qi "out of memory"; then
      echo "$(ts) OOM $job -> requeued"; JOBS=("$job" "${JOBS[@]}")
    else echo "$(ts) FAILED rc=$rc $job (NOT OOM) -> no new starts; report this"; stop=1; fi
  done
  running=("${still[@]}")
}
setup_done(){   # job start_line t0 pid
  local ctx=${1#* } f
  kill -0 "$4" 2>/dev/null || return 0
  tail -n +$(($2+1)) "$(logf "$1")" 2>/dev/null | grep -q 'Evaluating CLIP head model' && return 0
  for f in "$OUT/$ctx"/val_cache.json "$OUT/$ctx"/test_cache.json "$OUT/$ctx"/proxy_cache_*.json; do
    [ -f "$f" ] && [ "$(stat -c %Y "$f")" -gt "$3" ] && return 0
  done
  return 1
}
echo "$(ts) scheduler v3 start; ${#JOBS[@]} jobs queued"
while :; do
  reap
  if [ ${#running[@]} -eq 0 ] && { [ ${#JOBS[@]} -eq 0 ] || [ $stop -eq 1 ]; }; then
    echo "$(ts) SCHEDULER END (left: ${JOBS[*]:-none})"; break; fi
  if [ $stop -eq 0 ] && [ ${#JOBS[@]} -gt 0 ] && [ ${#running[@]} -lt $MAXJOBS ] && [ "$(free_mb)" -ge $NEED ]; then
    job=${JOBS[0]}; JOBS=("${JOBS[@]:1}")
    lf=$(logf "$job"); start=$( { wc -l < "$lf"; } 2>/dev/null || echo 0); t0=$(date +%s)
    bash scripts/run_exp7.sh $job >/dev/null 2>&1 &
    pid=$!; running+=("$pid|$job|$start|$t0"); echo "$(ts) START $job (free GPU $(free_mb) MiB)"
    while ! setup_done "$job" "$start" "$t0" "$pid"; do
      sleep 20
      if [ $(( $(date +%s) - t0 )) -gt $SETUP_MAX ]; then echo "$(ts) WARN $job still in setup after ${SETUP_MAX}s; continuing"; break; fi
    done
    echo "$(ts) setup finished for $job after $(( $(date +%s) - t0 ))s"
    continue
  fi
  sleep 30
done
