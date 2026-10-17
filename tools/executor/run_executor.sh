#!/bin/bash
# Keep-alive wrapper for the tensor executor (crontab @reboot + */5).
# flock guarantees a single instance; cron invocations exit 0 while alive.
LOCK=/nfs/wangxi/tensor_eco/tensor-executor.lock
exec 9>"$LOCK"
flock -n 9 || exit 0
cd "$(dirname "$0")"
exec /nfs/wangxi/venvs/tensor_eco/bin/python tensor_executor.py >> /nfs/wangxi/tensor_eco/executor-runs/daemon.log 2>&1
