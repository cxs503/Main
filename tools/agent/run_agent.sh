#!/bin/bash
# Keep-alive wrapper for the tensor-agent bridge daemon (crontab @reboot + */5).
# flock guarantees a single instance; cron invocations exit 0 while alive.
set -a; source "$(dirname "$0")/agent.env"; set +a
LOCK=/nfs/wangxi/tensor_eco/tensor-agent.lock
exec 9>"$LOCK"
flock -n 9 || exit 0
export HOME=/nfs/wangxi/tensor_eco/agent_work/home
export TMPDIR=/nfs/wangxi/tensor_eco/agent_work/tmp
mkdir -p "$HOME" "$TMPDIR"
exec /nfs/wangxi/venvs/tensor_agent/bin/python "$(dirname "$0")/tensor_agent.py" >> /nfs/wangxi/tensor_eco/agent_work/agent.log 2>&1
