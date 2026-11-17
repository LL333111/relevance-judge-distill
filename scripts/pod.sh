#!/usr/bin/env bash
# Helpers for the GPU pod over direct SSH (public IP + TCP port).
#
#   POD_HOST=<ip> POD_PORT=<port> POD_KEY=~/.ssh/<key>
#   scripts/pod.sh push              # send the committed code to /workspace/relevance-judge-distill
#   scripts/pod.sh run <cmd>         # run a command in the repo root on the pod
#   scripts/pod.sh bg <log> <cmd>    # start a command detached, logging to logs/<log>
#   scripts/pod.sh pull <path>...    # fetch files/dirs under the repo root
set -euo pipefail
: "${POD_HOST:?set POD_HOST}" "${POD_PORT:?set POD_PORT}" "${POD_KEY:?set POD_KEY}"
REPO=/workspace/relevance-judge-distill
ENVS="export HF_HOME=/workspace/hf PIP_CACHE_DIR=/workspace/pip-cache PATH=/workspace/venv/bin:\$PATH"
pod() { ssh -i "$POD_KEY" -p "$POD_PORT" -o IdentitiesOnly=yes -o ServerAliveInterval=30 \
            -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
            "root@$POD_HOST" "$@"; }

cmd=$1; shift
case "$cmd" in
  push)
    git archive --format=tar.gz HEAD \
      | pod "mkdir -p $REPO/logs && tar xzf - -C $REPO && echo $(git rev-parse HEAD) > $REPO/.git_commit && echo PUSH_OK \$(ls $REPO | wc -l) entries" ;;
  run)
    pod "$ENVS; cd $REPO && $*" ;;
  bg)
    log=$1; shift
    pod "$ENVS; cd $REPO && nohup bash -c '$*' > logs/$log 2>&1 < /dev/null & echo started logs/$log" ;;
  pull)
    # Through a temporary file: piping ssh straight into tar hangs on Git Bash.
    tmp=$(mktemp)
    pod -n "cd $REPO && tar czf - $*" > "$tmp" && tar xzf "$tmp" && echo "pulled: $*"
    rm -f "$tmp" ;;
  *) echo "usage: pod.sh push|run|bg|pull" >&2; exit 2 ;;
esac
