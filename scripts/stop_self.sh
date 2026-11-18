#!/usr/bin/env bash
# Stop this Runpod pod from the inside, so an unattended job cannot leave a
# GPU billing after it ends.
#   stop_self.sh                      stop now
#   stop_self.sh <log> <marker> <max hours> <grace minutes>
#       wait until <marker> appears in <log> (or <max hours> pass), wait the
#       grace period (time to copy results off), then stop.
# The pod id and its pod-scoped API key are in the container's environment.
set -uo pipefail
if [ $# -ge 4 ]; then
  deadline=$(( $(date +%s) + $3 * 3600 ))   # whole hours
  until grep -qE "$2" "$1" 2>/dev/null || [ "$(date +%s)" -ge "$deadline" ]; do sleep 30; done
  sleep $(( $4 * 60 ))
fi
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
curl -s -m 30 -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" \
  https://api.runpod.io/graphql
