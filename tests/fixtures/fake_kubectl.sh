#!/usr/bin/env bash
set -euo pipefail

: "${FAKE_KUBECTL_LOG:?FAKE_KUBECTL_LOG is required}"
{
  printf 'kubectl'
  for argument in "$@"; do
    printf '\t%s' "$argument"
  done
  printf '\n'
} >>"$FAKE_KUBECTL_LOG"

case " $* " in
  *" jsonpath="*) printf '%s' 'fake-perfseer-pod' ;;
  *" apply "*) printf '%s\n' 'job.batch/fake configured' ;;
  *" get job "*) printf '%s\n' 'NAME STATUS COMPLETIONS'; printf '%s\n' 'fake Running 0/1' ;;
  *" get pod"*) printf '%s\n' 'NAME READY STATUS'; printf '%s\n' 'fake-perfseer-pod 1/1 Running' ;;
  *" describe pod "*) printf '%s\n' 'Name: fake-perfseer-pod' ;;
  *" logs "*) printf '%s\n' 'sanitized fixture log' ;;
  *" get events "*) printf '%s\n' 'LAST SEEN TYPE REASON'; printf '%s\n' '1s Normal Scheduled' ;;
  *" exec "*) printf '%s\n' 'sanitized process/output fixture' ;;
esac
