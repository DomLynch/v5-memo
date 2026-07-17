#!/bin/sh
set -eu

unit_dir=${SYSTEMD_UNIT_DIR:-/etc/systemd/system}
config_dir=${V5_MEMO_CONFIG_DIR:-/etc/v5-memo}
deploy_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
route=${V5_MEMO_PORTFOLIO_SEARCH_ROUTE:-}
allow_dedicated=${V5_MEMO_ALLOW_DEDICATED_FULLRAW:-0}
dropin=zzzzz-v5-portfolio-fullraw-route.conf
dedicated_profile=v5-portfolio-publish-fullraw.conf
shared_profile=v5-portfolio-shared-fullraw.conf
shared_env=v5-memo-portfolio-shared-fullraw.env
dedicated_env=v5-memo-publish-fullraw.env
mount_unit=v5-memo-publish-fullraw-fts-mount.service
search_unit=v5-memo-publish-fullraw-search.service
owned_dropin=zzzzzzzz-v5-publish-fullraw-owned.conf
owned_profile=v5-memo-publish-fullraw-owned.conf
shared_sidecar_profile=v5-memo-publish-fullraw-shared.conf
lock_path=${V5_MEMO_PORTFOLIO_LOCK_PATH:-/run/v5-memo-portfolio.lock}
publish_mount=${V5_MEMO_PUBLISH_MOUNT_PATH:-/var/lib/v5-memo/v5-publish-fullraw-fts-remote}
publish_catalog=${V5_MEMO_PUBLISH_CATALOG_PATH:-/var/lib/v5-memo/v5-isolated-fullraw-shard-catalog.json}
dedicated_marker=$config_dir/allow-dedicated-fullraw

# The marker is the durable operator opt-in. Explicit route variables still
# win, so rollback can always force the shared route.
if [ -z "$route" ]; then
    if [ -f "$dedicated_marker" ]; then
        route=dedicated
        allow_dedicated=1
    else
        route=shared
    fi
fi

case "$route" in
    dedicated | shared) ;;
    *)
        echo "V5_MEMO_PORTFOLIO_SEARCH_ROUTE must be dedicated or shared" >&2
        exit 2
        ;;
esac
if [ "$route" = dedicated ] && [ "$allow_dedicated" != 1 ]; then
    echo "dedicated V5 fullraw requires V5_MEMO_ALLOW_DEDICATED_FULLRAW=1" >&2
    exit 2
fi
if [ "$route" = dedicated ] && [ ! -f "$dedicated_marker" ]; then
    echo "dedicated V5 fullraw requires $dedicated_marker" >&2
    exit 2
fi

install -d -m 0755 "$config_dir" "$unit_dir"
install -m 0644 \
    "$deploy_dir/$shared_env" \
    "$config_dir/portfolio-shared-fullraw.env"
install -m 0644 \
    "$deploy_dir/$dedicated_env" \
    "$config_dir/publish-fullraw.env"

for unit in "$mount_unit" "$search_unit"
do
    install -m 0644 \
        "$deploy_dir/$unit" \
        "$unit_dir/$unit"
done

install_sidecar_profile() {
    selected_profile=$1
    for unit in "$mount_unit" "$search_unit"
    do
        install -d -m 0755 "$unit_dir/$unit.d"
        install -m 0644 \
            "$deploy_dir/$selected_profile" \
            "$unit_dir/$unit.d/$owned_dropin" || return 1
        cmp -s \
            "$deploy_dir/$selected_profile" \
            "$unit_dir/$unit.d/$owned_dropin" || return 1
    done
}
if [ "$route" = dedicated ]; then
    install_sidecar_profile "$owned_profile"
else
    install_sidecar_profile "$shared_sidecar_profile"
fi

portfolio_units="v5-memo-portfolio-prepare.service v5-memo-portfolio-catchup.service v5-memo-portfolio-publish.service"
portfolio_timers="v5-memo-portfolio-prepare.timer v5-memo-portfolio-catchup.timer v5-memo-portfolio-publish.timer"
for unit in $portfolio_units; do
    install -d -m 0755 "$unit_dir/$unit.d"
    install -m 0644 \
        "$deploy_dir/$unit" \
        "$unit_dir/$unit"
done

systemctl daemon-reload
install_profile() {
    selected_profile=$1
    for portfolio_unit in $portfolio_units; do
        install -m 0644 \
            "$deploy_dir/$selected_profile" \
            "$unit_dir/$portfolio_unit.d/$dropin" || return 1
        cmp -s \
            "$deploy_dir/$selected_profile" \
            "$unit_dir/$portfolio_unit.d/$dropin" || return 1
    done
    systemctl daemon-reload || return 1
}

rollback_dedicated() {
    rollback_failed=0
    install_profile "$shared_profile" || rollback_failed=1
    install_sidecar_profile "$shared_sidecar_profile" || rollback_failed=1
    systemctl daemon-reload || rollback_failed=1
    systemctl disable --now "$search_unit" "$mount_unit" || rollback_failed=1
    if systemctl is-active --quiet "$search_unit" || systemctl is-active --quiet "$mount_unit"; then
        rollback_failed=1
    fi
    if systemctl is-enabled --quiet "$search_unit" || systemctl is-enabled --quiet "$mount_unit"; then
        rollback_failed=1
    fi
    if [ "$rollback_failed" -ne 0 ]; then
        echo "V5 publish fullraw rollback failed" >&2
        return 1
    fi
}

active_timers=
restore_timers() {
    for active_timer in $active_timers; do
        systemctl start "$active_timer"
    done
}

cleanup_on_exit() {
    rc=$?
    trap - 0
    if [ "$rc" -ne 0 ] && [ "$route" = dedicated ]; then
        rollback_dedicated || rc=1
    fi
    restore_timers || rc=1
    exit "$rc"
}
trap cleanup_on_exit 0

for timer_unit in $portfolio_timers; do
    if systemctl is-active --quiet "$timer_unit"; then
        active_timers="$active_timers $timer_unit"
        systemctl stop "$timer_unit"
    fi
done

exec 9>"$lock_path"
if ! flock -w 900 9; then
    echo "V5 portfolio lock did not drain" >&2
    exit 1
fi
# Any service still active now was waiting behind fd 9 with the old route.
systemctl stop $portfolio_units

if [ "$route" = dedicated ]; then
    systemctl enable "$mount_unit" "$search_unit"
    systemctl restart "$mount_unit"
    mounted=0
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        if mountpoint -q "$publish_mount"; then
            mounted=1
            break
        fi
        sleep 2
    done
    if [ "$mounted" -ne 1 ]; then
        echo "V5 publish fullraw mount did not become ready" >&2
        exit 1
    fi
    if ! jq -e '.entries | length == 1525' "$publish_catalog" >/dev/null 2>&1; then
        echo "V5 publish fullraw catalog is incomplete" >&2
        exit 1
    fi
    first_catalog_path=$(jq -r '.entries[0].path // ""' "$publish_catalog")
    first_batch=$(basename "$(dirname "$first_catalog_path")")
    first_shard=$(basename "$first_catalog_path")
    if [ -z "$first_catalog_path" ] || [ ! -f "$publish_mount/$first_batch/$first_shard" ]; then
        echo "V5 publish fullraw mounted corpus probe failed" >&2
        exit 1
    fi
    systemctl restart "$search_unit"
    token=$(awk -F= '$1 == "RESEARKA_FULLRAW_INDEX_TOKEN" {print substr($0, index($0, "=") + 1); exit}' "$config_dir/env")
    if [ -z "$token" ]; then
        echo "missing RESEARKA_FULLRAW_INDEX_TOKEN" >&2
        exit 1
    fi
    healthy=0
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        if curl -fsS --max-time 5 \
            -H "Authorization: Bearer $token" \
            http://127.0.0.1:9935/health |
            jq -e '.ok == true and .backend == "researka-fullraw-indexed-fts5" and .shard_dir == "/var/lib/v5-memo/v5-publish-fullraw-fts-remote" and .shard_receipt.shards_total == 1525 and .coverage_requirements.min_shards_searched == 1525 and .coverage_requirements.require_complete_search == 1 and .coverage_requirements.sweep_require_complete == 1 and .async_sweep.max_inflight == 1 and .async_sweep.workers == 1' \
                >/dev/null 2>&1
        then
            healthy=1
            break
        fi
        sleep 2
    done
    if [ "$healthy" -ne 1 ]; then
        echo "V5 publish fullraw health check failed" >&2
        exit 1
    fi
    auth_code=$(curl -sS --max-time 5 \
        -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $token" \
        -H 'Content-Type: application/json' \
        --data '{}' \
        http://127.0.0.1:9935/search)
    unset token
    if [ "$auth_code" != 400 ]; then
        echo "V5 publish fullraw authentication check failed" >&2
        exit 1
    fi
    systemctl is-active --quiet "$mount_unit" "$search_unit"
    systemctl is-enabled --quiet "$mount_unit" "$search_unit"
    install_profile "$dedicated_profile"
    restore_timers
    trap - 0
else
    install_profile "$shared_profile"
    systemctl disable --now "$search_unit" "$mount_unit"
    if systemctl is-active --quiet "$search_unit" || systemctl is-active --quiet "$mount_unit"; then
        echo "V5 publish fullraw units remain active" >&2
        exit 1
    fi
    if systemctl is-enabled --quiet "$search_unit" || systemctl is-enabled --quiet "$mount_unit"; then
        echo "V5 publish fullraw units remain enabled" >&2
        exit 1
    fi
    restore_timers
    trap - 0
fi
