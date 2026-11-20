# Resolve the OPC slab topology by walking upward from the working directory.
# Source this file; do not execute it. TOPOLOGY, when already set, is kept.

find_up() {
    rel="$1"
    dir="$PWD"
    while [ "$dir" != "/" ]; do
        if [ -s "$dir/$rel" ]; then
            printf '%s\n' "$dir/$rel"
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

if [ -z "${TOPOLOGY:-}" ]; then
    TOPOLOGY=$(find_up build/opc_slab.prmtop || true)
fi
if [ -z "${TOPOLOGY:-}" ] || [ ! -s "$TOPOLOGY" ]; then
    echo "Set TOPOLOGY to the Amber topology. The search looks upward for build/opc_slab.prmtop." >&2
    exit 1
fi
