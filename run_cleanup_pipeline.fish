#!/usr/bin/env fish
# Run the full cleanup + encode + merge + train pipeline with alias-aware removal.
# Override defaults by exporting variables before running (e.g. `set -x DATA_ROOT /path`).

function init_var
    set -l name $argv[1]
    set -l default $argv[2]
    if not set -q $name
        set -g $name $default
    end
end

set script_dir (dirname (status -f))
cd $script_dir

init_var PYTHON python3
init_var DATA_ROOT "/home/marqs/Bilder/pBook"
init_var WORKDIR "$script_dir/arcface_work-ppic"
init_var REMOVE_FILE "$script_dir/remove.txt"
init_var PROCESSED_DB "$WORKDIR/processed.db"
init_var EMBEDDINGS_PKL "$WORKDIR/embeddings_ppic.pkl"
init_var MERGED_EMBEDDINGS "$WORKDIR/embeddings_ppic_merged.pkl"
init_var MODEL_OUT "$WORKDIR/face_knn_arcface_ppic.pkl"

function run_step
    set -l msg $argv[1]
    set -l cmd $argv[2..-1]
    echo $msg
    $cmd
    if test $status -ne 0
        echo "Step failed: $msg" >&2
        exit $status
    end
end

function run_step_if_exists
    set -l msg $argv[1]
    set -l file_path $argv[2]
    set -l cmd $argv[3..-1]
    if test -e $file_path
        run_step $msg $cmd
    else
        echo "$msg (skippas – hittade inte $file_path)"
    end
end

run_step "[0/8] Resolve External Identities (Prolog/API)" $PYTHON fix_identities_fs.py --data-root $DATA_ROOT --db $PROCESSED_DB --embeddings $EMBEDDINGS_PKL --apply --yes --only-new
run_step "[1/8] Detect removed images" $PYTHON detect_removed.py \
    --data-root $DATA_ROOT --embeddings $EMBEDDINGS_PKL --db $PROCESSED_DB
run_step "[2/8] Apply merge candidates" $PYTHON apply_merge_candidates.py --db $PROCESSED_DB --candidates "$script_dir/to_be_merged.csv"
run_step_if_exists "[3/8] Pre-clean processed DB" $PROCESSED_DB \
    $PYTHON remove_processed.py --db $PROCESSED_DB --remove $REMOVE_FILE
run_step_if_exists "[4/8] Pre-clean embeddings" $EMBEDDINGS_PKL \
    $PYTHON remove.py --embeddings $EMBEDDINGS_PKL --remove $REMOVE_FILE
run_step "[5/8] Encode fresh embeddings" $PYTHON face_arc_pipeline.py --mode encode --data-root $DATA_ROOT --workdir $WORKDIR --allow-upsample --ui --max-yaw 40 $argv
run_step "[6/8] Merge aliases" $PYTHON merge.py
run_step "[7/8] Train KNN model" $PYTHON face_arc_pipeline.py --mode train --embeddings $MERGED_EMBEDDINGS --model-out $MODEL_OUT
run_step "[8/8] Normalize & move alias files" $PYTHON alias_cleanup.py --data-root $DATA_ROOT --db $PROCESSED_DB --prune-missing --missing-log "$WORKDIR/missing_after_alias_cleanup.txt"

echo "All steps completed."

# Töm remove.txt och to_be_merged.csv
echo "Tömmer $REMOVE_FILE och $script_dir/to_be_merged.csv..."
echo -n > "$REMOVE_FILE"
echo -n > "$script_dir/to_be_merged.csv"
