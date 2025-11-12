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
init_var MERGE_FILE "$script_dir/merge.txt"
init_var PROCESSED_JSON "$WORKDIR/processed-ppic.jsonl"
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

run_step "[1/7] Apply merge candidates" $PYTHON apply_merge_candidates.py --merge $MERGE_FILE --candidates "$script_dir/merge?.csv"
run_step "[2/7] Pre-clean processed JSONL" $PYTHON remove_processed.py --processed $PROCESSED_JSON --remove $REMOVE_FILE --merge $MERGE_FILE
run_step "[3/7] Pre-clean embeddings" $PYTHON remove.py --embeddings $EMBEDDINGS_PKL --remove $REMOVE_FILE --merge $MERGE_FILE --no-alias
run_step "[4/7] Encode fresh embeddings" $PYTHON face_arc_pipeline.py --mode encode --data-root $DATA_ROOT --workdir $WORKDIR --allow-upsample --verbose
run_step "[5/7] Merge aliases" $PYTHON merge.py
run_step "[6/7] Train KNN model" $PYTHON face_arc_pipeline.py --mode train --embeddings $MERGED_EMBEDDINGS --model-out $MODEL_OUT
run_step "[7/7] Normalize & move alias files" $PYTHON alias_cleanup.py --data-root $DATA_ROOT --merge $MERGE_FILE --processed $PROCESSED_JSON --prune-missing --missing-log "$WORKDIR/missing_after_alias_cleanup.txt"

echo "All steps completed."
