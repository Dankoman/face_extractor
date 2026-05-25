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

echo ""
read -P "❓ Vill du söka efter och ta bort visuella dubbletter i detta kör? (y/N): " run_dedup
if test "$run_dedup" = "y" -o "$run_dedup" = "Y"
    set RUN_DEDUP 1
else
    set RUN_DEDUP 0
end
echo ""

run_step "[0/11] Resolve External Identities (Prolog/API)" $PYTHON fix_identities_fs.py --data-root $DATA_ROOT --db $PROCESSED_DB --embeddings $EMBEDDINGS_PKL --apply --yes --only-new
run_step "[1/11] Detect removed images" $PYTHON detect_removed.py \
    --data-root $DATA_ROOT --embeddings $EMBEDDINGS_PKL --db $PROCESSED_DB
run_step "[2/11] Apply merge candidates" $PYTHON apply_merge_candidates.py --db $PROCESSED_DB --candidates "$script_dir/to_be_merged.csv"
run_step "[2.5/11] Clean orphan embeddings" $PYTHON clean_orphan_embeddings.py --embeddings $EMBEDDINGS_PKL --db $PROCESSED_DB --data-root $DATA_ROOT
run_step_if_exists "[3/11] Pre-clean processed DB" $PROCESSED_DB \
    $PYTHON remove_processed.py --db $PROCESSED_DB --remove $REMOVE_FILE
run_step_if_exists "[4/11] Pre-clean embeddings" $EMBEDDINGS_PKL \
    $PYTHON remove.py --embeddings $EMBEDDINGS_PKL --remove $REMOVE_FILE

if test "$RUN_DEDUP" -eq 1
    run_step "[5/11] Remove visual duplicates before encode" $PYTHON /home/marqs/Programmering/Python/3.11/doppelganger/tools/remove_visual_duplicates.py --dir $DATA_ROOT --delete --threshold 5 --db $PROCESSED_DB
else
    echo "[5/11] Remove visual duplicates before encode (skippas, valt av användare)"
end
run_step "[6/11] Encode fresh embeddings" $PYTHON face_arc_pipeline.py --mode encode --data-root $DATA_ROOT --workdir $WORKDIR --allow-upsample --ui --max-yaw 40 $argv
run_step "[7/11] Clean & Split Clusters (DBSCAN)" $PYTHON cluster_cleaner.py --data-root $DATA_ROOT --embeddings $EMBEDDINGS_PKL --db $PROCESSED_DB
run_step "[8/11] Merge aliases" $PYTHON merge.py
run_step "[9/11] Train KNN model" $PYTHON face_arc_pipeline.py --mode train --embeddings $MERGED_EMBEDDINGS --model-out $MODEL_OUT
run_step "[10/11] Normalize & move alias files" $PYTHON alias_cleanup.py --data-root $DATA_ROOT --db $PROCESSED_DB --prune-missing --missing-log "$WORKDIR/missing_after_alias_cleanup.txt" --affected-dirs-log "$WORKDIR/affected_dirs.txt"

if test "$RUN_DEDUP" -eq 1
    if test -s "$WORKDIR/affected_dirs.txt"
        run_step "[11/11] Final deduplication after alias move" $PYTHON /home/marqs/Programmering/Python/3.11/doppelganger/tools/remove_visual_duplicates.py --dir $DATA_ROOT --dirs-file "$WORKDIR/affected_dirs.txt" --delete --threshold 5
    else
        echo "[11/11] Final deduplication after alias move (skippas, inga mappar fick filer flyttade till sig)"
    end
else
    echo "[11/11] Final deduplication after alias move (skippas, valt av användare)"
end

echo "All steps completed."

# Töm remove.txt och to_be_merged.csv
echo "Tömmer $REMOVE_FILE och $script_dir/to_be_merged.csv..."
echo -n > "$REMOVE_FILE"
echo -n > "$script_dir/to_be_merged.csv"
