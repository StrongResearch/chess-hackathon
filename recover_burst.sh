# recover_burst.sh -> a helper shell script written by zac, to improve participant QoL :)
experiment_id=$1

if [ -z "$experiment_id" ] || [ $# != 1 ] ; then
        # Just a simple sanity check
        echo "Usage: bash $0 <experiment_id>"
        exit 1
fi

OUT_FOLDER="/root/exports/${experiment_id}/outputs" && latest=$(ls "$OUT_FOLDER" |grep '^CHK[0-9]*'|sort -V|tail -1)

if [ -z "$latest" ]; then
        echo "Error: could not find checkpoint for experiment ${experiment_id}! Check the ID and try again."
        exit 1
fi

echo "Found latest checkpoint: ${latest}. Last training log was:"

tail "${OUT_FOLDER}/rank_0.txt" -n 1 # To help participants easily confirm it's the right checkpoint

RECOVER_FOLDER=/root/chess-hackathon/recover/$experiment_id/

echo "Copying checkpoint.pt to ${RECOVER_FOLDER} in 1 second (CTRL+C to cancel; will override if exists)..."
sleep 2 # Give a chance to cancel, e.g if the logs were wrong


mkdir -p "$RECOVER_FOLDER"

cp "${OUT_FOLDER}/rank_0.txt" "/root/chess-hackathon/${latest}-rank_0.txt" || echo "Error copying logs?" # Copy logs too :)
cp "${OUT_FOLDER}/${latest}/checkpoint.pt" "${RECOVER_FOLDER}" && echo "Success! Now, run: 'isc train' as before, to resume burst!" && exit 0

# If we're here, the copying broke :(
echo "Error copying checkpoint from ${OUT_FOLDER}/${latest}/checkpoint.pt to ${RECOVER_FOLDER}" !" && exit 1
