#!/bin/bash
# run_demo.sh - ruleaza toata demonstratia pe placa i.MX 8M Plus, etapa cu etapa.
#   ./run_demo.sh              -> cu pauza (Enter) intre etape
#   ./run_demo.sh --no-pause   -> fara pauze
# Etape: 1) hardware  2) demo CPU FP32  3) demo NPU int8  4) benchmark CPU vs NPU
#        5) asociere intre vederi pe nuScenes (daca exista pack/)  6) (optional) copiere poze pe stick USB

BASE=/root/v2x-edge-demo/v2x-edge-demo
FP32=models/yolov8n_vehicles_ft.tflite
INT8=/root/v2x-edge-demo/ft_int8.tflite
DELEGATE=/usr/lib/libvx_delegate.so
BENCH=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model

PAUSE=1
[ "$1" = "--no-pause" ] && PAUSE=0

# zgomot de la delegat / tflite pe care nu vrem sa-l vedem pe ecran
NOISE='HandleLayoutInfer|Vx delegate:|UserWarning|tf.lite.Interpreter is deprecated|TF 2.20|migration guide|for details\.|_INTERPRETER_DELETION_WARNING'

banner() {
    echo
    echo
    echo "=============================================================="
    echo "  $1"
    echo "=============================================================="
    echo
}

next() {
    echo
    echo
    if [ "$PAUSE" = 1 ]; then
        read -r -p ">>> Apasa Enter pentru urmatoarea etapa... " _
    fi
    echo
}

# ruleaza benchmark_model si afiseaza lizibil; seteaza variabila AVG_US
bench() {
    local label="$1"; shift
    echo "--- $label ---"
    local line
    line=$("$BENCH" --graph="$INT8" "$@" 2>&1 | grep "Inference timings")
    if [ -z "$line" ]; then
        echo "EROARE: benchmark_model nu a returnat rezultate."
        AVG_US=""
        return
    fi
    local first avg
    first=$(echo "$line" | sed -E 's/.*First inference: ([0-9.e+]+),.*/\1/')
    avg=$(echo "$line"   | sed -E 's/.*Inference \(avg\): ([0-9.e+]+).*/\1/')
    awk -v f="$first" -v a="$avg" 'BEGIN {
        printf "  First inference (include compilarea pe NPU) : %9.1f ms\n", f/1000
        printf "  Inference (avg, regim stabil)              : %9.2f ms\n", a/1000
        printf "  Debit                                       : %9.1f FPS\n", 1000000/a }'
    AVG_US="$avg"
}

if [ ! -d "$BASE" ]; then echo "Lipseste $BASE"; exit 1; fi
if [ ! -f "$INT8" ]; then echo "Lipseste $INT8"; exit 1; fi
cd "$BASE" || exit 1
if [ ! -f "$FP32" ]; then echo "Lipseste $BASE/$FP32"; exit 1; fi

clear 2>/dev/null

# ---------------------------------------------------------------- 1
banner "ETAPA 1/5 - Constrangeri hardware"
echo "Arhitectura : $(uname -m)"
echo "Nuclee CPU  : $(nproc)  (4x Cortex-A53)"
echo "NPU         : ~2.3 TOPS (specificatie NXP), acces prin VX delegate"
echo
echo "Memorie (MB):"
free -m
echo
echo "Spatiu pe disc (/):"
df -h /
echo
echo "Modele:"
ls -lh "$FP32" "$INT8" | awk '{print "  " $5 "  " $9}'
next

# ---------------------------------------------------------------- 2
banner "ETAPA 2/5 - Demo pe CPU (model FP32 fine-tuned, XNNPACK, 4 fire)"
python3 demo_board.py --model "$FP32" --images demo_images --conf 0.25 2>&1 | grep -v -E "$NOISE"
next

# ---------------------------------------------------------------- 3
banner "ETAPA 3/5 - Demo pe NPU (model int8 fine-tuned, VX delegate)"
echo "(prima incarcare dureaza ~7 s: se compileaza graful pentru NPU)"
echo
python3 demo_board.py --model "$INT8" --images demo_images --delegate "$DELEGATE" --conf 0.2 2>&1 | grep -v -E "$NOISE"
next

# ---------------------------------------------------------------- 4
banner "ETAPA 4/5 - Benchmark: doar reteaua, int8, CPU vs NPU"
bench "CPU, 4 fire (XNNPACK)" --num_threads=4
CPU_US="$AVG_US"
echo
echo
echo "(pe NPU, prima inferenta dureaza ~7 s)"
echo
bench "NPU (VX delegate)" --external_delegate_path="$DELEGATE"
NPU_US="$AVG_US"
echo
echo
if [ -n "$CPU_US" ] && [ -n "$NPU_US" ]; then
    awk -v c="$CPU_US" -v n="$NPU_US" 'BEGIN {
        printf "==> NPU este de %.1fx mai rapid decat CPU (%.1f ms -> %.1f ms)\n", c/n, c/1000, n/1000 }'
fi
next

# ---------------------------------------------------------------- 5
PACK=/root/v2x-edge-demo/pack
NUSC=/root/v2x-edge-demo/nusc_int8.tflite
[ -f "$NUSC" ] || NUSC="$INT8"
if [ -f "$PACK/pairs.json" ] && [ -f demo_assoc.py ]; then
    banner "ETAPA 5/5 - Asociere intre vederi (nuScenes): acelasi vehicul in doua imagini?"
    echo "Model: $NUSC (NPU) | pack: $PACK"
    echo
    python3 demo_assoc.py --pack "$PACK" --model "$NUSC" --delegate "$DELEGATE" --only-shared --limit 12 --cues geo 2>&1 | grep -v -E "$NOISE"
    echo
    echo
    echo "--- evaluare pe toate cele 195 de perechi (rulata in prealabil, ~5 min): metode comparate ---"
    if [ -f results/assoc_eval.md ]; then tail -n 13 results/assoc_eval.md; fi
    echo
    next
else
    echo
    echo "(etapa 5 sarita: lipseste $PACK/pairs.json sau demo_assoc.py)"
    echo
fi

# ---------------------------------------------------------------- USB
read -r -p "Copiez pozele pe stick USB? [d/N] " ANS
if [ "$ANS" = "d" ] || [ "$ANS" = "D" ]; then
    banner "COPIERE POZE PE STICK"
    echo "Pune stickul in portul USB-A al placii."
    read -r -p ">>> Apasa Enter cand e conectat... " _
    echo
    sleep 2
    PART=$(ls /dev/sd?1 2>/dev/null | head -1)
    if [ -z "$PART" ]; then
        echo "Nu am gasit stickul (/dev/sdX1). Verifica cu: lsblk"
    else
        echo "Partitie gasita: $PART"
        umount /mnt 2>/dev/null
        if mount "$PART" /mnt 2>/dev/null && mountpoint -q /mnt; then
            rm -rf /mnt/demo
            cp -r results/demo /mnt/ && sync
            [ -d results/assoc ] && cp -r results/assoc /mnt/ && sync
            [ -f results/assoc_eval.md ] && cp results/assoc_eval.md /mnt/ && sync
            echo
            echo "Copiat. Continut pe stick:"
            ls /mnt/demo | head -50
            echo
            umount /mnt && echo "Stick demontat - il poti scoate."
        else
            echo "Nu am putut monta $PART (format nepotrivit?). Formateaza stickul FAT32 pe laptop."
        fi
    fi
fi

echo
echo
echo "Gata."
echo
