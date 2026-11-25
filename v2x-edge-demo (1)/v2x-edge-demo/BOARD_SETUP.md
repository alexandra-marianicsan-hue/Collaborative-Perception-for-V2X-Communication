# Conectare la placa i.MX 8M Plus și rulare

## 1. Acces la placă
**Consolă serială (merge mereu, fără rețea):** conectează cablul USB de la portul marcat *DEBUG* al plăcii la laptop.
Apar mai multe porturi seriale; deschide-le pe rând (115200, 8N1) până vezi consola Linux: PuTTY/Tera Term (Windows) sau
`picocom -b 115200 /dev/ttyUSB2` (Linux; încearcă ttyUSB0..3). Login: `root` (de obicei fără parolă pe imaginile NXP).

**Rețea (pentru scp/ssh):** cablu Ethernet de la placă la router. Pe consolă: `ip a` (caută IP-ul lui eth0), apoi de pe laptop:
```
ssh root@<IP>
scp -r v2x-edge-demo root@<IP>:/home/root/
```
Fără router: cablu direct; pe placă `ip addr add 192.168.10.2/24 dev eth0`, pe laptop IP static 192.168.10.1/24.

## 2. Verifică stack-ul de pe placă
```
uname -a; cat /etc/os-release | head -3; free -m; df -h /
python3 --version
python3 -c "import numpy; print('numpy ok')"
python3 -c "import tflite_runtime; print('tflite_runtime ok')"
ls /usr/lib/libvx_delegate.so                    # delegatul NPU
ls /usr/bin/tensorflow-lite-*/examples           # benchmark_model, label_image.py
```
Dacă lipsește `libvx_delegate.so` sau `tflite_runtime`, imaginea nu e „eIQ/imx-image-full": spune-mi ce afișează comenzile.

## 3. Măsurători rapide NPU vs CPU (fără Python, independent de proiect)
```
cd /usr/bin/tensorflow-lite-*/examples
./benchmark_model --graph=/home/root/v2x-edge-demo/models/yolov8n_vehicle_320_int8.tflite --num_runs=50 --num_threads=4
./benchmark_model --graph=/home/root/v2x-edge-demo/models/yolov8n_vehicle_320_int8.tflite --num_runs=50 \
                  --external_delegate_path=/usr/lib/libvx_delegate.so
```
Prima rulare pe NPU e lentă (se compilează graful); contează media celor 50. Adaugă `--enable_op_profiling=true` pentru timp pe operator.

## 4. Proiectul pe placă
```
cd /home/root/v2x-edge-demo
for m in fp32 fp16 int8; do python3 bench_tflite.py --model models/yolov8n_vehicle_320_$m.tflite --name $m-cpu; done
python3 bench_tflite.py --model models/yolov8n_vehicle_320_int8.tflite --name int8-npu --delegate /usr/lib/libvx_delegate.so
python3 collab_opv2v.py --model models/yolov8n_vehicle_320_int8.tflite --delegate /usr/lib/libvx_delegate.so --sleep 0.5
cat results/benchmark.md
```
Pe placă ai nevoie doar de `numpy` + `tflite_runtime` (nu OpenCV, nu PyTorch). `--save-vis dir` din `collab_opv2v.py` cere OpenCV.
