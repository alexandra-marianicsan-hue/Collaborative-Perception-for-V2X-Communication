# Practical demonstration 2 – ghid (termen: 24 noiembrie 2026, 16:00)

Feedback-ul profesorului: baza e bună; acum accentul cade pe **partea comună a două imagini** – cum își dă seama sistemul
că jumătatea de autobuz din imaginea A și cealaltă jumătate din imaginea B sunt **același autobuz** – și pe **nuScenes**.

## 1. Cerințe → ce am făcut

| cerință | stare | unde |
|---|---|---|
| update pe GitHub | de făcut (push) | tot folderul `v2x-edge-demo/` |
| demo live, **îmbunătățiri** în cod și metrici | gata (de rulat pe date reale) | `demo_assoc.py`, `eval_assoc.py` |
| dataset nou / extins | nuScenes (perechi de camere adiacente, GT din instance_token) | `prepare_nuscenes.py` |
| explică stack-ul Edge AI | scris mai jos (secțiunea 5) | – |
| optimizări comparate (performanță/FPS/metrici) | tabel în secțiunea 6, de completat cu măsurătorile noi | `eval_assoc.py`, `benchmark_model`, notebook |
| test într-un mediu real controlat | protocol în secțiunea 7 | `demo_assoc.py --pair` |

## 2. Cum ne dăm seama că două cutii sunt același vehicul

Când un autobuz stă la „cusătura" dintre două camere, o parte e în imaginea A și alta în B. Combinăm **trei indicii independente**
(`assoc.py`) și le dăm o singură notă de cost; apoi potrivim cutiile unu-la-unu (greedy) și le **unim într-un singur obiect**:

| indiciu | ce verifică | are nevoie de | cost | punct slab |
|---|---|---|---|---|
| **geo** – geometrie | cutia e ridicată pe planul solului în același cadru comun (calibrare + poziția camerelor, trimisă între vehicule). Același obiect = **unghiuri (bearing) care se suprapun** și **urmă pe sol apropiată**; toleranța crește pentru cutii tăiate | calibrare + poză relativă | < 1 ms | erori de poziție între vehicule |
| **app** – aspect | histogramă de culoare HSV a cutiei (Bhattacharyya) | doar imaginile | ~5 ms | vehicule de aceeași culoare; jumătățile pot arăta diferit |
| **kp** – keypoint-uri | potriviri ORB între cele două imagini (B rotită peste A când rotația e cunoscută); potrivirile care cad în cutia i din A **și** cutia j din B votează perechea (i, j) | OpenCV | 30+ ms pe x86, mai mult pe A53 | puține detalii în zona comună; scump |

Reguli în plus: poartă de clasă (car/truck/bus pot fi confundate, bicicletă cu mașină nu), poartă pe unghi (dacă direcțiile nu se ating, nu pot fi același obiect).
Mesajul trimis între vehicule rămâne mic: **10 B / detecție**, nu imagini.

**Ce demonstrează testele:** `test_assoc_synth.py` (lume sintetică, camere ca în nuScenes): F1 0,99 fără zgomot și ~0,96 cu 1 m / 2° eroare de poziție.
Geometria singură ajunge; culoarea și keypoint-urile sunt plasa de siguranță și singurele utilizabile când **nu ai calibrare** (două poze făcute de tine).

## 3. Pași (în ordine)

### A. Descarcă nuScenes mini (acum, sunt ~4 GB)
nuscenes.org → cont gratuit → Download → **Mini** (`v1.0-mini`) → dezarhivează (ex. `D:\nuscenes`, trebuie să conțină `v1.0-mini/`, `samples/`, `maps/`).

### B. (opțional, 1 minut) probă fără date reale
```
pip install nuscenes-devkit pillow numpy pyquaternion
python make_fake_nuscenes.py --out data/fake_nuscenes
python prepare_nuscenes.py --dataroot data/fake_nuscenes --out data/fake_out
python eval_assoc.py --pack data/fake_out/pack --dets oracle
```
Dacă ultimul tabel arată `geo` cu F1 ≈ 1.00, lanțul de calibrări funcționează.

### C. Pregătește datasetul (pe laptop)
```
python prepare_nuscenes.py --dataroot D:/nuscenes --version v1.0-mini --out data/nuscenes_v2x
```
Rezultat: `data/nuscenes_v2x/yolo/` (imagini + etichete pentru fine-tuning) și `data/nuscenes_v2x/pack/` (≈150 de perechi pentru placă).
Scriptul afișează numărul de imagini, de cutii pe clasă, de perechi cu vehicule comune și de cutii „tăiate". Dacă `pip install nuscenes-devkit` dă erori, folosește un venv cu Python 3.10/3.11.
Arhivează folderul `yolo` ca `nuscenes_yolo.zip`.

### D. Fine-tuning + export (Colab)
Deschide `v2x_nuscenes_finetune.ipynb`, runtime cu GPU, încarcă `best.pt` (modelul tău actual) și `nuscenes_yolo.zip`, `Run all`.
Primești `v2x_nuscenes_export.zip` cu `nusc_fp32.tflite`, `nusc_w8a32.tflite`, `nusc_int8.tflite`, `best_nuscenes.pt` și `results_nuscenes.md`
(modelul vechi pe nuScenes **înainte** de fine-tuning, după fine-tuning, FP32/w8a32/int8) – exact tabelul „înainte/după".
Dacă Colab nu are GPU, pune `EPOCHS = 0` ca să faci doar exportul.

### E. Pe placă (cu stick USB; fără rețea)
Pune pe stick: `pack/` (din `nuscenes_v2x/`), `nusc_int8.tflite`, `nusc_fp32.tflite` și fișierele noi `assoc.py nusc_geometry.py detect_utils.py eval_assoc.py demo_assoc.py tflite_utils.py collab.py run_demo.sh`.
```
lsblk
mount /dev/sda1 /mnt
cp -r /mnt/pack /root/v2x-edge-demo/
cp /mnt/nusc_*.tflite /root/v2x-edge-demo/
cd /root/v2x-edge-demo/v2x-edge-demo
cp /mnt/*.py /mnt/run_demo.sh .
sed -i 's/\r$//' run_demo.sh
chmod +x run_demo.sh
umount /mnt
```
(Dacă folderul `pack` cu `pairs.json` nu e la `/root/v2x-edge-demo/pack`, editează `PACK=` în `run_demo.sh`.)

### F. Rulează
```
# demo cu legături colorate, hartă de sus și latență pe etape
python3 demo_assoc.py --pack ../pack --model ../nusc_int8.tflite --delegate /usr/lib/libvx_delegate.so --only-shared --limit 12

# metrici, toate metodele comparate (pe NPU); --tune alege ponderile pe prima jumătate și raportează pe a doua
python3 eval_assoc.py --pack ../pack --dets model --model ../nusc_int8.tflite --delegate /usr/lib/libvx_delegate.so --tune

# robustețe la erori de poziție între vehicule (GPS/pose V2X)
python3 eval_assoc.py --pack ../pack --dets model --model ../nusc_int8.tflite --delegate /usr/lib/libvx_delegate.so --pose-noise-m 0.5 --pose-noise-deg 1

# totul, cu pauze între etape (ca până acum)
./run_demo.sh
```
Imaginile rezultate sunt în `results/assoc/` (le copiezi pe stick: `run_demo.sh` o face la final). Tabelul se adaugă în `results/assoc_eval.md`.

**OpenCV pe placă:** e instalat pe imaginea plăcii (indiciul `kp` rulează acolo, ~127 ms/pereche). Dacă lipsește pe altă imagine, demo-ul merge oricum cu geo + app și te anunță.

## 4. Cum citești rezultatele

* **P (verified)**: un link e *corect* (ambele cutii = același vehicul din GT), *greșit* (ambele cutii au GT, dar vehicule diferite) sau *nejudecat* (o cutie nu are corespondent în GT: vehicul detectat dar neadnotat, departe/parcat/ocluz). Precizia se calculează doar pe primele două categorii; linkurile nejudecate apar separat. O versiune anterioară a scriptului număra nejudecatele ca greșeli și subestima precizia.
* **count error**: „fără asociere" numără de două ori un vehicul văzut în ambele imagini; cu asociere, cutiile legate se unesc. Se calculează doar pe cutiile cu corespondent în GT.
* **recall on cut objects**: pe cutiile lipite de marginea imaginii (exact cazul „jumătate de autobuz").
* **ms/pair**: costul asocierii. `geo` e sub 1 ms, `kp` domină. Asta e partea de **bottleneck** nouă: detecția (~21 ms pe NPU) vs asocierea.
* În `demo_assoc.py` vezi acum **toate etapele** cronometrate: letterbox, inferență, decodare+NMS, asociere – inclusiv costurile pe care în prima demonstrație nu le măsuram.

## 5. Stack-ul Edge AI ales și de ce

**Ce folosim:** YOLOv8n → export **LiteRT/TFLite** (int8 static) → **tflite_runtime 2.19** pe i.MX 8M Plus; CPU cu **XNNPACK**, NPU prin **VX delegate** (`libvx_delegate.so`, driver Vivante/VeriSilicon).
Am măsurat noi: FP32 pe NPU e inutilizabil (~1,4 s), int8 pe NPU 21 ms față de 66 ms pe CPU.

| opțiune | potrivire aici |
|---|---|
| **TFLite/LiteRT** (ales) | rulează pe Linux/A53, are delegat pentru NPU-ul plăcii (testat), export int8 direct din Ultralytics |
| ONNX Runtime | alternativă validă pe Linux; nu l-am testat; acceleratorul NPU ar necesita providerul NXP/VSI – de verificat în documentația NXP eIQ |
| TFLite Micro | pentru microcontrolere fără sistem de operare; placa noastră rulează Linux, deci nu e cazul |
| ML Kit (Firebase) / Core ML | pentru Android/iOS, respectiv Apple; nu rulează pe această placă |
| PyTorch Mobile / ExecuTorch | țintește telefoane; nu am o cale spre NPU-ul plăcii |
| ELL (Embedded Learning Library) | proiect Microsoft care nu mai e dezvoltat activ (de verificat) |

Spune clar că alternativele nu au fost testate pe placă; alegerea se bazează pe suportul pentru NPU, pe care l-am verificat doar pentru TFLite + VX delegate.

## 6. Comparația optimizărilor (completează coloanele goale cu măsurătorile noi)

Deja măsurat (COCO, 130 imagini de validare, 5 clase, 320×320):

| variantă | dimensiune | mAP50 | latență pe placă |
|---|---|---|---|
| PyTorch FP32 | 6,5 MB | 0,456 | – |
| TFLite FP32, CPU 4 fire | 12,7 MB | 0,462 | ~90 ms (11 FPS) |
| TFLite FP32, NPU | 12,7 MB | 0,462 | ~1380 ms |
| TFLite w8a32 | 3,4 MB | 0,456 | de măsurat |
| TFLite int8, CPU 4 fire | 3,4 MB | 0,379 | 66,0 ms (15 FPS) |
| TFLite int8, **NPU** | 3,4 MB | 0,379 | **21,2 ms (47 FPS)** |

De adăugat din rulările noi (nuScenes): mAP50 înainte/după fine-tuning și pe FP32/w8a32/int8 (din `results_nuscenes.md`), plus două axe ieftine:
```
B=/usr/bin/tensorflow-lite-2.19.0/examples/benchmark_model
for t in 1 2 4; do $B --graph=/root/v2x-edge-demo/nusc_int8.tflite --num_threads=$t 2>&1 | grep "Inference timings"; done   # scalare cu firele
$B --graph=/root/v2x-edge-demo/nusc_w8a32.tflite --num_threads=4 2>&1 | grep "Inference timings"                       # w8a32 pe CPU
```
și rezoluția 320 față de 416 (celula opțională din notebook, apoi `benchmark_model` pe fișierul exportat).
Pentru asociere, coloana `ms/pair` din `eval_assoc.py` compară costul indiciilor (geo, geo+app, geo+app+kp).
**Optimizare de preprocesare** (măsurată pe placă): inputul int8 are `scale = 1/255`, deci cuantizarea e doar `pixel + zero_point`. Calea veche făcea float (`/255`, `/scale`, `round`, `clip`); calea nouă face aritmetică întreagă (identică bit cu bit, verificat pe 4 combinații de layout/tip). Compari înainte/după cu aceeași comandă:
```
V2X_SLOW_PRE=1 python3 eval_assoc.py --pack ../pack --dets model --model ../nusc_int8.tflite --delegate /usr/lib/libvx_delegate.so   # calea veche
python3 eval_assoc.py --pack ../pack --dets model --model ../nusc_int8.tflite --delegate /usr/lib/libvx_delegate.so                  # calea nouă
```
și citești `quantize` din linia „detector per image". Pe x86 diferența e 0,76 → 0,23 ms; pe A53 nu am măsurat-o încă.

Optimizări aplicate: transfer learning cu backbone înghețat, rezoluție mică, cuantizare int8, fuziune de operatori făcută de convertor (BN în convoluții), mesaje de detecții în loc de imagini, asociere geometrică ieftină. **Nu am aplicat** pruning și distillation.

## 7. Test în mediu real controlat

Protocol: 3–5 obiecte (mașinuțe de jucărie sau mașini parcate), două fotografii ale aceleiași scene din două poziții la 1–2 m distanță, cu zonă comună; într-o scenă lasă un obiect **tăiat de marginea** fiecărei poze.
Pune pozele pe stick și rulează:
```
python3 demo_assoc.py --pair photoA.jpg photoB.jpg --model ../nusc_int8.tflite --delegate /usr/lib/libvx_delegate.so
```
Fără calibrare nu există indiciul geometric: rămân culoarea și keypoint-urile (**ai nevoie de OpenCV pe placă**, vezi mai sus). Notează câte legături sunt corecte, câte greșite, și latența. Raportează sincer mărimea mică a testului.

## 8. Ce îi spui profesorului (3–5 minute)

1. **Problema:** doi observatori văd părți diferite ale aceluiași vehicul; fără asociere îl numără de două ori și nu poți fuziona informația.
2. **Soluția:** trei indicii (geometrie în cadru comun, culoare, potriviri de keypoint-uri), cost combinat, potrivire unu-la-unu, unire în obiect unic. Geometria funcționează cu poziția relativă trimisă între vehicule; culoarea și keypoint-urile merg și fără calibrare.
3. **Date:** nuScenes: camere vecine ale aceleiași mașini ca „doi observatori" cu perspective reale; adevărul de teren vine din `instance_token`. Fine-tuning pe nuScenes: mAP50 înainte → după (din notebook).
4. **Rezultate:** tabelul de asociere (fără asociere / geo / geo+app / geo+app+kp): precizie, recall, recall pe obiecte tăiate, eroare de numărare; robustețe la eroarea de poziție.
5. **Edge:** latențe pe etape pe NPU vs CPU, costul asocierii pe A53, stack-ul ales și de ce, optimizările comparate.
6. **Limitări:** vezi mai jos. Spune-le tu primul.

## 9. Limitări (de spus deschis)

* nuScenes: cele două „vehicule" sunt două camere ale **aceleiași mașini** (rigid, calibrare exactă); în V2X real mașinile sunt distincte, cu erori de poziție – de aceea am testat și cu zgomot de poziție adăugat, dar nu cu două vehicule reale.
* Suprapunerea dintre camere vecine e doar ~15–20°; obiectele din zona comună sunt puține.
* Vehiculele ocluse nu apar în nicio imagine; nu le rezolvăm.
* Asocierea nu e învățată (reguli + costuri reglate pe puține date); un model de re-identificare ar fi pasul următor.
* Energia nu e măsurată; pruning și distillation nu au fost aplicate.
* Rezultatele pe sintetic (F1 0,99) sunt un test de corectitudine al codului, **nu** rezultate pe date reale.
* Setul de validare e mic (nuScenes mini): diferențele mici nu sunt semnificative statistic.

## 10. Plan până la termen (24 nov, 16:00)

1. Azi: descarcă nuScenes mini; rulează pașii B și C.
2. Azi/mâine dimineață: Colab (D), apoi stick → placă (E, F).
3. Mâine: completează tabelele (secțiunea 6), fă testul din secțiunea 7 dacă ai timp, push pe GitHub.
4. Rulează o repetiție completă cu `run_demo.sh` și păstrează ieșirea ca backup.
