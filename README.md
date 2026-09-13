# JEPA Lab: Görüntü Temsilinden Robotik World Model'e

Bu çalışma alanı, bir JEPA checkpoint'ini yalnızca çalıştırmak yerine yöntemi katman katman anlamak için kurulmuştur. Ana öğrenme rotası şudur:

```text
I-JEPA → DINO karşılaştırması → V-JEPA → V-JEPA2
       → action-conditioned latent dynamics → CEM/MPC
```

I-JEPA ile görüntü içindeki latent prediction ve EMA teacher anatomisi öğrenilir. DINO, JEPA olarak değil; stop-gradient, EMA teacher, collapse ve ViT representation karşılaştırması olarak kullanılır. V-JEPA ile zaman boyutu eklenir. V-JEPA2 ve V-JEPA2-AC üzerinden action-free representation ile action-conditioned world model arasındaki sınır kurulur. Son adımda aynı fikirlerin küçük, eğitilebilir karşılığı 2D planar reaching simülasyonunda CEM/MPC ile sınanır.

Paper ölçeğinde reproduction bu projenin hedefi değildir. Hedef; tensor akışını, objective'i, masking'i, teacher güncellemesini ve planning bağlantısını açıklayabilecek kadar küçük ama gerçek deneyler yapmaktır.

## Kaynak kod düzeni

Meta kodları değiştirilmeden Git submodule olarak tutulur. Bu workspace'in öğretici ve cihaz-bağımsız kodu `src/jepa_lab` altındadır.

| Kaynak | Dizin | Sabit commit |
|---|---|---|
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | `upstream/ijepa` | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | `upstream/vjepa` | `51c59d518fc63c08464af6de585f78ac0c7ed4d5` |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | `upstream/vjepa2` | `9a061fffe395573a2ee61dadcd3cf5baa188e3f2` |

`jepa-lab doctor --strict`, submodule'lerin mevcut ve tam olarak bu commit'lerde olduğunu doğrular. Upstream kod üzerinde deney değişikliği yapılmaz; cihaz adaptasyonu, küçük model, görselleştirme, run metadata ve simülasyon kodu laboratuvar katmanında kalır.

Upstream depoların birden fazlası top-level `src` namespace'ini kullanır. Bu nedenle I-JEPA, V-JEPA ve V-JEPA2 aynı Python process/kernel'ına import edilmemelidir. `scripts/official_*.py` dosyalarını ayrı process olarak çalıştırın; notebook'ta upstream değiştirirken kernel'ı yeniden başlatın. Modeller arası karşılaştırmayı ortak `.npz` feature çıktıları üzerinden yapın.

## Temiz kurulum

Python `3.11–3.13` desteklenir. Git deposunu submodule'lerle birlikte alın:

```bash
git clone --recurse-submodules <YOUR_REPO_URL> jepa-study
cd jepa-study
git submodule update --init --recursive
```

VS Code ile notebook çalışmak için önerilen hafif kurulum:

```bash
uv sync --extra vscode --extra test
source .venv/bin/activate
jepa-lab doctor --strict
pytest -q
code .
```

VS Code'da bir `.ipynb` dosyasını açın; sağ üstteki **Select Kernel** menüsünden
**Python Environments → `.venv/bin/python`** seçin. Ayrı bir `jupyter lab`
sunucusu başlatmanız veya kullanıcı düzeyinde kernelspec kaydetmeniz gerekmez.
Workspace, Microsoft Python ve Jupyter eklentilerini önerir ve `.venv`
interpreter'ını varsayılan olarak gösterir.

Tüm geliştirme, video ve Hugging Face araçlarını da kurmak isterseniz:

```bash
uv sync --extra dev
source .venv/bin/activate
jepa-lab doctor --strict
pytest -q
```

`uv` yoksa standart sanal ortam kullanılabilir:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
jepa-lab doctor --strict
pytest -q
```

Kurulumdan sonra en küçük uçtan uca doğrulama:

```bash
jepa-lab foundations
jepa-lab image-smoke --device auto
jepa-lab video-smoke --device auto
jepa-lab action-smoke --device auto --steps 200
python scripts/official_ijepa_smoke.py --device auto
python scripts/official_ijepa_train.py \
  --device auto --image-size 112 --batch-size 1 --steps 1
python scripts/official_ijepa_features.py --device auto --image-size 32
python scripts/official_vjepa_features.py --device auto
python scripts/official_vjepa2_ac_replay.py --preflight-only
```

İlk dört komut öğretici `jepa_lab` modellerini sınar. `official_ijepa_smoke.py`, pinlenmiş resmî I-JEPA model, predictor ve mask bileşenleriyle bir gerçek forward/backward/EMA adımı yapar. `official_ijepa_train.py`, aynı pinlenmiş bileşenlerle tek cihazda eğitim, gradient accumulation, checkpoint/resume ve M2 ablation seçeneklerini sağlar; yukarıdaki komut yalnız bir adımlık sentetik shape/gradient kontrolüdür. `official_ijepa_features.py` resmî encoder koduyla patch feature çıkarır; varsayılanı güvenli random-init shape smoke'dur, pretrained ViT-H/14 için checkpoint ve uyumlu model argümanları gerekir. `official_vjepa_features.py` de checkpoint verilmediğinde yalnız resmî mimari üzerinde random-init shape smoke testidir. `official_vjepa2_ac_replay.py --preflight-only`, resmî trajectory şemasını, pinned kaynak kodu, opsiyonel bağımlılıkları ve donanımı büyük modeli ayırmadan kontrol eder.

VS Code yerine tarayıcı tabanlı JupyterLab kullanmak isterseniz:

```bash
uv sync --extra notebooks
uv run jupyter lab
```

## Çalışma ortamları

Ortam profillerinin makinece okunabilir tanımı `configs/profiles.yaml` dosyasındadır.

### Kaggle GPU — ana ortam

Zorunlu küçük training ve ablation'lar Kaggle GPU için tasarlanmıştır. Yeni notebook'ta Internet'i açın ve ilk hücrelerde şu akışı kullanın. Uzak Kaggle kernel'ında `vscode` veya `dev` extra'sını kurmayın; kernel ve Jupyter paketleri Kaggle tarafından yönetilir:

```python
!git clone --recurse-submodules https://github.com/ErcanKurtoglu/JEPA_experiment.git /kaggle/working/jepa-study
%cd /kaggle/working/jepa-study
%pip install -e . --no-deps
!jepa-lab doctor --strict
!pytest -q
```

Önce Kaggle'ın verdiği GPU ile kurulu PyTorch wheel'inin gerçekten uyumlu
olduğunu kontrol edin:

```python
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
```

Tesla P100 `sm_60` kullanır. Eğer arch listesinde `sm_60` yoksa
`torch.cuda.is_available()` tek başına yeterli değildir ve CUDA işlemi
başarısız olur. Mümkünse Kaggle ayarından T4/L4 seçip oturumu yeniden
başlatın. P100 kullanmak zorundaysanız temiz oturumun ilk hücresinde resmî
CUDA 12.6 wheel'ini kurun:

```python
%pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

Bu kurulumdan sonra kernel'ı yeniden başlatın, VS Code Compatible URL ile
yeniden bağlanın ve clone/install hücresini çalıştırın. PyTorch 2.8 ve sonrası
CUDA 12.8 binary'lerinde P100'ü kapsayan `sm_60` kaldırılmıştır; CUDA 12.6
binary'leri bu mimariyi korur.

VS Code arayüzünü kullanıp hücreleri Kaggle GPU'da çalıştırmak için Kaggle
Notebook Editor'da **Settings → Accelerator → GPU** seçin. Ardından
**Run → Kaggle Jupyter Server → Start Session** yolunu izleyip paneldeki
**VS Code Compatible URL** değerini kopyalayın. VS Code'da yerel `.ipynb`
dosyasını açın ve **Select Kernel → Select Another Kernel → Existing Jupyter
Server** üzerinden bu URL'yi yapıştırın. Bu özellik Kaggle tarafında deneysel
olduğundan oturum kapanınca yeni URL almak gerekebilir.

Uzak kernel yerel diski otomatik olarak bağlamaz. Bu nedenle yukarıdaki clone
ve install hücrelerini Kaggle web notebook'unda veya bağlantıdan sonra VS Code
notebook'una eklenen geçici bir bootstrap hücresinde çalıştırın.
Notebook bootstrap hücreleri hem `/kaggle/working/jepa-study` hem eski
`/kaggle/working/I-JEPA` yolunu tanır.

`kaggle_16gb` profili image microbatch `32` ve gradient accumulation `4` ile effective batch `128` hedefler. OOM durumunda microbatch'i yarıya indirip accumulation'ı iki katına çıkarın. Önce FP32 smoke testini tamamlayın, sonra mixed precision'a geçin. Sonuçları `artifacts/` veya `runs/` altında tutup oturum bitmeden dışarı aktarın.

VRAM en az 24 GB ise `cuda_24plus` profili V-JEPA2-AC GPU replay için seçilebilir. Daha küçük GPU'da reduced replay veya CPU smoke kullanılır; tam AC inference bir önkoşul değildir.

### Colab GPU — yedek ortam

Kaggle hücrelerindeki `/kaggle/working/jepa-study` yolunu `/content/jepa-study` ile değiştirin. `colab_fallback` profili görünür VRAM'e göre uygun CUDA profilini miras alır. Aynı config, seed ve dataset split'ini koruyun; Kaggle ve Colab sonuçlarını farklı training reçeteleri gibi yorumlamayın.

### Apple Silicon M4 — kod okuma ve smoke

Yerel kurulumdan sonra `--device auto` sırasıyla CUDA, MPS ve CPU kullanılabilirliğini denetler; M4 üzerinde MPS seçimi kullanılan PyTorch build'inin MPS desteğine bağlıdır. Bu seçim işlem bazında otomatik CPU fallback sağlamaz. Bir operatör MPS üzerinde desteklenmiyorsa veya kararsızsa aynı komutu açıkça `--device cpu` ile yeniden çalıştırın ve cihaz değişikliğini run metadata'sına yazın. `local_mps` profili küçük microbatch, `num_workers=0` ve FP32 kullanır. M4; unit test, kaynak kod izleme, mask görselleştirme, küçük forward ve uygun modellerde MPS inference içindir. Paper ölçeğinde training veya tam V-JEPA2-AC replay hedeflenmez.

## Öğrenme ve laboratuvar sırası

Her deneyde **dur → beklenen sonucu yaz → çalıştır → tensor/metric'i incele → farkı açıkla** döngüsünü uygulayın. Hücreleri yalnızca sırayla çalıştırmak geçiş kriteri değildir.

1. `notebooks/00_foundations_and_ijepa.ipynb` — M0/M1: image/video token hesabı, stop-gradient, EMA, multi-block mask ve resmî I-JEPA white-box step.
2. `notebooks/01_ijepa_training_and_dino.ipynb` — M2: `official_ijepa_train.py` ile Imagenette overfit/baseline/ablation akışı, collapse metrikleri, trained-vs-random probe ve resmî DINO objective karşılaştırması. Uzun koşular ayrı `RUN_*` bayraklarıyla isteğe bağlıdır.
3. `notebooks/02_video_jepa_and_vjepa2.ipynb` — M3/M4: tubelet token'ları, temporal perturbation, kısa smoke; isteğe bağlı 5.000-step tiny V-JEPA ile eşit bütçeli causal/non-causal eğitim; resmî V-JEPA/V-JEPA2 feature akışı.
4. `notebooks/03_action_world_model_and_cem.ipynb` — M5/M6: yerelde doğrulanan V-JEPA2-AC preflight ve opt-in resmî replay wrapper'ı, action-conditioned latent dynamics, rollout hatası ve receding-horizon CEM.
5. `docs/JEPA_STUDY_GUIDE.md` — M7: paper–kod eşlemesi, denklemler, model karşılaştırma tablosu, sınırlamalar ve tez hocasına aktarılabilir anlatı.

Detaylı çalışma sözleşmesi, beklenen tensor shape'leri ve her kilometre taşının kabul ölçütleri `docs/LABS.md` içindedir. Küçük simülatörün hızlı kontrolü ve Kaggle kabul koşusu şöyledir:

```bash
python scripts/planar_world_model_demo.py --profile smoke --device auto
python scripts/planar_world_model_demo.py \
  --profile smoke --latent-source visual --device auto
python scripts/planar_world_model_demo.py \
  --profile full --latent-source visual --device cuda \
  --output artifacts/planar_visual_full.json --enforce
# İsteğe bağlı mekanizma kontrolü:
python scripts/planar_world_model_demo.py \
  --profile full --latent-source analytic --device cuda \
  --output artifacts/planar_analytic_full.json
```

`visual` modu M6'nın ana kabul yoludur: planar RGB kliplerinde tiny video-JEPA'yı önce eğitir, EMA encoder'ı dondurup AC optimizer'dan çıkarır ve goal-image latent cost ile MPC çalıştırır. `analytic` modu aynı action dynamics ve CEM mekanizmasını tam bilinen XY latent ile izole eden ayrı bir teşhis/kontrol deneyidir; görsel world-model kabulünün yerine geçmez. `--enforce`, seçilen koşunun 100 held-out seed üzerindeki başarı, random-planner marjı, action-error ve CEM kontrollerini sağlamaması halinde non-zero exit code üretir. JSON özeti başarı oranıyla birlikte Wilson `%95` güven aralığını da yazar. Visual smoke'un başarısız olması geçerli bir teşhistir; bu sonuçların hiçbiri paper ölçeğinde V-JEPA2-AC iddiası değildir.

## Resmî modeller ile öğretici modellerin sınırı

İki uygulama katmanını karıştırmayın:

- `upstream/*` ve `scripts/official_*.py`: belirtilen Meta commit'lerindeki resmî model bileşenlerini kullanan izole training/feature/smoke girişleri veya resmî checkpoint arayüzleri. `official_ijepa_train.py` M2'nin tek-GPU runner'ıdır; upstream kaynak değişmeden kalır.
- `src/jepa_lab/*`: objective ve veri akışını küçük cihazlarda görünür kılan özgün eğitim uygulamaları. Bunlar paper-scale reproduction ya da resmî Meta implementasyonu değildir.

Bir dosya adında `official` geçmesi pretrained ağırlığın yüklenmiş olduğu anlamına gelmez. Doğrudan indirilen checkpoint dosyalarında dosya yolu ve SHA-256'yı; Hugging Face snapshot'larında ise model ID'si ile immutable repository revision'ını provenance olarak kontrol edin. Bunlar aynı bütünlük göstergesi değildir. Özellikle:

- I-JEPA resmî smoke, training anatomisini random input ve küçük model üzerinde kanıtlar.
- I-JEPA pretrained feature deneyi için `official_ijepa_features.py --model vit_huge --patch-size 14 --image-size 224 --checkpoint <PATH>` kullanılır; checkpoint SHA-256 ve yükleme uyuşmazlıkları çıktıda görünür.
- V-JEPA resmî feature scripti varsayılan olarak `vit_tiny` random-init'tir; `--model vit_large --checkpoint <PATH>` yalnız uyumlu resmî checkpoint ile representation deneyine dönüşür.
- V-JEPA2 scripti `facebook/vjepa2-vitl-fpc64-256` resmî Hugging Face modelini immutable revision üzerinden indirir ve `get_vision_features()` encoder yoluyla feature çıkarır; bunun için `uv sync --extra hf` gerekir. Hugging Face'in snapshot içi ağırlık doğrulaması dosya düzeyi SHA sidecar politikasından ayrıdır.
- V-JEPA2-AC wrapper'ının varsayılan preflight modu model ayırmaz. Ağır replay yalnız `--run-replay`, doğrulanmış yerel checkpoint ve uygun kaynaklarla açılır; pinned yerel kodla current/goal latent, ground-truth/zero energy, action grid ve reduced CEM üretir. Bundled örnek tek transition içerdiği için gerçek shuffled-action ve ölçülmüş iki-step error sonuçları dürüstçe `unsupported` bırakılır.

Resmî V-JEPA2-AC replay, tercihen en az 24 GB CUDA VRAM bulunan oturumda açıkça başlatılır:

```bash
uv sync --extra ac
jepa-lab download-checkpoint vjepa2_ac --root checkpoints
uv run python scripts/official_vjepa2_ac_replay.py \
  --run-replay --device cuda \
  --checkpoint checkpoints/vjepa2-ac-vitg.pt \
  --output artifacts/vjepa2_ac_replay.json
```

Checkpoint downloader'ın ürettiği `.pt.json` sidecar SHA-256 güvenini sağlar. Sidecar yoksa `--sha256` açıkça verilmelidir. Bu komut fiziksel robota action göndermez.

Örnek V-JEPA2 feature koşusu:

```bash
uv sync --extra hf
uv run python scripts/official_vjepa2_features.py \
  --device auto --frames 64 --image-size 256 \
  --output artifacts/vjepa2_temporal_features.npz
```

`fpc64` checkpoint sözleşmesi 64 kare kullanır; daha kısa bir giriş ancak ayrı bir smoke/duyarlılık deneyi olarak etiketlenmelidir. Script, Hugging Face model deposunu varsayılan olarak `b3c1679b7c34d3255ef3547f27c7b226aefab26f` immutable revision'ına sabitler.

Opsiyonel DINO karşılaştırması resmî Torch Hub girişini kullanır:

```bash
uv run python scripts/official_dino_features.py \
  --device auto --output artifacts/dino_views.npz
```

Script varsayılan olarak resmî DINO kodunu immutable `facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a` commit'inden ister ve bu ref'i JSON provenance'a yazar. `--hub-ref ...:main` gibi mutable bir branch açıkça verilirse sonuç reproducible pin sayılmaz; script bunu ayrıca uyarı olarak raporlar.

## Veri, checkpoint ve deney kaydı

Resmî model URL'leri ve kaynak commit'ler `configs/checkpoints.yaml` içinde kayıtlıdır. Büyük dosyalar otomatik olarak Git'e alınmaz: `data/`, `checkpoints/`, `artifacts/`, `runs/`, `.pt`, `.pth`, `.safetensors` ve benzeri çıktılar `.gitignore` kapsamındadır.

Imagenette-160 indirme ve provenance kaydı açık bir işlem olarak yapılır:

```bash
jepa-lab download-imagenette --root data
```

Resmî **doğrudan dosya** checkpoint URL'leri registry anahtarıyla, atomik indirme ve SHA sidecar ile alınabilir:

```bash
jepa-lab download-checkpoint ijepa_vith14_in1k --root checkpoints
jepa-lab download-checkpoint vjepa_vitl16_224 --root checkpoints
```

Bilinen bir yayıncı hash'i varsa ilk koşuda `--sha256 <DIGEST>` verin. Aksi halde ilk başarılı resmî URL indirmesinin hash'i kaydedilir; sonraki kullanımda dosya bu değerden saparsa reddedilir. Hugging Face yüklemesi bu komuttan farklıdır: model ID'si ile immutable repository revision kaydedilir; bunu tek bir yerel checkpoint dosyasının SHA-256'sı gibi raporlamayın.

Kurallar:

- Yalnız resmî repo, resmî model card veya paper'ın verdiği checkpoint URL'sini kullanın.
- Doğrudan dosya indirmesinde SHA-256 üretip yan metadata'ya yazın ve sonraki kullanımda zorunlu kılın; Hugging Face snapshot'ında model ID'si ile immutable revision'ı kaydedin.
- Model adı, URL, source commit, preprocessing, seed, config ve cihaz bilgisini JSON/CSV run özetinde saklayın.
- Checkpoint yüklemede missing/unexpected key listesini görünür bırakın; sessiz bir `strict=False` sonucunu başarı saymayın.
- Model karşılaştırmalarında feature'ları ortak `.npz` şemasına export edin; her upstream'i ayrı process/kernel'da çalıştırın.
- Her aşama için ayrı Markdown raporu üretmeyin. Notebook çıktısı ile makinece okunabilir run metadata yeterlidir.

## Kapsam ve gerçekçi beklentiler

Bu ilk sürüm şunları hedefler:

- 196 image token ve 1568 video token hesabını gerçek tensor'larla doğrulamak;
- context/predictor gradient'i, stop-gradient target ve EMA'yı sayısal olarak göstermek;
- masking, temporal perturbation ve collapse metrikleri üzerinden JEPA davranışını incelemek;
- action-free video representation ile action-conditioned causal rollout'u ayırmak;
- 2D reaching simülatöründe latent dynamics ve CEM/MPC bağlantısını çalıştırmak;
- ileride gerçek robot için preprocessing, coordinate transform, action clipping ve safety hook'larının arayüzünü hazırlamak.

Full ImageNet, VideoMix2M/22M, DROID post-training, VQA/LLM alignment, multi-GPU paper reproduction, gerçek ROS/Franka kontrolü ve fiziksel robot güvenliği bu sürümün dışındadır. Küçük ablation'ların paper sıralamasını birebir üretmesi beklenmez; veri, model ve compute ölçeği her yorumda belirtilmelidir. V-JEPA2.1 de ayrı bir sonraki çalışma konusudur.

## Lisans

Workspace'in özgün `jepa_lab` kodu MIT lisanslıdır. Ancak bağlı upstream kodların lisansı ayrıca geçerlidir:

- I-JEPA: Creative Commons Attribution-NonCommercial 4.0.
- V-JEPA (`facebookresearch/jepa`): Creative Commons Attribution-NonCommercial 4.0.
- V-JEPA2: ağırlıklı olarak MIT; bazı dosyalar farklı bildirimler taşır.
- Opsiyonel DINO karşılaştırması: Apache-2.0.

Bu çalışma akademik/non-commercial kullanım varsayımıyla hazırlanmıştır. Özellikle I-JEPA ve V-JEPA kodu veya türevleri robotik bir ürüne entegre edilmeden önce lisans uygunluğu ayrıca değerlendirilmelidir. Bağlayıcı ayrıntılar için upstream `LICENSE` dosyaları ile `THIRD_PARTY_NOTICES.txt` esas alınır.
