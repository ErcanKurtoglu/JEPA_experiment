# JEPA Laboratuvarları: M0–M7

Bu belge, JEPA çalışma rotasının elle uygulanacak laboratuvar sözleşmesidir. Her aşama **tahmin et → çalıştır → incele → açıkla** döngüsünü kullanır. Amaç hücreleri mekanik olarak yürütmek değil, çalıştırmadan önce beklenen davranışı yazabilmek ve sonuçla karşılaştırmaktır.

Komut ve notebook adlarında `README.md` kaynak gerçektir. Bu belgede “ilgili CLI komutu”, “ilgili notebook” gibi ifadeler kullanılması bilinçlidir; upstream entrypoint'leri veya henüz sabitlenmemiş root CLI adları burada kopyalanmaz.

## 1. Çalışma protokolü

### Ortam önceliği

1. **Kaggle GPU:** Zorunlu küçük training ve ablation'ların ana ortamı.
2. **Colab GPU:** Kaggle erişimi veya quota yoksa aynı config ve seed ile yedek.
3. **M4:** Unit test, kod okuma, mask görselleştirme, küçük CPU/MPS smoke ve uygun checkpoint inference.

Her oturumun başında ilgili ortam doğrulama komutu çalıştırılmalı ve şu bilgiler run metadata'sına yazılmalıdır:

- Git commit ve submodule commit'leri.
- Python, PyTorch ve torchvision/transformers sürümleri.
- `cuda`, `mps` veya `cpu`; GPU adı ve görünen VRAM.
- Config adı, seed, dataset split kimliği.
- Doğrudan checkpoint dosyasında ad/yol ve SHA-256; Hugging Face snapshot'ında model ID'si ve immutable repository revision.

Kaggle'da internet yalnız dependency, dataset ve resmî checkpoint indirmek için açılmalıdır. Kalıcı sonuçlar `/kaggle/working` altından dışarı aktarılmalıdır. Colab'da Drive kullanımı zorunlu değildir; kullanılırsa cache ile kaynak kod birbirinden ayrılmalıdır. M4'te `auto` MPS'yi seçtikten sonra desteklenmeyen bir operatör çıkarsa işlem bazında otomatik fallback beklenmez; komut açıkça `--device cpu` ile yeniden çalıştırılır ve bu değişiklik kaydedilir.

### Deney kaydı: rapor değil, metadata

Her aşama için ayrı Markdown raporu oluşturulmaz. Bir run'ın makinece okunabilir özeti JSON veya CSV'de şunları içerir:

```text
run_id, milestone, source_commit, config, seed, device,
dataset_id, split_hash, checkpoint_sha256_or_null, model_revision_or_null,
train_loss, validation_metrics, wall_time, peak_memory, status
```

Plot ve `.npz` feature çıktıları run klasöründe tutulabilir. Büyük artifact ve checkpoint'ler Git'e commit edilmez.

### Pause–predict–run–inspect kuralı

Her deneyden önce:

1. **Pause:** Hücreyi/komutu henüz çalıştırma.
2. **Predict:** Beklenen shape, gradient, loss sıralaması veya grafik yönünü bir cümleyle not et.
3. **Run:** Yalnız ilgili küçük deneyi çalıştır.
4. **Inspect:** Sonucu assertion, tablo veya grafikle doğrula.
5. **Explain:** Beklenti ile sonuç farklıysa önce data flow ve config'i incele; hemen “paper yanlış” sonucuna varma.

## 2. Dataset ve checkpoint politikası

### Dataset

- I-JEPA eğitiminde **Imagenette-160**, seed `42` ve sabit train/validation manifest kullanılacaktır.
- Dataset indirme URL'si, sürümü, lisansı, arşiv checksum'u ve split manifest hash'i kaydedilecektir.
- `ImageFolder` class sıralaması bir metadata dosyasında saklanacaktır; klasör sırası değiştiğinde label kayması sessiz kalmamalıdır.
- Overfit testi, train manifestinden seçilen sabit 128 örnek kullanacaktır.
- Sentetik hareketli şekil ve planar reaching verisi deterministic generator ve kaydedilmiş seed ile üretilecektir.
- Robot trajectory replay yalnız resmî örnek dosyanın beklenen key, dtype ve shape'leri doğrulandıktan sonra başlayacaktır.
- Dataset dosyaları, cache, extracted frames ve büyük `.npz` çıktıları Git'e alınmayacaktır.

### Checkpoint

- Yalnız resmî repo, resmî model card veya paper'ın işaret ettiği resmî URL kullanılacaktır.
- Doğrudan checkpoint dosyası için `model_id`, kaynak URL, erişim tarihi, upstream commit/tag ve dosya SHA-256'sı kaydedilecektir. Hugging Face snapshot'ı için model ID'si ve immutable repository revision kaydedilir; repository revision dosya SHA-256'sı diye adlandırılmaz.
- Yükleme `strict` sonucu, missing/unexpected key listesiyle görünür olmalıdır. Sessiz `strict=False` kabul edilmez.
- Context encoder, target encoder ve predictor checkpoint'leri birbirinin yerine kullanılmaz.
- Checkpoint dosyası Git'e alınmaz; metadata registry alınabilir.
- Doğrudan dosyanın checksum'u uyuşmazsa dosya kullanılmaz ve yeniden indirilir; Hugging Face yüklemesi beklenen immutable revision dışında çözülürse koşu reddedilir.

### Ortak feature export

Model karşılaştırmaları çakışan upstream `src` namespace'lerini aynı Python process'ine yüklemez. Her adapter ayrı process/kernel'da çalışır ve ortak `.npz` şemasına export eder:

```text
features       float32[S,N,D]
labels         int64[S] veya görev metadata'sı
sample_ids     string[S]
frame_indices  int64[S,T]
```

Yan metadata'da model/checkpoint, doğrudan dosya SHA-256'sı veya model repository revision'ı, preprocessing, input order, pooling, fps/stride, resolution ve seed bulunmalıdır.

## M0 — PyTorch, ViT ve deney ortamı

### Amaç

Patch/tubelet tokenization, positional information, attention, detach, optimizer gradient ve EMA güncellemesini gerçek tensor'lar üzerinde görmek.

### Kurulum ve çalışma

- README'deki environment bootstrap adımlarını temiz bir Kaggle oturumunda çalıştır.
- Aynı smoke akışını Colab'da ve M4 üzerinde tekrar et.
- İlgili M0 notebook/CLI ile image patchify ve video tubelet örneklerini üret.
- Küçük iki-layer context/target modelinde bir forward/backward/EMA step çalıştır.

Beklenen shape'ler:

```text
image:       [B,3,224,224]
patch=16:    [B,196,D]       çünkü 14×14=196

video:       canonical [B,16,3,224,224]
upstream:              [B,3,16,224,224]
tubelet=2×16×16:       [B,1568,D] çünkü 8×14×14=1568
```

### Çalıştırmadan önce tahmin et

- `patch=14` olursa image token sayısı kaç olur?
- `T=15`, tubelet temporal size `2` ise implementasyon reject mi, crop mı, pad mi etmeli?
- Target output `detach()` edildiğinde hangi parametrelerin `.grad` alanı `None` olmalı?
- \(\tau=0.996\) iken target parametre yeni context parametreye ne kadar yaklaşmalı?

### İncele

- Patch/tubelet reconstruction indeksleri spatial ve temporal sırayı koruyor mu?
- Target parametrelerinde gradient yok mu?
- Context/predictor gradient'leri finite ve en az biri sıfırdan farklı mı?
- EMA sonrası değer, elle hesaplanan `tau*old_target + (1-tau)*context` ile tolerans içinde mi?

### Kabul kriteri

- Yukarıdaki image ve video shape assertion'ları geçer.
- Context ve predictor gradient alır; target encoder gradient almaz.
- EMA sayısal testi `rtol=1e-5`, `atol=1e-7` içinde geçer.
- Aynı seed ile mask ve toy loss tekrarlanabilir.
- Kullanıcı 196 ve 1568 token hesabını ve stop-gradient/EMA farkını kod bakmadan açıklayabilir.

## M1 — I-JEPA anatomisi ve ilk inference

### Amaç

I-JEPA paper denklemini resmî kodun context encoder, target encoder, predictor, mask collator ve training loop'una bağlamak.

### Çalışma

- Pinlenmiş resmî I-JEPA submodule'ünde ilgili sembolleri kaynak okuma haritasıyla bul.
- Resmî pretrained encoder'ı batch-1 bir görüntü üzerinde çalıştır.
- Patch token, mean-pooled feature ve iki augmentation arasındaki cosine similarity'yi çıkar.
- Dört target'lı multi-block maskeyi `14×14` grid üzerinde görselleştir.
- Tek batch forward/backward ve ardından tek EMA step çalıştır.

Maske kontrolü:

- target scale `0.15–0.20`;
- target aspect ratio `0.75–1.5`;
- context scale `0.85–1.0`;
- context ve target indeksleri ayrık;
- target-target overlap yasaklanmamış.

### Çalıştırmadan önce tahmin et

- Target mask encoder girdisinde uygulanırsa target representation nasıl değişir?
- Dört target'ın alan oranlarını toplamak neden gerçek masked ratio'yu vermez?
- Predictor target konum bilgisini almazsa hangi belirsizlik oluşur?
- Target encoder'a optimizer step uygulanırsa EMA teacher mantığı nasıl bozulur?

### İncele

- Target encoder gerçekten tam görüntüyü işliyor mu?
- Target token seçimi encoder çıktısından sonra mı yapılıyor?
- Target `.grad` değerleri `None`, context ve predictor gradient normları finite mi?
- Görselleştirmede target context'e sızıyor mu?
- Checkpoint loader missing/unexpected key üretmiş mi?

### Kabul kriteri

- Resmî pretrained ViT-H/14, `224×224` girdide finite `[1,256,1280]` patch feature üretir (`16×16=256` token). M0'daki ViT/16 öğretici hesabı ayrı olarak `[1,196,D]`'dir.
- Context-target kesişimi her örnekte boştur; mask indeksleri kullanılan grid'e uyar: ViT/16 için `[0,195]`, resmî ViT-H/14 için `[0,255]`.
- Target gradient'leri yoktur; context/predictor gradient normları sıfırdan büyüktür.
- Bir EMA step elle hesapla aynı sonucu verir.
- Kullanıcı paper'daki squared L2 ile kodun target LayerNorm + SmoothL1 reçetesini açıkça ayırabilir.

## M2 — Küçük I-JEPA eğitimi ve DINO karşılaştırması

### Amaç

Küçük veride resmî I-JEPA bileşenleriyle öğrenme sinyalini ölçmek; DINO'yu JEPA saymadan iki objective'i karşılaştırmak.

### Baseline config

```text
dataset:          Imagenette-160, fixed split, seed 42
encoder:          ViT-Tiny, image 224, patch 16
predictor:        width 192, depth 4
targets:          4, multi-block
target recipe:    LayerNorm + SmoothL1
EMA:              0.996 → 1.0
effective batch:  128
```

GPU belleği yetmezse microbatch ikiye bölünür ve gradient accumulation aynı oranda artırılır. Effective batch, optimizer step sayısı, augmentation ve seed değişmez. Bu OOM politikası bir defada uygulanır; hâlâ OOM varsa mixed precision veya daha küçük microbatch seçilir ve run metadata'sında kaydedilir.

### Çalıştırma sırası

1. 10 optimizer-step smoke: loss finite, checkpoint save/load çalışıyor.
2. Sabit 128 örnekte 300-step overfit: pipeline'ın öğrenebildiğini göster.
3. 30-epoch baseline.
4. Aynı seed, veri, augmentation ve step bütçesiyle 10-epoch ablation'lar:
   - dört target ↔ tek target;
   - multi-block ↔ random patch;
   - target output masking ↔ input masking;
   - scheduled EMA ↔ momentum `0` teacher.
5. Resmî pretrained DINO ViT-S/16 ile aynı görüntülerin original, crop, color jitter ve occlusion sürümlerini encode et. DINO sıfırdan eğitilmez.

Her evaluation'da training loss yanında feature std, mean pairwise cosine, covariance effective rank, k-NN ve frozen linear probe raporlanır. Random-init aynı mimari kontrolü aynı split ve probe ayarıyla değerlendirilir.

Bu sıra `notebooks/01_ijepa_training_and_dino.ipynb` içinden pinlenmiş Meta model/predictor/mask bileşenlerini kullanan `scripts/official_ijepa_train.py` ayrı process'ine bağlanmıştır. `RUN_IMAGENETTE_SMOKE`, `RUN_IMAGENETTE_OVERFIT`, `RUN_LONG_IMAGENETTE` ve `RUN_IMAGENETTE_ABLATIONS` varsayılan olarak kapalıdır; Kaggle'da istenen aşama bilinçli biçimde açılır. Baseline runner `--epochs 30 --evaluate-probes`, dört ablation ise tek tek `--epochs 10` ile çalışır. Notebook'un varsayılan hızlı hücreleri bu uzun koşuların tamamlandığı iddiası değildir.

### Çalıştırmadan önce tahmin et

- Overfit loss'u düşmezse ilk şüphe data/gradient/mask akışının hangisi olmalı?
- Momentum `0` teacher temsil kararlılığını nasıl etkiler?
- DINO'da crop benzerliği mi, I-JEPA'da occluded spatial target prediction mı daha doğrudan objective'e bağlıdır?
- Loss düşerken feature std de sıfıra gidiyorsa sonuç başarı sayılır mı?

### İncele

- Effective batch gerçekten `microbatch × accumulation` olarak 128 mi?
- Probe sırasında encoder tamamen frozen mı?
- Random encoder ile trained encoder preprocessing'i birebir aynı mı?
- Ablation farkı birden fazla değişkenden kaynaklanıyor olabilir mi?
- Küçük deney paper sıralamasını üretmediyse confidence interval ve compute farkı not edilmiş mi?

### Kabul kriteri

- 10-step smoke finite tamamlanır ve resume aynı step/loss civarından devam eder.
- 128-image overfit'te son 50-step ortalama loss, ilk 50-step ortalamasından en az `%30` düşüktür.
- Baseline frozen linear veya k-NN accuracy, aynı mimaride random encoder'dan en az `5` yüzde puan yüksektir.
- L2-normalize pooled feature'larda mean off-diagonal cosine `<0.99`, covariance effective rank `≥max(2, 0.05D)` ve feature-dimension standard deviation ortalaması `≥0.01/sqrt(D/192)` olur; üç eşikten biri ölçek nedeniyle değiştirilirse değişiklik baseline çalışmadan önce config'te sabitlenir.
- Her ablation yalnız tek faktörü değiştirir ve aynı bütçeyle tamamlanır.
- Kullanıcı DINO'nun view-invariance cross-entropy'si ile I-JEPA'nın spatial latent prediction'ını şekil ve loss düzeyinde ayırabilir.

## M3 — V-JEPA: görüntüden videoya

### Amaç

Tubelet tokenization, full-time video masking ve non-causal video representation öğrenmeyi anlamak.

### Resmî checkpoint laboratuvarı

- Aynı 16-frame test klibini normal, reverse, deterministic shuffled ve first-frame-repeat olarak hazırla.
- Her sürümde aynı resolution, normalization, frame stride ve pooling kullan.
- Resmî V-JEPA ViT-L/16 target encoder'dan token ve pooled feature export et.
- Short/long maskeleri `[T/2, H/16, W/16]` grid üzerinde görselleştir.

Beklenen ana shape:

```text
canonical input: [1,16,3,224,224]
upstream input:  [1,3,16,224,224]
latent grid:     [1,8,14,14,1024]
flattened:       [1,1568,1024]
```

### Öğretici küçük eğitim

```text
data:             deterministic moving shapes
clip:             8 frames, 112×112
encoder:          ViT-Tiny
tubelet:          2×16×16
predictor:        width 192, depth 4
loss:             L1 latent prediction
budget:           5,000 optimizer steps
```

Notebook varsayılan olarak `8×64×64`, width `48`, 20-step smoke çalıştırır. Plan config'ine uyan `8×112×112`, width `192`, 5.000-step yol `RUN_5000_STEP_VJEPA=True` ile; non-causal multi-block ve yalnız geçmişten target future'a bakan eşit bütçeli kısa eğitim ise `RUN_CAUSAL_ABLATION=True` ile isteğe bağlı açılır. Aynı eğitilmiş modelde iki maskenin tek-forward loss'unu kıyaslamak yalnız mekanizma kontrolüdür; causal/non-causal ablation sonucu sayılmaz.

### Çalıştırmadan önce tahmin et

- Normal ve first-frame-repeat feature'larından hangisi daha farklı olmalı?
- Full-time spatial tube mask neden hareket izini tek bir frame maskesinden farklı etkiler?
- Non-causal modelin reverse klibe duyarlı olması causal future prediction kanıtı mıdır?
- Causal maske daha zor bir objective ise loss'u nasıl değişebilir?

### İncele

- Frame permutation'dan önce ve sonra tensor order doğru mu?
- Maske temporal eksenin tamamını gerçekten kapsıyor mu?
- Normal-target L1, sample'lar arasında shuffled target L1'dan küçük mü?
- Motion direction probe split'i aynı trajectory'nin yakın frame'lerini train ve validation'a sızdırıyor mu?

### Kabul kriteri

- Resmî encoder beklenen `[1,1568,1024]` finite tensor'ı üretir.
- Her tube mask seçilen spatial konumu tüm temporal token katmanlarında kapsar.
- Held-out testte matched target ortalama L1, shuffled target ortalama L1'dan düşüktür.
- Frozen motion-direction probe, random encoder kontrolünü en az `10` yüzde puan aşar.
- Normal/reverse/shuffle/repeat cosine-distance tablosu aynı preprocessing ile üretilir.
- Kullanıcı V-JEPA'nın neden güçlü video representation modeli fakat varsayılan haliyle action-conditioned causal rollout modeli olmadığını açıklayabilir.

## M4 — V-JEPA2 representation deneyi

### Amaç

V-JEPA2'nin V-JEPA'ya göre ölçekleme değişikliklerini ve resmî checkpoint davranışını incelemek.

### Çalışma

- Paper ve pinlenmiş kodda şu değişimleri eşleştir:
  - absolute 3D sin-cos → 3D RoPE;
  - daha büyük image/video karışımı;
  - progressive spatial/temporal resolution;
  - action-free pretraining ile AC post-training ayrımı.
- Resmî `facebook/vjepa2-vitl-fpc64-256` model card'ına göre processor ve encoder'ı yükle.
- M3'teki aynı sürümlenmiş normalize hareket yörüngesini ve aynı perturbation tanımlarını encode et. Her checkpoint kendi native sözleşmesini kullandığı için V-JEPA v1 bunu `16×224×224`, V-JEPA2 fpc64 ise `64×256×256` olarak yeniden render/sample eder; ham tensor'ların birebir aynı olduğu iddia edilmez.
- V-JEPA ve V-JEPA2 export'larını ortak `.npz` şemasında, aynı probe split'iyle kıyasla.

### Çalıştırmadan önce tahmin et

- 3D RoPE checkpoint resolution/clip length aktarımına nasıl yardımcı olabilir?
- Farklı model preprocessing'i zorunluysa “aynı girdi” karşılaştırması nasıl adil kaydedilir?
- Daha yüksek probe accuracy, action-conditioned planning kabiliyeti gösterir mi?

### İncele

- V-JEPA2 model ID'si ve immutable Hugging Face revision sabit mi; doğrudan V-JEPA dosyasının SHA-256'sı ayrıca kayıtlı mı?
- Decoder aynı frame indekslerini veriyor mu?
- Pooling yöntemi iki model için açıkça kayıtlı mı?
- Probe hyperparameter araması test setine bakmadan mı yapıldı?

### Kabul kriteri

- Resmî ViT-L checkpoint batch-1 short-video feature'ı üretir; OOM halinde CPU/offload smoke sonucu ve neden kaydedilir.
- Dört perturbation'ın feature distance tablosu ve motion-direction probe sonucu ortak schema'dan üretilir.
- Kullanıcı V-JEPA2'nin action-free pretraining çıktısı ile V-JEPA2-AC predictor'ı birbirine karıştırmadan data flow'u çizebilir.
- Full VideoMix22M pretraining veya paper accuracy reproduction denenmez.

## M5 — V-JEPA2-AC trajectory replay

### Amaç

Frozen visual encoder, block-causal predictor, 7D state/action, teacher forcing, autoregressive rollout ve latent energy ilişkisini resmî robot trajectory üzerinde görmek.

### Donanım kapısı

- Görünen CUDA VRAM `≥24 GB`: resmî V-JEPA2-AC GPU replay ve reduced CEM çalıştır.
- VRAM `<24 GB`: candidate `25`, refinement `2`, düşük batch CPU/GPU smoke dene; full AC inference bu aşamanın zorunlu kriteri değildir.
- OOM durumunda paper sonucu çalışmış gibi raporlanmaz. `status=resource_limited` ve son başarılı shape/step kaydedilir.

### Çalışma

- Yerelde `scripts/official_vjepa2_ac_replay.py --preflight-only` ile `franka_example_traj.npz` key/dtype/shape'lerini, pinned commit'i, bağımlılıkları, donanımı ve verilmişse AC checkpoint SHA-256'sını doğrula. Bu mod modeli yüklemez.
- Uygun donanım, `uv sync --extra ac` ve doğrulanmış checkpoint hazırsa aynı script'i `--run-replay` ile ayrı process'te çalıştır. Wrapper remote HEAD yerine pinned yerel kodu kullanır; current/goal latent, ground-truth/zero energy, 125 noktalı xyz grid, reduced CEM ve latency'yi JSON'a yazar.
- Bundled dosya yalnız iki frame ve bir bağımsız transition içerir. Bu nedenle gerçek bir shuffled-action kontrolü ve ölçülmüş iki-step rollout error üretilemez; çıktı ikisini de `unsupported` işaretler. Action eksenlerini karıştırmak geçerli shuffled baseline sayılmaz.
- `upstream/vjepa2/notebooks/energy_landscape_example.ipynb`, yayımlanmış görsel referanstır; 125-action grid ve reduced CEM gösterir ama scalar ground-truth/zero/shuffled kıyasını hazır olarak vermez.
- Donanım kapısının izin verdiği candidate/refinement ile reduced CEM çalıştır; latency'yi ölç.

### Çalıştırmadan önce tahmin et

- Ground-truth action her sample'da kesinlikle en düşük energy vermek zorunda mı?
- Teacher-forced one-step hata ile open-loop two-step hata arasında nasıl fark beklenir?
- Goal image viewpoint'i değişirse latent cost hangi yanlış davranışı teşvik edebilir?

### İncele

- Action sırası `[dx,dy,dz,droll,dpitch,dyaw,dgripper]` mı?
- State/action normalization checkpoint reçetesiyle aynı mı?
- Causal attention gelecekteki gerçek latent veya action'a sızıyor mu?
- Paper'ın candidate/horizon sonucu ile reduced smoke config'i açıkça ayrılmış mı?

### Kabul kriteri

- Dataset schema ve preflight kontrolleri geçer; checkpoint verilmişse dosya SHA-256'sı kaydedilir. Bu tek başına resmî AC inference başarısı sayılmaz.
- Donanım yeterliyse tek gözlenen transition için ground-truth action latent error'ı zero action error'ından küçüktür. Shuffled kıyası ancak en az iki bağımsız transition içeren ek veriyle kabul kriterine dönüşür.
- 1-step error raporlanır; bundled trajectory'de ölçülemeyen 2-step error ile modelin iki-adımlı CEM rollout'u birbirine karıştırılmaz.
- Reduced CEM'in candidate, elite, iteration, horizon ve wall-clock latency'si metadata'da bulunur.
- Donanım yetersizse inference sınırı dürüstçe kaydedilir ve M6'ya geçiş engellenmez.

## M6 — Eğitilebilir action-conditioned world model ve CEM/MPC

### Amaç

V-JEPA2-AC ilkelerini küçük, tamamen eğitilebilir ve closed-loop ölçülebilir bir planar reaching ortamında uygulamak.

### Ortam sözleşmesi

```text
observation:  uint8[64,64,3]
state:        normalized end-effector/agent state
action:       [dx,dy,0,0,0,0,0]
goal:         target RGB image + goal state yalnız evaluation için
success:      final normalized distance ≤ 0.08
```

Train ve held-out evaluation seed'leri ayrık manifestlerde saklanır. Planner goal state'i doğrudan kullanamaz; cost goal image latent'inden gelir.

### Model

1. Tiny video JEPA encoder, sentetik transition kliplerinde eğitilir.
2. Encoder dondurulur; optimizer parametre listesinde bulunmadığı assertion ile doğrulanır.
3. Action-conditioned predictor:
   - 4 block-causal Transformer layer;
   - hidden size `256`;
   - `8` attention head;
   - visual/state/action projections;
   - teacher-forcing L1 + two-step rollout L1.

### Zorunlu deneyler

- Ground-truth, zero ve shuffled action latent prediction error.
- 1-, 2- ve 4-step open-loop rollout error.
- Goal-image terminal latent energy.
- CEM: horizon `4`, candidates `256`, elites `32`, refinements `5`.
- MPC: her environment step'inde yeniden planla, yalnız ilk action'ı uygula.
- Aynı 100 held-out seed'de random-action baseline.

### Çalıştırmadan önce tahmin et

- Encoder frozen değilse dynamics predictor hangi kestirme çözümü öğrenebilir?
- One-step loss iyi olup four-step rollout kötü olduğunda planner davranışı ne olur?
- CEM elite mean/std iterasyonlar boyunca nasıl değişmeli?
- Terminal latent cost fiziksel mesafeyle monoton olmak zorunda mı?

### İncele

- Future token leakage'i engelleyen block-causal mask testten geçiyor mu?
- Action değişince aynı initial latent'ten farklı prediction çıkıyor mu?
- Elite seçimi minimum energy yönünde mi?
- Action bounds her sample ve mean update sonrasında uygulanıyor mu?
- MPC environment'tan yeni observation alıp yeniden encode ediyor mu?
- Evaluation seed'leri training datasetinde bulunuyor mu?

### Kabul kriteri

- Doğru action ortalama latent error'ı zero ve shuffled action'dan küçüktür.
- 1/2/4-step error ayrı ölçülür; tüm değerler finite'dır.
- Deterministic objective altında her refinement'ın **ham gözlenen elite-mean** enerjisi artmaz; history sonradan `min` ile yapay biçimde monotonlaştırılmaz. En iyi tek sequence ayrıca korunur.
- 100 held-out seed'de başarı oranı en az `%80`'dir.
- Aynı seed'lerde random-action başarı oranından en az `30` yüzde puan yüksektir.
- Final normalized goal distance eşiği `≤0.08` kod ve metric metadata'sında aynıdır.
- Planner yalnız ilk action'ı uygular ve her adımda yeniden planlar.
- Başarı sayısı/oranı ile Wilson `%95` güven aralığı aynı JSON özetinde bulunur.
- Bu eşikler RGB'den öğrenilen dondurulmuş tiny video-JEPA encoder'ını kullanan **visual** koşunun ana kabulüdür. Exact-XY `analytic` koşu dynamics/CEM mekanizmasını izole eden teşhistir ve visual kabulün yerine geçmez.

## M7 — Tez hocasına aktarılabilir sentez

### Amaç

Kod göstermeden ana fikri, kod açıldığında ise her iddiayı tensor, loss ve deneyle savunabilmek.

### Hazırlık

Ana çalışma rehberindeki tek anlatıyı şu üç kanıtla bağla:

1. I-JEPA masking/EMA ablation.
2. V-JEPA/V-JEPA2 temporal perturbation.
3. Action-conditioned predictor + CEM/MPC closed-loop sonucu.

Rehberdeki iki ASCII diyagramı ezberden yeniden çizebil: I-JEPA için full-image no-grad target kolu ile maskeli context/predictor gradient kolu; V-JEPA için `2×16×16` tubelet'ten `8×14×14` latent grid'e geçiş ve aynı spatial bloğun bütün temporal katmanlarda maskelenmesi. İkinci diyagramdaki full-time maskenin causal attention ile aynı şey olmadığını açıkla.

Sunum provasında aşağıdaki sorulara kısa ve teknik cevap ver:

- Neden pixel prediction yerine latent prediction?
- DINO neden JEPA değildir ve neden yine de bu rotadadır?
- I-JEPA'da target encoder neden tam görüntüyü görür?
- Target encoder neden gradient almaz; EMA ne yapar?
- V-JEPA neden varsayılan haliyle causal future model değildir?
- V-JEPA2 ile V-JEPA2-AC arasındaki eğitim sınırı nedir?
- Latent distance nasıl planning cost'a dönüşür?
- CEM neden differentiable planner gerektirmez?
- Oyuncak simülasyon sonucu gerçek robot deployment'ını neden garanti etmez?

### Kabul kriteri

- DINO → I-JEPA → V-JEPA → V-JEPA2 → V-JEPA2-AC zinciri 10 dakikada anlatılabilir.
- Her model için input, target, loss, teacher update, mask/view ve downstream rol doğru söylenir.
- `196` ve `1568` token hesabı tahtada yapılabilir.
- Paper sonucu, resmî checkpoint inference'ı ve bu projenin küçük deney sonucu açıkça etiketlenir.
- Kullanıcı en az üç failure mode açıklayabilir: collapse, non-causal leakage, rollout error accumulation; robotik aşamada camera/viewpoint ve safety sınırlarını da ekler.
- M6'nın başarı tablosu random baseline, 100 held-out seed, threshold ve confidence interval ile gösterilir.

## Son tamamlanma tanımı

Çalışma, bütün notebook'ların çalıştırılmış olmasıyla değil aşağıdaki zincirin kanıtlanmasıyla tamamlanır:

```text
görüntü patch'lerinden kararlı latent temsil
→ maskelenmiş image latent prediction
→ spatiotemporal tubelet representation
→ action'ın gelecek latent üzerindeki ölçülebilir etkisi
→ latent goal energy ile closed-loop action seçimi
```

Bu zincirin herhangi bir halkasında acceptance criterion sağlanmıyorsa sonraki büyük eğitim başlatılmadan önce o aşamanın data flow, shape, gradient ve baseline kontrollerine dönülür. Yeni Markdown raporu açmak yerine config, test, JSON/CSV metric ve mevcut rehberdeki ilgili açıklama düzeltilir.
