# JEPA Çalışma Rehberi

Bu rehberin amacı yalnızca bir checkpoint çalıştırmak değil; görüntü temsili öğrenmeden başlayıp eylem koşullu bir robotik **world model** kurmaya kadar uzanan düşünce zincirini kavramaktır:

```text
DINO → I-JEPA → V-JEPA → V-JEPA2 → V-JEPA2-AC → latent-space planning
```

Ana fikir şudur: Bir modelin geleceği veya eksik bir bölgeyi piksel düzeyinde yeniden üretmesi şart değildir. Tahmin, öğrenilmiş bir temsil uzayında (**latent/embedding space**) yapılabilir. Böylece model, kolayca değişen piksel ayrıntılarından çok nesne, hareket, geometri ve eylemin sonucu gibi daha soyut düzenlilikleri öğrenmeye zorlanır.

> Not: Bu rota akademik ve non-commercial kullanım varsayımıyla hazırlanmıştır. Kullanmadan önce her upstream deponun güncel lisansını ve model kartını ayrıca kontrol edin. Özellikle I-JEPA ve V-JEPA materyallerindeki CC BY-NC koşulları downstream kullanım açısından önemlidir.

## 1. Önce doğru kavramlar

### 1.1 Representation model ile world model aynı şey değildir

Bir **representation model**, gözlemi yararlı bir latent temsile dönüştürür:

\[
z_t = E(o_t).
\]

Bir **predictive representation model**, gözlemin saklanan kısmının temsilini görünür kısımdan tahmin eder. I-JEPA bunun görüntü, V-JEPA bunun video karşılığıdır. Fakat robotik açıdan güçlü anlamdaki bir world model ayrıca eylemin sonucu hakkında konuşabilmelidir:

\[
\hat z_{t+1} = F(z_{\le t}, s_{\le t}, a_{\le t}),
\]

burada \(s_t\) proprioceptive state, \(a_t\) action'dır. V-JEPA2-AC bu ikinci sınıfa yaklaşır. Bu ayrım çalışma boyunca korunmalıdır:

- I-JEPA ve V-JEPA: ağırlıklı olarak **action-free representation learning**.
- V-JEPA2: daha büyük ölçekte action-free video representation pretraining.
- V-JEPA2-AC: dondurulmuş görsel encoder üzerinde **action-conditioned latent dynamics**.

### 1.2 JEPA neyi tahmin eder?

JEPA, girdinin saklanan kısmını doğrudan piksel olarak üretmez. Görünür **context**'ten, saklanan **target** bölgenin target encoder tarafından üretilmiş latent temsilini tahmin eder:

\[
z_x = f_\theta(x, M_c), \qquad
y = \operatorname{sg}\!\left(f_{\bar\theta}(x)\right),
\]

\[
\hat y_j = g_\phi(z_x, M_j), \qquad
\mathcal L = \frac{1}{K}\sum_{j=1}^{K} d(\hat y_j, y[M_j]).
\]

- \(f_\theta\): gradient alan **context encoder**.
- \(f_{\bar\theta}\): gradient almayan **target encoder**.
- \(g_\phi\): target konumlarını da kullanan **predictor**.
- \(M_c\), \(M_j\): context ve target maskeleri.
- \(\operatorname{sg}\): stop-gradient/detach.
- \(d\): paper veya implementasyona göre L2, SmoothL1 ya da L1.

Target encoder, context encoder'ın üstel hareketli ortalamasıyla güncellenir:

\[
\bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta.
\]

\(\tau\) genellikle eğitim ilerledikçe `0.996`'dan `1.0`'a gider. Target encoder optimizer tarafından güncellenmez. Bu iki güncelleme mekanizmasını birbirine karıştırmamak gerekir.

### 1.3 Collapse neden izlenir?

Modelin her girdiye aynı vektörü üretmesi tahmin loss'unu anlamsız biçimde kolaylaştırabilir. Buna **representation collapse** denir. Yalnız eğitim loss'una bakmak bu sorunu yakalamaz. Bu nedenle laboratuvarlarda birlikte izlenecek ölçüler şunlardır:

- Boyutlar boyunca feature standard deviation.
- Örnekler arası mean cosine similarity.
- Covariance matrix'in effective rank'i.
- Frozen k-NN ve linear probe başarımı.

Bu metrikler tek başına semantik kalite garantisi değildir; fakat loss düşerken temsilin sabite çökmesini ayırt etmek için gereklidir.

## 2. DINO: JEPA değil, önemli bir öncül

[DINO paper](https://arxiv.org/html/2104.14294) ve [resmî DINO deposu](https://github.com/facebookresearch/dino), ViT'lerde self-supervised representation öğrenmenin önemli bir referansıdır. DINO bir JEPA değildir; iki farklı görünüm arasında olasılık dağılımı eşleştiren bir **self-distillation** yöntemidir.

Teacher ve student çıktıları sıcaklıkla ölçeklenmiş softmax dağılımlarına çevrilir:

\[
p_s(x) = \operatorname{softmax}(g_{\theta_s}(x)/T_s),
\]

\[
p_t(x) = \operatorname{softmax}((g_{\theta_t}(x)-c)/T_t),
\]

ve student, teacher'ın farklı global crop'larda ürettiği dağılımı cross-entropy ile eşler:

\[
\mathcal L_{DINO} = -\sum_k p_t^{(k)}(x_g)\log p_s^{(k)}(x_v).
\]

Teacher ağırlıkları EMA ile güncellenir ve teacher tarafında stop-gradient vardır. **Centering** ve **sharpening**, farklı collapse türlerini engellemeye yardım eder. Student global ve local crop'ları görür; teacher yalnız global crop'ları görür.

DINO'dan JEPA'ya taşınacak kavramlar:

- EMA teacher/student asimetrisi.
- Stop-gradient ile hedef üretme.
- ViT patch token'larının semantik yapı taşıması.
- Self-supervised temsili k-NN veya frozen probe ile değerlendirme.

Taşınmaması gereken yanlış eşdeğerlik:

- DINO, bir hedef bölgenin konuma koşullu latent'ini tahmin etmez.
- DINO'nun temel objective'i view/crop invariance'dır.
- I-JEPA'da predictor ve spatial mask bilgisi, çözülmesi gereken tahmin probleminin parçasıdır.

## 3. I-JEPA: görüntü içinde latent tahmin

Kaynaklar: [I-JEPA paper](https://arxiv.org/html/2301.08243) ve [resmî I-JEPA deposu](https://github.com/facebookresearch/ijepa).

### 3.1 Veri akışı

Bir görüntü \(224\times224\), patch boyutu `16` olduğunda:

\[
14\times14 = 196 \text{ patch token}
\]

elde edilir. Resmî akışın kavramsal sırası şöyledir:

```text
image
 ├─ target encoder (tam patch dizisi, no-grad)
 │    └─ encoder çıktısından target blokları seç → y
 └─ context mask uygula → context encoder
      └─ görünür context token'ları + target konum bilgisi
           → predictor → ŷ

loss(ŷ, stop_gradient(y))
context encoder + predictor: optimizer step
target encoder: EMA update
```

Gradient ve maske sınırını birlikte okuyun:

```text
224×224 image, patch=16 → 14×14 konum grid'i

image
 ├─ target block'ları T1..T4 context'ten çıkar
 │      C ∩ (T1∪…∪T4)=∅; target-target overlap olabilir
 │          ↓
 │   görünür C → context encoder [grad] → predictor(position T) → ŷ ─┐
 │                                                                  │
 └─ tam image → target encoder [no-grad, EMA] → full [N,D]           │
                                      └─ çıktıda T1..T4 gather → y ──┤
                                                                     ▼
                                                         loss(ŷ, stop-grad(y))

optimizer: context encoder + predictor
EMA:       target ← τ·target + (1−τ)·context
```

Şema ViT/16'nın `14×14=196` öğretici grid'ini gösterir. Resmî pretrained I-JEPA ViT-H/14, aynı `224×224` görüntüde `16×16=256` token ve `D=1280` üretir; bu iki shape birbirine karıştırılmamalıdır.

Kritik ayrıntı: Target encoder'ın girdisine yalnız hedef patch'ler verilmez. Encoder tam görüntüyü işler; target blokları encoder çıktısından seçilir. Böylece target temsil, çevresindeki görüntü bağlamıyla zenginleşir. Context tarafında target bölgeler kaldırılır; aksi halde predictor saklanan bilgiyi doğrudan görebilir.

### 3.2 Multi-block masking neden önemlidir?

Paper'daki temel reçete:

- Bir görüntüden çoğunlukla dört target block.
- Her target için yaklaşık `0.15–0.20` scale ve `0.75–1.5` aspect ratio.
- Büyük ve tek parçalı bir context bölgesi için yaklaşık `0.85–1.0` scale.
- Context ile target'lar ayrık; farklı target'ların kendi aralarında örtüşmesi mümkündür.

Buradaki oranlar doğrudan “görüntünün tam olarak yüzde X'i maskelenir” şeklinde toplanamaz. Target'lar örtüşebilir, örnekleme grid üzerinde yuvarlanır ve context-target çıkarma adımı gerçek görünür token sayısını değiştirir.

Büyük semantic target'lar modelin yalnız kenar/doku gibi yerel ipuçlarıyla işi çözmesini zorlaştırır. Dağıtılmış context ise eksik bölgeyi nesne ve sahne düzeyinde kestirebilmek için yeterli bilgi bırakır.

### 3.3 Paper ile kod arasındaki önemli fark

I-JEPA paper objective'i squared L2 biçiminde açıklar. Resmî eğitim kodunda ise target özelliklerine normalization uygulanıp `smooth_l1_loss` kullanılır. Pratik akış şu şekilde okunmalıdır:

\[
y \leftarrow \operatorname{LayerNorm}(y), \qquad
\mathcal L_{code} = \operatorname{SmoothL1}(\hat y, y).
\]

Bu bir dipnot değil, reproducibility açısından önemli bir farktır. Bir deney “paper objective” ve “official code recipe” seçeneklerinden hangisini kullandığını metadata içinde açıkça belirtmelidir.

### 3.4 I-JEPA neyi kanıtlamaz?

I-JEPA'nın görünmeyen bir görüntü bölgesini latent uzayda tahmin edebilmesi, onun zamansal veya eylem koşullu bir world model olduğu anlamına gelmez. Statik görüntüde “eksik semantik bilgiyi tamamlama” güçlü bir giriş noktasıdır; fakat robot kontrolü için zaman, action, state ve çok adımlı rollout hâlâ eksiktir.

## 4. V-JEPA: latent tahmini videoya taşımak

Kaynaklar: [V-JEPA paper](https://arxiv.org/html/2404.08471) ve [resmî V-JEPA deposu](https://github.com/facebookresearch/jepa).

### 4.1 Tubelet tokenization

Video, yalnız spatial patch'lere değil **spatiotemporal tubelet**'lere ayrılır. Örneğin:

```text
input:       [B, C, T, H, W] = [B, 3, 16, 224, 224]
tubelet:     2 × 16 × 16
token grid:  8 × 14 × 14
token count: 1568
ViT-L dim:   1024
output:      [B, 1568, 1024]
```

Bir token artık tek bir görüntü patch'ini değil, iki kare boyunca uzanan küçük bir uzay-zaman hacmini temsil eder.

### 4.2 Video masking

V-JEPA'da spatial block, temporal eksenin tamamı boyunca uzatılarak bir **tube mask** oluşturur. Ana reçete iki maske ailesini birlikte kullanır:

- Yaklaşık sekiz küçük/short-range spatial block.
- Yaklaşık iki büyük/long-range spatial block.
- Birlikte yüksek, yaklaşık `%90` masking oranı.

Bu oran maske örtüşmeleri ve örnekleme ayrıntıları nedeniyle yaklaşık değerdir. İki maskeyi aynı encoded target üzerinde kullanmak target encoder hesabını amorti eder.

Tubelet ve full-time maske akışı:

```text
[B,3,16,224,224]
        │ Conv3D / tubelet 2×16×16
        ▼
 [B,8,14,14,D] ── flatten ─→ [B,1568,D]
      time 0   time 1             time 7
        ┌───┐    ┌───┐              ┌───┐
mask S: │ S │ == │ S │ ==  ...  ==  │ S │
        └───┘    └───┘              └───┘
                 aynı spatial blok bütün temporal katmanlarda maskeli

non-causal V-JEPA: görünür token'lar geçmişten ve gelecekten gelebilir
causal future lab: yalnız geçmiş context → gelecekteki target token'lar
```

Dolayısıyla “full-time tube mask” temporal eksenin tamamında aynı spatial konumu saklar; bu özellik tek başına attention akışını causal yapmaz.

V-JEPA objective'i resmî reçetede L1 latent prediction olarak düşünülebilir:

\[
\mathcal L_{VJEPA} = \frac{1}{|M|}\sum_{i\in M}
\left\|\hat y_i-y_i\right\|_1.
\]

### 4.3 En kritik nüans: V-JEPA nedensel bir gelecek tahmincisi değildir

V-JEPA videodan hareket bilgisi öğrenir, fakat pretraining maskesi genel olarak **non-causal**'dır. Context, klibin zaman eksenindeki hem erken hem geç karelerde maskelenmemiş tubelet'leri görebilir. Dolayısıyla normal V-JEPA sonucu şu şekilde yorumlanmamalıdır:

> “Model yalnız geçmişi görüp geleceği rollout ediyor.”

Daha doğru yorum:

> “Model, bir video klibindeki eksik uzay-zamansal temsilleri görünür uzay-zamansal bağlamdan tahmin ediyor.”

Bu yüzden frame reversal veya shuffle duyarlılığı V-JEPA'nın hareket temsilini araştırır; tek başına causal dynamics kabiliyetini kanıtlamaz.

## 5. V-JEPA2: ölçek ve daha güçlü video temsili

Kaynaklar: [V-JEPA2 paper](https://arxiv.org/html/2506.09985), [resmî V-JEPA2 deposu](https://github.com/facebookresearch/vjepa2) ve [resmî ViT-L model kartı](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).

V-JEPA2, V-JEPA'nın temel masked latent prediction fikrini korurken ölçek ve eğitim reçetesini ileri taşır:

- Çok daha büyük image/video karışık pretraining verisi.
- Yaklaşık 300M–1B parametre aralığında encoder'lar.
- Daha uzun eğitim ve progressive spatial/temporal resolution.
- Absolute 3D sine-cosine positional embedding yerine **3D RoPE**.
- Tubelet tabanlı video encoder mimarisi.

Bu aşamadaki ana öğrenme hedefi paper ölçeğini yeniden üretmek değil, aynı sürümlenmiş normalize hareket yörüngesi ve aynı perturbation tanımlarında V-JEPA ile V-JEPA2 feature'larını incelemektir. Checkpoint sözleşmeleri aynı değildir: V-JEPA v1 deneyi yörüngeyi `16×224×224`, V-JEPA2 fpc64 ise `64×256×256` olarak yeniden render/sample eder. Bu nedenle “aynı stimulus” ortak hareket yolu ve varyant anlamına gelir, birebir aynı ham tensor anlamına gelmez. Probe sonucu; preprocessing, frame sampling, pooling, checkpoint ve bu native çözünürlük/uzunluk farklarına duyarlıdır; bunlar metadata'da görünmeden skor farkı model üstünlüğü diye yorumlanmamalıdır.

### 5.1 İki aşamayı birbirine karıştırmayın

V-JEPA2 projesinde iki farklı eğitim problemi vardır:

1. **Action-free pretraining:** Çok büyük video/image verisinde genel görsel encoder öğrenilir.
2. **Action-conditioned post-training:** Görsel encoder dondurulur; robot state/action dizilerinden gelecek latent'i tahmin eden ayrı bir predictor eğitilir.

V-JEPA2 encoder checkpoint'ini çalıştırmak, V-JEPA2-AC world modelini çalıştırmak değildir. Robotik planning başarısı ikinci aşamadaki action-conditioned model ve planner'dan gelir.

## 6. V-JEPA2-AC: action-conditioned latent dynamics

V-JEPA2-AC, dondurulmuş görsel encoder'ın feature map'lerini action ve end-effector state ile birleştirir. Kavramsal akış:

```text
RGB observation o_t ─→ frozen V-JEPA2 encoder ─→ z_t
robot state s_t ────────────────────────────────┐
action a_t ─────────────────────────────────────┤
                                               ↓
                                block-causal AC predictor
                                               ↓
                                   predicted latent ẑ_t+1
```

`scripts/official_vjepa2_ac_replay.py` ayrı process'te pinned yerel V-JEPA2 kodunu kullanır. Varsayılan `--preflight-only` yolu model ayırmadan trajectory şemasını, kaynak commit'i, bağımlılıkları, donanımı ve checkpoint SHA-256'sını denetler. Açık `--run-replay` modu uygun donanım ve doğrulanmış checkpoint ile current/goal latent, ground-truth/zero action energy, xyz energy grid ve upstream reduced CEM'i çalıştırıp latency/config'i JSON'a yazar. Resmî örnek yalnız iki frame içerdiğinden bağımsız bir shuffled action veya ölçülmüş iki-step hata yoktur; bunları uydurmak yerine `unsupported` kaydedilir. Upstream `energy_landscape_example.ipynb` görsel referanstır, fakat ek scalar karşılaştırmalar workspace deneyidir.

Predictor eğitiminde iki tamamlayıcı loss vardır:

\[
\mathcal L_{TF} = \|P(z_t,s_t,a_t)-\operatorname{sg}(z_{t+1})\|_1,
\]

\[
\mathcal L_{roll} = \|P(\hat z_{t+1},s_{t+1},a_{t+1})-
\operatorname{sg}(z_{t+2})\|_1.
\]

İlki **teacher forcing**, ikincisi modelin kendi tahmininden devam ettiği kısa **autoregressive rollout**'tur. Rollout eğitimi, inference sırasında biriken dağılım kaymasını azaltmayı amaçlar; uzun ufukta hata birikmesini ortadan kaldırmaz.

### 6.1 Latent energy ve planning

Goal image \(o_g\) encoder'dan geçirilerek \(z_g=E(o_g)\) elde edilir. Aday action sequence'leri world model üzerinden rollout edilir. Basit terminal cost:

\[
J(a_{t:t+H-1}) =
d\left(\hat z_{t+H}(a_{t:t+H-1}), z_g\right).
\]

Cross-Entropy Method (**CEM**) bu cost'u minimize eden action dağılımını iteratif olarak günceller:

1. Mevcut Gaussian action dağılımından aday diziler örnekle.
2. Her adayı world model ile rollout et ve latent energy hesapla.
3. En düşük energy'li elite adayları seç.
4. Dağılımın mean/std değerlerini elite'lardan yeniden hesapla.
5. Birkaç refinement turundan sonra ilk action'ı uygula.
6. Yeni gözlemle yeniden planla (**receding-horizon MPC**).

Yalnız ilk action'ın uygulanması önemlidir; model hatası ve dış bozucular yeni gözlemde kısmen düzeltilir.

### 6.2 Robotik sınırlamalar

- Camera pose değişimi latent goal distance'i bozabilir.
- Aynı görsel hedefe giden farklı güvenli/güvensiz yolları tek terminal cost ayırt etmeyebilir.
- Uzun rollout'ta model hatası birikir.
- CEM çok sayıda forward pass ister ve gerçek zamanlı kontrolü yavaşlatabilir.
- Eğitim dağılımı dışındaki nesne, ışık, robot ve action scale'leri güvenilir değildir.
- Latent yakınlık; collision avoidance, joint limit veya güvenlik garantisi değildir.

Bu nedenle gerçek robot entegrasyonunda preprocessing, coordinate transform, action/workspace clipping, collision kontrolü ve low-level controller ayrı güvenlik katmanları olmalıdır.

## 7. Modellerin karşılaştırması

| Model | Girdi | Öğrenilen hedef | Teacher/target güncellemesi | Temel loss | Mask/view yapısı | Action? | Doğru yorum |
|---|---|---|---|---|---|---|---|
| DINO | Image crops | Teacher probability distribution | EMA teacher + stop-grad | Cross-entropy | Global/local multi-crop | Hayır | View-invariant self-distillation |
| I-JEPA | Image patches | Masked target patch latent'leri | EMA target + stop-grad | Paper: squared L2; code: target LN + SmoothL1 | Multi-block spatial | Hayır | Görüntü içi konuma koşullu latent prediction |
| V-JEPA | Video tubelets | Masked spatiotemporal latent'ler | EMA target + stop-grad | L1 | Full-time short/long tube masks | Hayır | Non-causal video representation learning |
| V-JEPA2 | Image/video tubelets | Masked latent'ler | EMA target + stop-grad | Latent prediction | Ölçeklenmiş masking, 3D RoPE | Hayır | Büyük ölçekli genel video encoder |
| V-JEPA2-AC | Video + state + action | Gelecek visual latent | Frozen visual encoder; AC predictor eğitilir | Teacher-forcing + rollout L1 | Block-causal sequence | Evet | Action-conditioned latent dynamics ve planning |

## 8. Paper–kod eşlemesinde kontrol listesi

Paper okumak ile kodu doğru okumak aynı iş değildir. Her deneyde aşağıdakiler açıkça doğrulanmalıdır:

- **Tensor order:** Laboratuvar standardı `[B,T,C,H,W]`, bazı upstream video kodları `[B,C,T,H,W]` ister.
- **Mask location:** Target mask encoder girdisinde mi, encoder çıktısında mı uygulanıyor?
- **Normalization:** Target feature'a LayerNorm uygulanıyor mu?
- **Distance:** L2, SmoothL1 ve L1 birbirinin eşanlamlısı değildir.
- **EMA schedule:** Sabit momentum mu, `0.996 → 1.0` schedule mı?
- **Context leakage:** Target token'ları context tarafından görülebiliyor mu?
- **Causality:** Model yalnız geçmişi mi görüyor, yoksa tüm klibin görünür token'larını mı?
- **Pooling:** Mean pooling, attentive probe veya başka bir yöntem mi?
- **Frame sampling:** Frame sayısı, stride ve fps aynı mı?
- **Checkpoint:** Context encoder mı, target encoder mı, predictor dahil mi?
- **Resolution/position encoding:** Checkpoint'in beklediği resolution ve positional encoding ne?

İlk incelemede izlenecek kod aileleri aşağıdadır. Bir upstream refactor olduğunda dosya adını ezberlemek yerine sembolün veri akışındaki rolü izlenmelidir.

| Kavram | I-JEPA'da başlangıç noktası | V-JEPA/V-JEPA2'de başlangıç noktası |
|---|---|---|
| Training ve EMA sırası | `src/train.py` | ilgili `app/.../train.py` |
| Context/target ViT | `src/models/vision_transformer.py` | `src/models/vision_transformer.py` |
| Predictor | aynı model modülü içindeki predictor oluşturucuları | `src/models/predictor.py`, AC için `src/models/ac_predictor.py` |
| Mask üretimi | `src/masks/multiblock.py` | `src/masks/multiseq_multiblock3d.py` veya pinlenmiş sürümdeki 3D mask karşılığı |
| Tensor gather/repeat | `src/utils/tensors.py` | `src/utils/tensors.py` |
| AC planning örneği | — | `notebooks/utils/world_model_wrapper.py` ve `notebooks/utils/mpc_utils.py` |

Onaylanan kaynak pinleri:

| Upstream | Commit |
|---|---|
| `facebookresearch/dino` (submodule değil; resmî Torch Hub immutable ref) | `7c446df5b9f45747937fb0d72314eb9f7b66930a` |
| `facebookresearch/ijepa` | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` |
| `facebookresearch/jepa` | `51c59d518fc63c08464af6de585f78ac0c7ed4d5` |
| `facebookresearch/vjepa2` | `9a061fffe395573a2ee61dadcd3cf5baa188e3f2` |

Üç JEPA upstream pini çalışma reposunun submodule kayıtlarında kaynak gerçekliği olarak tutulur. DINO burada submodule değildir; `official_dino_features.py` resmî Torch Hub kodunu tablodaki immutable Git ref'inden ister ve bunu run provenance'a yazar. Bu rehberdeki açıklama, pinlenmiş kodun yerine geçmez. V-JEPA2'nin daha yeni branch'lerinde V-JEPA2.1 dosyaları görülebilir; bu çalışma verilen V-JEPA2 paper'ı ve yukarıdaki pinle sınırlıdır.

## 9. Ortak laboratuvar sözleşmesi

Deneylerin ortak veri biçimi:

```text
RawVideo.frames       uint8[B,T,H,W,3]
VideoBatch.frames     float32[B,T,C,H,W]
VideoBatch.actions    float32[B,H,A] | None
VideoBatch.states     float32[B,H,S] | None

VisualEncoder.encode(frames, mask=None)
    -> Latents[B,N,D]

ActionWorldModel.rollout(z0, actions, states)
    -> PredictedLatents[B,H,N,D]

CEMPlanner.plan(current_rgb, goal_rgb, state, action_bounds)
    -> PlanResult(first_action, sequence, energy, elite_mean, elite_std)
```

Robot action sırası:

```text
[dx, dy, dz, droll, dpitch, dyaw, dgripper]
```

Öğretici planar ortam yalnız `dx,dy` kullanır; diğer boyutları sıfır doldurur. Bu, oyuncak ortamı gerçek Franka controller'ıymış gibi göstermeden arayüzü erkenden sabitlemeyi sağlar.

Karşılaştırılabilir feature dosyaları en az şunları taşımalıdır:

```text
features:       float32[num_samples, num_tokens, dim]
labels:         int64[num_samples] veya görev metadata'sı
sample_ids:     string[num_samples]
frame_indices:  int64[num_samples, T]
model_id, source_commit, checkpoint_sha256 veya model_revision,
preprocessing, pooling, seed
```

## 10. Compute açısından gerçekçi beklenti

Bu çalışmanın amacı paper-scale sonuç yeniden üretmek değildir.

- **Kaggle GPU:** Küçük I-JEPA/V-JEPA training, probe ve ablation'ların ana ortamı.
- **Colab:** Aynı kilitli bağımlılıklarla yedek ortam.
- **Apple M4 / 16 GB:** Kod okuma, unit test, küçük CPU/MPS smoke ve sığ inference. CUDA/NCCL varsayan upstream training entrypoint'leri doğrudan çalışmayabilir.
- **24 GB ve üzeri CUDA:** Resmî V-JEPA2-AC replay için tercih edilir; bunun altında azaltılmış candidate/batch veya yalnız smoke kabul edilir.

Beklenmemesi gerekenler:

- ImageNet-1K üzerinde tam I-JEPA pretraining.
- VideoMix2M/22M ölçeğinde V-JEPA/V-JEPA2 pretraining.
- Paper tablosundaki doğruluğu küçük veri ve ViT-Tiny ile aynen üretmek.
- Tek tüketici GPU'sunda paper'ın tam CEM candidate/latency profilini yakalamak.

Beklenmesi gerekenler:

- Gradient ve EMA akışını sayısal olarak doğrulamak.
- Masking tercihinin öğrenme sinyalini değiştirdiğini görmek.
- Temporal perturbation'ların video feature'larını etkilediğini ölçmek.
- Doğru action'ın gelecek latent'ini yanlış action'dan daha iyi açıkladığını göstermek.
- Küçük simülasyonda latent energy ile closed-loop planning yapmak.

Planar simülasyon iki yolu bilerek ayırır. `visual`, RGB transition kliplerinden tiny video-JEPA öğrenir, target encoder'ı dondurur ve goal-image latent cost ile planning yapar; M6 kabul ölçütlerinin ana yoludur. `analytic`, exact normalize XY latent kullanarak dynamics ve CEM'i görsel temsil hatasından ayıran teşhis kontrolüdür. Her iki koşunun episode özeti başarı oranına ek olarak Wilson `%95` güven aralığı üretir; analytic başarı visual world-model başarısı yerine sunulmaz.

## 11. Tez hocasına anlatılacak ana hikâye

Sunum, model isimleri listesi yerine tek bir problem ekseninde kurulabilir:

1. **Sorun:** Piksel prediction, dünyanın semantik olarak önemsiz ayrıntılarını modellemeye çok kapasite harcar.
2. **DINO'nun dersi:** EMA teacher ve self-distillation ile ViT, labelsız veriden semantik patch feature'ları öğrenebilir; fakat objective esasen view invariance'dır.
3. **I-JEPA'nın adımı:** Görünür context'ten saklanan target'ın pikselini değil, target encoder latent'ini konuma koşullu tahmin eder.
4. **V-JEPA'nın adımı:** Aynı ilke tubelet'lerle videoya taşınır ve hareket içeren representation öğrenilir; pretraining causal dynamics değildir.
5. **V-JEPA2'nin adımı:** Veri, model ve spatiotemporal eğitim ölçeklenir; genel bir video encoder elde edilir.
6. **V-JEPA2-AC'nin adımı:** Encoder dondurulur, state/action koşullu predictor gelecek latent'leri öğrenir.
7. **Kontrol köprüsü:** Goal image latent'i bir energy/cost tanımlar; CEM aday action dizilerini world model üzerinden değerlendirir, MPC her gözlemde yeniden planlar.
8. **Bizim kanıtımız:** Masking ablation, temporal perturbation ve action-conditioned CEM deneyleri bu zincirin üç ayrı iddiasını küçük fakat ölçülebilir biçimde sınar.
9. **Sınır:** Temsil kalitesi ve latent prediction başarısı gerçek robot güvenliği veya uzun ufuklu görev başarısı anlamına gelmez.

Bu hikâyeyi anlatabilen ve her ok için bir tensor şekli, objective ve deneysel kanıt gösterebilen kişi JEPA'yı yalnız “çalıştırmış” değil, araştırma düzeyinde anlamaya başlamış demektir.

## 12. Birincil kaynaklar

- Assran ve diğerleri, [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)](https://arxiv.org/html/2301.08243); [resmî kod](https://github.com/facebookresearch/ijepa).
- Caron ve diğerleri, [Emerging Properties in Self-Supervised Vision Transformers (DINO)](https://arxiv.org/html/2104.14294); [resmî kod](https://github.com/facebookresearch/dino).
- Bardes ve diğerleri, [Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)](https://arxiv.org/html/2404.08471); [resmî kod](https://github.com/facebookresearch/jepa).
- Assran ve diğerleri, [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/html/2506.09985); [resmî kod](https://github.com/facebookresearch/vjepa2); [ViT-L checkpoint/model card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256).
