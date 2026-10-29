# Benchmarks Expérimentaux — ByteLLM

> **Politique de rigueur** : Ce document recense **exclusivement** les mesures réellement exécutées au cours du développement, avec leurs hyperparamètres exacts, leurs baselines de compression, les biais méthodologiques découverts en cours de route et les correctifs appliqués. Aucun chiffre n'est extrapolé ou estimé.

---

## Vue d'ensemble des 6 campagnes de benchmarks

| # | Campagne | Corpus | Modèles / Tailles | Métrique clé (Test / OOD BPB) | Conclusion principale |
|---|---|---|---|---|---|
| **1** | Fumée CPU (Synthétique) | Builders déterministes (6 domaines) | `1.14M` (25 pas) | Train BPB : `7.9905 → 5.4302` | Validation mécanique de la boucle d'entraînement et de l'initialisation $\mathcal{N}(0, 0.02)$. |
| **2** | Fumée `enwik8` (2 Mo) | Premiers 2 Mo de `enwik8` | `1.14M` (200 pas, Colab T4) | Test BPB : `3.3999` (vs `zlib_9` `3.1318`) | Le pipeline GPU AMP `fp16` converge sous l'entropie d'ordre 0 (`5.0197` BPB) en 15.6 s. |
| **3** | Ablation `reasoning_steps` (`enwik8` 20 Mo) | Premiers 20 Mo de `enwik8` | `15.43M` (`r=3` vs `r=1`, 3000 pas) | Test BPB : `1.7565` (`r=3`) vs `1.7892` (`r=1`) | Gain modeste de `-0.0327` BPB pour `r=3`, mais pénalité de vitesse de `-24%` (avant correctif du patcher). |
| **4** | Duel Architectural `Vanilla` vs `ByteLLM` (`enwik8` 20 Mo) | Premiers 20 Mo de `enwik8` (`seed=1`) | `Vanilla 14.56M` vs `ByteLLM 15.14M` (3000 pas) | Test BPB : `1.5974` (`Vanilla`) vs `1.7613` (avant fix) → **`1.5968`** (après fix `e91e0da`) | Découverte et correction de l'angle mort causal de 4 octets dans `DynamicBytePatcher` (`-0.1645` BPB récupérés). Sur texte pur à 3000 pas, `ByteLLM` égale `Vanilla` mais coûte `+29%` de temps. |
| **5** | Code Machine Natif `x86_64` (`ELF` `/usr/bin`) | 217 à 229 exécutables ELF (`66–82.3 Mo`) | `14.0M` (2000–3000 pas) & `58.7M` (4000 pas) | Décodage `objdump`/Capstone : **99.6%–100%** d'instructions `x86_64` valides | Apprentissage sans contrainte de l'ABI System V AMD64, prologues/épilogues, canaris TLS `%fs:0x28`, vtables C++. **Limite identifiée** : ~65–73% des sauts relatifs internes *forward* tombent au milieu d'une instruction multi-octets. |
| **6** | Scaling Sweep Contrôlé (`4M / 14M / 58M`) | Entraînement : 217 ELF (`82.3 Mo`)<br>Test OOD : **60 bibliothèques `.so` inédites** (`1.02 Mo`) | `4.62M`, `13.96M`, `58.69M` (3000 pas fixes = `24.58 Mo` vus) | Test OOD BPB :<br>• `4M` : **`2.6963`**<br>• `14M` : **`2.6165`**<br>• `58M` : **`2.5617`** (`2.4825` à 4000 pas) | Rendements décroissants forts sur budget d'octets fixe (`24.6 Mo` vus) : à partir de 14M, le modèle est **affamé en données/pas d'entraînement** (*data/step-starved*), pas en paramètres. |

---

## 1. Test de fumée initial et correction de l'initialisation des poids

- **Problème initial observé** : Lors du tout premier test à l'initialisation, `nn.Embedding` utilisait la loi normale standard $\mathcal{N}(0, 1)$ de PyTorch couplée par *weight tying* à `lm_head`, produisant des logits de variance $\approx d_{\text{model}}$ et une perte initiale aberrante de **`149.23 BPB`**.
- **Correctif appliqué** : Initialisation explicite $\mathcal{N}(0, 0.02)$ sur toutes les couches `nn.Linear` et `nn.Embedding` (`ByteLLM._init_weights`).
- **Résultat mesuré (CPU, 25 pas, `1,140,995` paramètres)** :
  - BPB initial : **`7.9905`** (exactement aligné avec une distribution uniforme sur 256 octets : $\log_2(271) \approx 8.08$ bits).
  - BPB entraînement final (25 pas) : **`5.4302`**.
  - *Avertissement* : Ce test valide uniquement la mécanique de descente de gradient sur un mini-lot synthétique.

---

## 2. Benchmark de fumée `enwik8` — Premiers 2 Mo (Google Colab T4)

- **Configuration** : `preset=tiny` (`d_model=128, n_layers=3, n_heads=4, patch_size=4, reasoning_steps=3`), `1,141,123` paramètres, `steps=200`, `batch_size=16`, `seq_len=256`, AMP `fp16`, découpage `1.8 Mo` train / `100 Ko` val / `100 Ko` test.
- **Baselines sur la tranche test de 100 Ko** :
  - Entropie d'ordre 0 : `5.0197 BPB`
  - `zlib -9` : `3.1318 BPB`
  - `lzma -9` : `2.8544 BPB`
  - `bz2 -9` : `2.7990 BPB`
- **Résultats `ByteLLM tiny` (200 pas, `819,200` octets vus, `15.6 s`)** :
  - `final_train_bpb` : **`3.3752`**
  - `val_bpb` : **`3.4312`**
  - `test_bpb` : **`3.3999`**
  - Débit d'inférence : **`197,721.6 octets/s`**

---

## 3. Ablation `reasoning_steps=3` vs `reasoning_steps=1` sur `enwik8` (20 Mo)

- **Protocole** : Premiers 20 Mo de `enwik8` (`18 Mo` train / `1 Mo` val / `1 Mo` test), `preset=small` (`d_model=384, n_layers=6, n_heads=6, patch_size=4`), `steps=3000`, `batch_size=32`, `seq_len=256` (`24,576,000` octets d'entraînement vus), GPU Tesla T4 avec AMP `fp16`.
- *Note historique* : Exécuté **avant** le correctif `e91e0da` du `DynamicBytePatcher`.

### Baselines de compression sur le Mo de test (`enwik8` tranche 19–20 Mo)
| Compresseur / Baseline | Bits par Octet (BPB) |
|---|---:|
| Distribution uniforme ($\log_2 256$) | `8.0000` |
| Entropie de Shannon d'ordre 0 | `5.1228` |
| `zlib` (niveau 9) | `2.7691` |
| `lzma` (niveau 9) | `2.1530` |
| `bz2` (niveau 9) | `2.0767` |

### Comparaison `reasoning_steps=3` vs `reasoning_steps=1`
| Métrique | `small` (`reasoning_steps=3`) | `small` (`reasoning_steps=1`) | Écart (`r=3` vs `r=1`) |
|---|---:|---:|---:|
| Paramètres | `15,431,043` | `15,430,275` | `+768` (portes ACT) |
| Temps d'entraînement (3000 pas) | `359.2 s` | `324.8 s` | `+10.6%` |
| `final_train_bpb` | `1.7699` | `1.7755` | `-0.0056` |
| `val_bpb` | **`1.9141`** | `1.9513` | **`-0.0372`** |
| `test_bpb` | **`1.7565`** | `1.7892` | **`-0.0327`** |
| Débit d'inférence (`forward_bytes_per_sec`) | `156,095.5 B/s` | **`204,755.3 B/s`** | `-23.8%` |

### Mesure de validité binaire hors-distribution (modèle entraîné uniquement sur `enwik8`)
Lorsque ce modèle entraîné à 100% sur du texte Wikipedia XML (`enwik8`) est évalué sur les 6 domaines binaires synthétiques (`wasm`, `midi`, `network`, `audio`, `compressed`, `asm`) :
- **`synthetic_heldout_bpb`** : **`12.2163 BPB`** (pire qu'une prédiction aléatoire uniforme à `8.0 BPB`, ce qui est attendu car le modèle assigne une probabilité quasi nulle aux octets non-ASCII).
- **Taux de validité sans contrainte (`unconstrained_rate`)** : **`0.0%`** sur tous les formats binaires (`100%` sur `text` UTF-8).
- **Taux de validité avec contrainte (`constrained_rate`)** :
  - `0.0%` sur `wasm`, `midi`, `audio`, `compressed`, `asm` — prouvant que les contraintes de `bytellm/constraints.py` ne forcent actuellement que les **préfixes/magic numbers** (les 4 à 14 premiers octets) et non la structure interne complète des sections (longueurs `ULEB128`, balises de fin de piste MIDI `FF 2F 00`, tailles `RIFF`, CRC32).
  - `100%` sur `network` (`IPv4/UDP`) uniquement grâce au post-processeur déterministe `NetworkPacketGrammarConstraint.repair_ipv4_checksums` qui recalcule les champs longueur et RFC 1071 après génération.

---

## 4. Duel Architectural : `VanillaByteTransformer` vs `ByteLLM` (`enwik8` 20 Mo, `seed=1`)

Pour vérifier si les composants spécifiques de `ByteLLM` (`BitPlaneByteEmbedding`, `LocalByteEncoder`, `DynamicBytePatcher`, `AbstractReasoningCore`) apportent un gain réel face à un Transformer standard à nombre de paramètres équivalent, nous avons implémenté `VanillaByteTransformer` (même backbone RoPE + SwiGLU + RMSNorm, `n_layers=8` au lieu de `6` pour égaliser les paramètres à ~14.6M–15.4M).

### Diagnostic et correction du bug d'angle mort causal (`commit e91e0da`)
1. **Avant correctif** : `ByteLLM` (`patch_size=4, reasoning_steps=3`) obtenait **`1.7613` BPB** sur test contre **`1.5974` BPB** pour `VanillaByteTransformer` (soit une dégradation de **`+0.1639` BPB**).
2. **Cause racine identifiée** : Dans `DynamicBytePatcher.downsample`, la séquence de longueur $T$ était découpée en patchs de $P=4$ octets, puis décalée d'un patch complet vers la droite dans `upsample` pour éviter toute fuite du futur. En conséquence, **les couches Transformer 1 à 5 ainsi que le `AbstractReasoningCore` étaient totalement aveugles aux 1 à 4 derniers octets** précédant la position courante $t$ ; seule la couche 0 avait vu l'octet $t-1$.
3. **Correctif (`e91e0da`)** : Le tronc principal conserve désormais la résolution exacte à l'octet ($1\times$), tandis que `DynamicBytePatcher` agit comme une branche contextuelle résiduelle pondérée par une porte scalaire initialisée à $0$ (`gate = tanh(patch_gate)`).

### Résultats comparatifs (`enwik8` 20 Mo, 3000 pas, `batch=32`, `seq_len=256`, `seed=1`)
| Modèle | Paramètres | Couches | Temps (3000 pas) | `final_train_bpb` | `val_bpb` | `test_bpb` | Débit Inférence |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`VanillaByteTransformer`** (`vanilla_small_s1`) | `14,561,280` | 8 | `511.5 s` | `1.5240` | `1.7546` | `1.5974` | `86,175.5 B/s` |
| **`ByteLLM` AVANT fix** (`r=3, p=4`) | `15,431,043` | 6 (+3 r) | `357.3 s` | `1.7913` | `1.9152` | `1.7613` | `157,833.6 B/s` |
| **`ByteLLM` APRÈS fix `e91e0da`** (`r=3, p=4`) | `15,136,137` | 6 (+3 r) | `661.7 s` | `1.5584` | **`1.7522`** | **`1.5968`** | `61,353.2 B/s` |

**Enseignement scientifique** :
- Le correctif `e91e0da` fait chuter le BPB test de `1.7613` à **`1.5968`** (`-0.1645 BPB`), effaçant intégralement le retard face au Transformer standard.
- En revanche, sur du texte `enwik8` à 3000 pas, `ByteLLM (r=3)` fait jeu égal avec `VanillaByteTransformer` (`1.5968` vs `1.5974` BPB, écart non significatif de `-0.0006` BPB) tout en étant **29% plus lent à l'entraînement** (car 6 couches + 3 pas de raisonnement récurrent à pleine résolution équivalent à 9 traversées de blocs par pas contre 8 pour `Vanilla`).
- Pour l'entraînement sur binaires natifs `x86_64`, nous avons donc adopté `patch_size=1` et `reasoning_steps=1` par défaut afin de maximiser le débit d'octets vus par seconde tout en conservant `BitPlaneByteEmbedding` et `LocalByteEncoder`.

---

## 5. Entraînement sur Exécutables Natifs `x86_64` (`ELF` `/usr/bin`)

### 5.1 Run 14M sur 229 exécutables `/usr/bin` (2000 pas)
- **Configuration** : `13,955,334` paramètres (`d_model=384, n_layers=6, n_heads=6, patch_size=1, reasoning_steps=1`), 229 binaires ELF `/usr/bin` (~66 Mo, `117,813` chunks train, `17,942` chunks val de 512 octets), `steps=2000`, `batch=16`, `seq_len=512`.
- **Perte validation interne (`/usr/bin` split CRC32)** : `val_bpb = 1.9793`.
- **Audit d'originalité et de désassemblage (`objdump -D -b binary -m i386:x86-64`)** sur génération **sans contrainte** (`use_grammar_constraints=False`) à partir de l'amorce `55 48 89 e5` (`push %rbp; mov %rsp, %rbp`) :
  - **Zéro copie intégrale** : la séquence générée de 68 octets a `0` occurrence exacte dans les 229 fichiers du corpus (seul le prologue standard de 20 octets est partagé par 27 binaires GCC/Clang).
  - **100% d'instructions `x86_64` valides** :
    - Sauvegarde des registres *callee-saved* dans l'ordre exact de l'ABI System V AMD64 (`push %r15; push %r14; push %r13; push %r12; push %rbx`).
    - Alignement de pile sur 16 octets (`sub $0x8, %rsp`) suivi d'un épilogue symétrique complet (`mov -0x8(%rbp), %rbx; leave; ret`).
    - Alignement inter-fonctions par NOP canonique de 9 octets (`66 0f 1f 84 00 00 00 00 00`), instruction de sécurité Intel CET **`endbr64`** (`f3 0f 1e fa`) au début de la fonction suivante, et mise en place du **Stack Canary TLS** (`mov %fs:0x28, %rax; mov %rax, -0x38(%rbp); xor %eax, %eax`).
  - **Défaut observé à 2000 pas** : Au-delà de ~150 octets générés sans `repetition_penalty`, le modèle 14M entrait dans une boucle répétant l'offset de pile `-0x38(%rbp)` (`0xc8`).

### 5.2 Run 14M sur sections `.text` pures + paires C/ASM (3000 pas)
- **Extraction `.text` pure (`bytellm/extractor.py`)** : Suppression des zéros de padding, tables de symboles et chaînes `.rodata` pour ne conserver que le code exécutable `SHF_EXECINSTR`.
- **Résultat qualitatif** : Disparition de la boucle dégénérée sur `-0x38(%rbp)`. Le modèle alloue des variables locales distinctes (`-0xb0(%rbp)`, `-0xb8(%rbp)`, `-0x38(%rbp)`, `-0xd0(%rbp)`), génère des comparaisons de caractère `/` (`cmpb $0x2f, (%r12)` pour tester un chemin Unix) et des multiplications rapides par 5 via `lea (%rax,%rax,4), %rcx`.

### 5.3 Run 58.7M sur 217 exécutables `/usr/bin` (82.3 Mo, 4000 pas) et Audit Capstone des sauts
- **Configuration** : `58,694,406` paramètres (`d_model=640, n_layers=10, n_heads=10, patch_size=1, reasoning_steps=1`), 217 exécutables ELF (`82.3 Mo`, `158,118` chunks train, `11,215` chunks val), `steps=4000`, `batch=16`, `seq_len=512`.
- **Audit statique rigoureux via Capstone (`cs_disasm`) sur deux séquences de 260 et 269 octets générées sans contrainte** :

| Critère d'audit statique | Séquence A (`temp=0.95`, 260 B) | Séquence B (`temp=0.80`, 269 B) |
|---|---:|---:|
| Octets décodés comme instructions `x86_64` valides | **259 / 260 (99.6%)** (1 octet tronqué en fin de buffer) | **269 / 269 (100.0%)** |
| Instructions `bad` / opcodes invalides | **0** | **0** |
| Idiomes C / C++ appris | Appels de vtable C++ (`mov rax, [rdi]; call qword ptr [rax+8]`), boucles `cmp rbx, r14; jne` | Décrémentation de compteur de référence (`mov rax, [rbx]; sub rax, 1; mov [rbx], rax`), *tail-call* `[A\] jmp rax` |
| **Sauts relatifs internes tombant sur un début d'instruction valide** | **3 / 9 (33.3%)** | **4 / 15 (26.7%)** |
| **Sauts relatifs internes tombant au milieu d'une instruction (*désalignés*)** | **6 / 9 (66.7%)** | **11 / 15 (73.3%)** |
| Boucle de répétition en fin de séquence (sans `repetition_penalty`) | Non | Oui (motif de 19 octets répété 6 fois à partir de l'octet `0x9b`) |

**Conclusion de l'audit d'assemblage** :
- Le modèle 58.7M maîtrise **la morphologie locale `x86_64`** (préfixes REX, ModR/M, SIB, ABI des registres, alignements NOP, idiomes vtable/refcount) et **les sauts courts arrière** (boucles locales `jne 0x99`, `je 0x62` qui pointent exactement sur le début d'une instruction précédente).
- En revanche, **l'arithmétique des sauts relatifs *forward* (`je +0x19c`) échoue ~2 fois sur 3** : en génération autorégressive octet par octet, le modèle émet le déplacement relatif 32 bits *avant* d'avoir généré les 100 à 400 octets suivants, et ne maintient pas un compteur exact de la somme des longueurs d'instructions futures pour atterrir pile sur une frontière d'opcode.

### 5.4 Biais d'échantillonnage de validation identifié et corrigé (`commit 9ecb058`)
- Lors de l'évaluation initiale du run 58.7M, `evaluate_per_domain` affichait un score anormalement bas de **`val_bpb = 1.1985`**.
- **Cause identifiée** : `evaluate_per_domain` prenait les 512 premiers chunks consécutifs de validation (`idxs[:512]`), qui provenaient tous des 1 ou 2 premiers fichiers ELF triés par ordre alphabétique (fortement redondants avec des binaires cousins du split d'entraînement comme `x86_64-linux-gnu-lto-dump` vs `g++`).
- **Correctif (`commit 9ecb058`)** : Remplacement par un échantillonnage uniformément réparti sur l'intégralité du corpus (`_spread`, 2000 chunks espacés couvrant tous les fichiers) et création d'un jeu de test externe strictement disjoint (`heldout_libs`).

---

## 6. Benchmark OOD Équitable & Scaling Sweep Contrôlé (`4M` vs `14M` vs `58M`)

Pour éliminer tout biais de fuite entre exécutables proches de `/usr/bin`, tous les modèles ont été évalués sur un corpus **hors-distribution (OOD)** constitué de **60 bibliothèques partagées inédites** (`/usr/lib/x86_64-linux-gnu/*.so*`, `128,575` chunks au total, évalués sur `2,000` chunks uniformément répartis soit **`1,021,919` octets évalués**) via la commande :

```bash
python -m bytellm.cli evaluate --ckpt <checkpoint.pt> --corpus heldout_libs --max-chunks 2000
```

### 6.1 Comparaison des checkpoints 14M (2000 pas) et 58.7M (4000 pas) sur `heldout_libs`
| Checkpoint | Paramètres | Pas d'entraînement | Octets d'entraînement vus | `heldout_libs` BPB | Loss (nats) | Perplexité Octet ($2^{\text{BPB}}$) |
|---|---:|---:|---:|---:|---:|---:|
| `run_binaires/bytellm.pt` | `13,955,334` | `2,000` | `16.38 Mo` | **`2.7036`** | `1.8740` | `6.5145` |
| `run_large_exe/bytellm.pt` | `58,694,406` | `4,000` | `32.77 Mo` | **`2.4825`** | `1.7207` | `5.5885` |

### 6.2 Scaling Sweep Contrôlé à budget de calcul/données constant (`3000 pas` = `24.58 Mo` vus)
Trois modèles entraînés exactement dans les mêmes conditions sur `corpus_massif_exe` (`217` ELF, `82.3 Mo`), `steps=3000`, `batch=16`, `seq_len=512`, `lr=6e-4`, `patch_size=1`, `reasoning_steps=1`, puis évalués sur les `1,021,919` octets de `heldout_libs` :

| Modèle | Architecture (`d_model`, `layers`, `heads`) | Paramètres Exacts | Pas (`Octets vus`) | `heldout_libs` BPB | Gain marginal $\Delta\text{BPB}$ | Ratio Paramètres |
|---|---|---:|---:|---:|---:|---:|
| **ByteLLM-4M** | `d=256, L=4, H=4` | `4,624,902` | `3,000` (`24.58 Mo`) | **`2.6963`** | — | $1.00\times$ |
| **ByteLLM-14M** | `d=384, L=6, H=6` | `13,955,334` | `3,000` (`24.58 Mo`) | **`2.6165`** | **`-0.0798 BPB`** | $3.02\times$ |
| **ByteLLM-58M** | `d=640, L=10, H=10` | `58,694,406` | `3,000` (`24.58 Mo`) | **`2.5617`** | **`-0.0547 BPB`** | $4.21\times$ (vs 14M) |
| *(Réf. 58M @ 4000 pas)* | `d=640, L=10, H=10` | `58,694,406` | `4,000` (`32.77 Mo`) | **`2.4825`** | **`-0.0793 BPB`** *(+1000 pas)* | $1.00\times$ (vs 58M) |

### Analyse de la loi d'échelle (Scaling Law) observée
1. **Loi de puissance monotone vérifiée** : À nombre de pas constant (`3000`), la perte OOD diminue systématiquement avec la taille du modèle (`2.6963 → 2.6165 → 2.5617 BPB`).
2. **Sous-entraînement au sens de Chinchilla (*Data/Step Starvation*)** :
   - Tripler les paramètres de `4.6M` à `14.0M` rapporte `-0.0798 BPB`.
   - Quadrupler les paramètres de `14.0M` à `58.7M` à `3000` pas ne rapporte que `-0.0547 BPB`.
   - En revanche, ajouter simplement **1000 pas d'entraînement supplémentaires** (`+33%` d'octets vus, de `24.6 Mo` à `32.8 Mo`) au modèle `58.7M` fait baisser le BPB de `2.5617` à **`2.4825`** (`-0.0793 BPB`), soit **1.45× plus de gain que le quadruplement des paramètres de 14M à 58M**.
   - **Explication physique** : À `3000` pas (`batch=16, seq_len=512`), le modèle n'a vu que $24.58 \times 10^6$ octets ($\approx 0.42$ octet par paramètre pour le modèle 58.7M, alors que le ratio optimal de Chinchilla est d'environ $20$ tokens/octets par paramètre, soit $\sim 1.17\text{ Go}$ d'octets vus).
