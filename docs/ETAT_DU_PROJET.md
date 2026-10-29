# État Actuel du Projet & Bilan Technique Sans Filtre

> Ce document distingue strictement :
> 1. **Ce qui est prouvé et fonctionnel** (code vérifié par tests unitaires et résultats empiriques reproductibles).
> 2. **Ce qui a donné un résultat neutre ou négatif** lors des benchmarks.
> 3. **Les bugs découverts et corrigés** au fil des expériences.
> 4. **Les limites actuelles non résolues** du modèle et de l'infrastructure.

---

## 1. Ce qui est prouvé et fonctionnel

### 1.1 Infrastructure logicielle et pipeline d'octets (`bytellm/`)
- **`ByteCodec` (`0..255` + 15 tokens de contrôle `256..270`, `VOCAB_SIZE = 271`)** : Encodage/décodage réversible sans perte pour n'importe quel fichier binaire ou texte UTF-8, masquage de perte sur les consignes (`encode_instruction`), et calcul vectorisé d'entropie locale de Shannon (`compute_shannon_entropy`).
- **Extracteur de code machine pur (`bytellm/extractor.py` & `scripts/build_elf_corpus.py`)** :
  - Parsing natif des en-têtes `ELF64` (`e_shoff`, `e_shentsize`, `e_shnum`, flag `SHF_EXECINSTR = 0x4`) et `PE/EXE` Windows (`IMAGE_SCN_CNT_CODE`) pour extraire uniquement les sections `.text` exécutables.
  - Déduplication exacte par empreinte `SHA-256` et exclusion garantie d'un répertoire de test externe (`--exclude`) pour empêcher toute fuite de données entre train et test.
- **Entraînement et Évaluation (`bytellm/training.py`, `bytellm/corpus.py`, `bytellm/cli.py`)** :
  - Support GPU avec précision mixte automatique (`torch.amp.GradScaler('cuda', enabled=use_amp)`).
  - Mise en cache des fichiers en RAM (`_file_cache`) et désactivation du calcul d'entropie à la volée lorsque `patch_size == 1` (**accélération $\times 10$ du DataLoader**).
  - Évaluation hors-distribution impartiale (`bytellm.cli evaluate`) par échantillonnage uniformément réparti (`_spread`) sur l'ensemble d'un corpus de test.

### 1.2 Capacités réelles apprises par le modèle sur code machine `x86_64`
Sur le domaine `x86_64` (`ELF`), après entraînement sur `66 à 82 Mo` de binaires systèmes (`/usr/bin`), les modèles `14M` et `58.7M` génèrent **sans aucune contrainte de grammaire (`use_grammar_constraints=False`)** des séquences de 256 octets qui :
1. **Se désassemblent à 99.6%–100% en instructions `x86_64` valides** sous `objdump` et Capstone (`0` opcode invalide `(bad)`).
2. **Ne sont pas des copies mémorisées** du corpus (vérifié par recherche de sous-chaînes exactes sur les 229 binaires d'entraînement).
3. **Respectent les conventions d'appel et de sécurité du compilateur GCC/Clang sous Linux** :
   - Prologues et épilogues conformes à l'ABI System V AMD64 (`push rbp; mov rbp, rsp`, sauvegarde ordonnée de `r15, r14, r13, r12, rbx`, restauration symétrique et `leave; ret`).
   - Alignement inter-fonctions par séquences `NOP` multi-octets canoniques (`0f 1f 80 00 00 00 00`, `66 0f 1f 84 00 00 00 00 00`).
   - Émission de l'instruction de protection matérielle Intel CET `endbr64` (`f3 0f 1e fa`) exactement en tête de nouvelle fonction après le padding `NOP`.
   - Initialisation du *Stack Canary* via le registre de segment Thread-Local Storage (`mov %fs:0x28, %rax; mov %rax, -0x38(%rbp); xor %eax, %eax`).
   - Synthèse d'idiomes de haut niveau : appels virtuels C++ via *vtable* (`mov rax, [rdi]; call qword ptr [rax+8]`), gestion de compteurs de référence et boucles courtes avec sauts conditionnels arrière alignés.

---

## 2. Résultats neutres ou négatifs (Ce qui n'a PAS surperformé)

1. **`ByteLLM` (`r=3, p=4`) vs `VanillaByteTransformer` sur texte `enwik8` (20 Mo, 3000 pas)** :
   - Une fois l'angle mort causal corrigé (`e91e0da`), `ByteLLM` atteint **`1.5968` BPB** contre **`1.5974` BPB** pour un `VanillaByteTransformer` de taille équivalente (`15.1M` vs `14.6M`).
   - **Verdict** : Sur du texte pur à cette échelle, la boucle de réflexion récurrente (`AbstractReasoningCore`, `reasoning_steps=3`) et le patcher (`patch_size=4`) **n'apportent aucun gain statistiquement significatif en BPB** (`-0.0006` BPB) tout en augmentant le temps d'entraînement de **`+29%`**.
2. **Contraintes de grammaire préfixes (`bytellm/constraints.py`) sur modèle non entraîné sur le domaine cible** :
   - Un modèle entraîné uniquement sur `enwik8` obtient **`0.0%` de fichiers valides** sur `wasm`, `midi`, `audio`, `compressed` et `asm` même avec `use_grammar_constraints=True`.
   - **Verdict** : Les classes actuelles `WasmGrammarConstraint`, `MidiGrammarConstraint`, `WavAudioGrammarConstraint` et `AsmGrammarConstraint` ne contraignent que l'en-tête magique initial (les 4 à 14 premiers octets). Elles ne suffisent pas à garantir la validité structurelle globale d'un fichier si le modèle n'a pas appris le format ou si un post-processeur ne répare pas les longueurs/checksums (comme le fait `repair_ipv4_checksums` pour `network`).

---

## 3. Historique des Bugs Découverts et Corrigés

| Commit / Étape | Problème découvert | Impact mesuré | Correctif appliqué |
|---|---|---|---|
| Initialisation | `nn.Embedding` initialisé en $\mathcal{N}(0,1)$ avec poids liés à `lm_head` | Perte initiale absurde à **`149.23 BPB`** | Initialisation $\mathcal{N}(0, 0.02)$ dans `_init_weights` → perte initiale à **`7.99 BPB`**. |
| `e91e0da` | Angle mort causal de 4 octets dans `DynamicBytePatcher` (décalage d'un patch entier masquant les 1 à 4 derniers octets aux couches 1..N) | Pénalité de **`+0.1645 BPB`** face à `VanillaByteTransformer` (`1.7613` vs `1.5974`) | Tronc principal maintenu à résolution 1 octet + branche de patch résiduelle contrôlée par porte `tanh(patch_gate)` → **`1.5968 BPB`**. |
| `9d75bda` | Ouverture disque à chaque `__getitem__` + calcul systématique de `compute_shannon_entropy` sur CPU | Goulot d'étranglement CPU sévère pendant `bytellm.cli train` sur GPU | Cache mémoire `_file_cache` dans `CorpusDataset` + `compute_entropy=False` par défaut quand `patch_size=1`. |
| `046fc83` | `generate_bytes` dépourvu de pénalité de répétition | Boucles d'octets périodiques en fin de génération longue (>150 octets) | Ajout du paramètre `repetition_penalty` dans `ByteLLM.generate_bytes` et `--repetition-penalty` dans le CLI. |
| `9ecb058` | `evaluate_per_domain` prenait `idxs[:512]` (les 512 premiers chunks consécutifs des 1–2 premiers fichiers alphabétiques) | Score de validation artificiellement bas (**`1.1985 BPB`**) par fuite entre binaires cousins (`g++` / `lto-dump`) | Échantillonnage uniformément espacé `_spread` sur tout le corpus + sous-commande `bytellm.cli evaluate` sur jeu externe `heldout_libs` (**`2.4825 BPB`** réel). |

---

## 4. Limites Actuelles Non Résolues

1. **Désalignement des sauts relatifs *forward* en `x86_64` (~67–73% d'échec sur les sauts avant)** :
   - Lorsqu'il génère `0f 84 9c 01 00 00` (`je +0x19c`), le modèle choisit l'offset `0x19c` avant d'avoir généré les 412 octets qui suivent. Comme les instructions `x86_64` ont une longueur variable de 1 à 15 octets, la cible `pc + 6 + 0x19c` tombe fréquemment au milieu d'une instruction multi-octets.
2. **Sous-exposition aux données (Chinchilla Starvation)** :
   - Nos runs de 3000 à 4000 pas (`batch=16, seq_len=512`) n'exposent le modèle qu'à **24.6–32.8 Mo d'octets**, soit moins d'un demi-passage (*epoch*) sur le corpus de 82.3 Mo et ~35× moins que le volume optimal pour 58.7M paramètres.
3. **Fenêtre de contexte limitée à 512 octets** :
   - 512 octets de code machine correspondent à environ 100–140 instructions `x86_64` (1 à 2 fonctions C moyennes). Le modèle ne peut donc pas voir simultanément l'en-tête ELF, la table PLT/GOT et les fonctions appelantes/appelées.
4. **Absence de tâche de décompilation bout-en-bout évaluée** :
   - Le générateur `build_c_asm_instruction_pairs` dans `bytellm/extractor.py` ne produit pour l'instant que 5 patrons synthétiques courts. Aucun entraînement ni benchmark sur de vraies paires `(code machine DWARF/objdump ↔ code C source réel)` n'a encore été mené.
