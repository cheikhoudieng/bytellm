# Pistes d'Amélioration & Feuille de Route Technique

> Chaque piste ci-dessous répond directement à un **goulot d'étranglement chiffré** identifié lors de nos benchmarks (voir [`docs/BENCHMARKS.md`](BENCHMARKS.md) et [`docs/ETAT_DU_PROJET.md`](ETAT_DU_PROJET.md)). Elles sont classées par **rapport impact / coût d'implémentation**.

---

## Priorité 1 — Sortir de la famine d'octets (*Data & Step Scaling*)

### Constat empirique
Le *Scaling Sweep* contrôlé sur `heldout_libs` (60 bibliothèques `.so` inédites) a prouvé qu'à `3000` pas (`24.58 Mo` vus) :
- Passer de **14M à 58.7M paramètres** ($4.2\times$ plus gros) fait gagner **`-0.0547 BPB`** (`2.6165 → 2.5617`).
- Garder **58.7M paramètres** mais passer de **3000 à 4000 pas** (`+8.2 Mo` vus) fait gagner **`-0.0793 BPB`** (`2.5617 → 2.4825`).

### Actions concrètes
1. **Augmenter le budget d'octets vus de `25 Mo` à `500 Mo – 1 Go` avant d'augmenter les paramètres** :
   - Sur un modèle **14M ou 30M**, passer à `steps=25000`–`50000` avec accumulation de gradient (`grad_accum=4`, soit un batch effectif de `32 768` octets par pas) et `seq_len=1024`.
2. **Élargir et dédupliquer le corpus `.text` avec `scripts/build_elf_corpus.py`** :
   - Extraire uniquement les sections `.text` exécutables pures (`--pure-text`) depuis `/usr/bin`, `/usr/sbin`, `/usr/lib` et les paquets `.deb` du cache APT, en excluant strictement `heldout_libs` (`--exclude heldout_libs`) :
     ```bash
     python scripts/build_elf_corpus.py \
       --sources /usr/bin /usr/sbin /usr/libexec \
       --out corpus_pure_text \
       --exclude heldout_libs \
       --pure-text --target-mb 500
     ```
3. **Déduplication au niveau des fonctions ELF (`SHT_SYMTAB` / `.dynsym`)** :
   - Actuellement, la déduplication se fait par hash `SHA-256` de fichier entier. Or, tous les exécutables C partagent le même code d'amorçage CRT (`_start`, `__libc_csu_init`, `deregister_tm_clones`, `register_tm_clones`).
   - Extraire et dédupliquer les fonctions individuelles (via les entrées `STT_FUNC` de la table des symboles ou le découpage sur séquences `ret + NOP padding + endbr64`) éliminera le sur-apprentissage des prologues CRT et forcera le modèle à apprendre le cœur algorithmique varié.

---

## Priorité 2 — Résoudre le désalignement des sauts relatifs *forward* en `x86_64`

### Constat empirique
L'audit Capstone sur le modèle `58.7M` montre que **99.6% à 100%** des octets générés forment des instructions `x86_64` syntaxiquement valides, mais que **67% à 73% des sauts relatifs *forward* internes** (`74 rel8`, `0f 84 rel32`) atterrissent au milieu d'une instruction multi-octets située plus loin.

### 3 pistes techniques complémentaires

#### Piste 2A — Post-processeur d'accrochage d'offsets (*Jump Target Snapper*) via Capstone (Court terme, déterministe)
Tout comme `NetworkPacketGrammarConstraint.repair_ipv4_checksums` recalcule les sommes de contrôle RFC 1071 après génération, ajouter dans `AsmGrammarConstraint` une passe `repair_x86_64_relative_jumps(data: bytes) -> bytes` :
1. Désassembler le flux généré avec Capstone pour obtenir l'ensemble exact des adresses de début d'instruction Valides $A = \{a_0, a_1, \dots, a_k\}$.
2. Pour chaque saut conditionnel ou inconditionnel interne (`eb rel8`, `70..7f rel8`, `e9 rel32`, `0f 80..8f rel32`) dont la cible $t = \text{pc} + \text{len} + \text{rel}$ tombe dans le buffer mais $t \notin A$ :
3. Remplacer $\text{rel}$ par $a^* - (\text{pc} + \text{len})$, où $a^* = \arg\min_{a \in A} |a - t|$ est la frontière d'instruction valide la plus proche.
- *Avantage* : Zéro coût d'entraînement, garantit immédiatement **100% de cibles de sauts internes alignées**.

#### Piste 2B — Masquage de logits guidé par un décodeur d'instructions incrémental pendant `generate_bytes` (Moyen terme)
Enrichir `AsmGrammarConstraint` pour suivre l'automate de décodage `x86_64` en temps réel pendant la génération :
- Maintenir la liste des cibles de sauts *forward* promises $T_{\text{pending}} = \{t_1, t_2, \dots\}$.
- Lorsqu'on approche à moins de 15 octets d'une cible promise $t_i$ (distance $d = t_i - \text{pc} \in [1, 15]$), masquer dans `BinaryLogitProcessor` les opcodes dont la longueur d'instruction dépasserait $t_i$, ou favoriser une séquence d'alignement `NOP` exacte de longueur $d$ si un bloc basique vient de se terminer (`ret` / `jmp`).

#### Piste 2C — Objectif d'entraînement auxiliaire d'alignement d'instructions (Long terme, apprentissage)
Ajouter une tête linéaire légère `boundary_head: nn.Linear(d_model, 2)` pendant l'entraînement sur `.text` qui prédit si l'octet courant $b_t$ est le **premier octet d'une instruction `x86_64`** (étiquette fournie gratuitement par Capstone lors de la préparation du corpus). Cela oblige l'état caché du Transformer à maintenir explicitement le compteur de phase d'instruction (préfixe → opcode → ModR/M → SIB → déplacement → immédiat).

---

## Priorité 3 — Transformer les contraintes préfixes en automates d'états complets (`bytellm/constraints.py`)

### Constat empirique
Sur un modèle non spécialisé, `WasmGrammarConstraint`, `MidiGrammarConstraint` et `WavAudioGrammarConstraint` donnent `0.0%` de validité à l'audit `BinaryValidator.validate`, car elles ne vérifient que les 4 à 14 premiers octets (magic header) et laissent la fin du fichier tronquée ou incohérente.

### Actions concrètes
1. **`WavAudioGrammarConstraint`** : Ajouter `repair_wav_sizes(data: bytes)` qui met à jour les champs `ChunkSize` (octets `4..8` = `len(data) - 8`) et `Subchunk2Size` (octets `40..44` = `len(data) - 44`) et tronque sur un multiple de `BlockAlign` (2 octets en PCM 16-bit).
2. **`MidiGrammarConstraint`** : Suivre l'état `MTrk`, forcer l'émission de la séquence de fin de piste obligatoire `00 FF 2F 00` dans les 4 derniers octets, et recalculer l'entier `uint32` big-endian de longueur de piste aux octets `18..22`.
3. **`WasmGrammarConstraint`** : Suivre la machine à états des sections WASM (`id` croissant `1 → 3 → 7 → 10`, décodage `ULEB128` de la taille de section, et obligation de terminer le corps d'une fonction de la section Code par `0x0b` `end`).

---

## Priorité 4 — Évaluation statique automatisée du code machine généré

### Constat empirique
Le seul **BPB (Bits Par Octet)** ne suffit pas à juger de la qualité structurelle du code machine généré (un modèle peut avoir un bon BPB grâce aux prologues/NOPs tout en faisant des boucles de répétition ou des sauts désalignés).

### Action concrète
Intégrer dans `bytellm/benchmark.py` une suite `evaluate_asm_generation_quality(model, n_samples=50, length=256)` qui mesure automatiquement avec Capstone :
1. **`valid_instruction_byte_ratio`** : Proportion d'octets décodés sans erreur jusqu'à la fin du buffer.
2. **`internal_jump_alignment_rate`** : Proportion de sauts relatifs internes dont la cible `pc + len + rel` tombe exactement sur une adresse de début d'instruction.
3. **`stack_frame_balance_rate`** : Proportion de fonctions complètes générées (`push rbp ... ret`) où le nombre de `push` callee-saved correspond exactement au nombre de `pop` avant `leave`/`ret`.
4. **`distinct_4gram_ratio`** : Ratio de 4-grammes d'octets uniques sur la séquence générée (détecte immédiatement toute dégénérescence en boucle périodique).

---

## Priorité 5 — Vrai jeu de données Bi-Modal `(Code Machine ELF + DWARF ↔ Code C/Assembleur)`

### Constat empirique
L'entraînement actuel se fait à 99% sur des octets `.text` bruts non annotés ; `build_c_asm_instruction_pairs` ne contient que 5 patrons statiques de démonstration.

### Action concrète
Créer un script `scripts/build_decompiler_pairs.py` qui :
1. Compile des fichiers C open-source (ou extrait les symboles des binaires avec `objdump -d --no-show-raw-insn`) fonction par fonction.
2. Formate chaque fonction comme une paire d'instruction `ByteCodec.encode_instruction` :
   - **Dans le sens Compilation (`C/ASM → x86_64`)** : Consigne = code C ou assembleur de la fonction ; Cible = octets `.text` exacts de la fonction.
   - **Dans le sens Décompilation (`x86_64 → ASM/C`)** : Consigne = octets `.text` bruts de la fonction ; Cible = code assembleur annoté ou code C source correspondant.
