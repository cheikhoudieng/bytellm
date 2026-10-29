# ByteLLM — Modèle de Langage Natif sur Octets Bruts (0–255)

**ByteLLM** est une architecture **Tokenizer-Free** expérimentale qui opère directement sur le flux d'octets bruts (`0x00`–`0xFF`, soit `0..255`) sans aucun découpage sous-lexical de type BPE (*Tiktoken*, *SentencePiece* ou *WordPiece*).

## Documentation Technique Complète (`docs/`)

- 📊 **[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)** : Tous les résultats chiffrés vérifiables (`enwik8` 2 Mo & 20 Mo, ablation `reasoning_steps`, duel `VanillaByteTransformer` vs `ByteLLM` avant/après correction de l'angle mort causal, désassemblage `x86_64` `/usr/bin`, audit Capstone des sauts relatifs et *Scaling Sweep* contrôlé `4M / 14M / 58M` sur 60 bibliothèques `.so` inédites).
- 🔍 **[`docs/ETAT_DU_PROJET.md`](docs/ETAT_DU_PROJET.md)** : Bilan sans filtre de ce qui est prouvé, de ce qui n'a pas surperformé les baselines, des bugs corrigés et des limites actuelles.
- 🛠️ **[`docs/PISTES_AMELIORATION.md`](docs/PISTES_AMELIORATION.md)** : Feuille de route priorisée (passage à l'échelle des données/pas selon Chinchilla, alignement des sauts relatifs *forward* en `x86_64`, automates de contraintes complets et paires de décompilation).

## Caractéristiques Clés

1. **Zéro Tokenizer BPE (`ByteCodec` 0–255 + Décomposition Bit-Plane & Nibbles)**
   - Chaque octet `0..255` est projeté via une table exacte enrichie par ses **8 plans binaires** ($b_0 \dots b_7 \in \{-1, +1\}^8$) et ses **quartets haut/bas (nibbles 4-bit)**.
   - Confère un biais inductif immédiat pour l'arithmétique binaire, les bits de continuation MSB (`LEB128` WebAssembly, `VLQ` MIDI), les masques de sous-réseau IPv4, le chiffrement XOR et les sommes de contrôle.

2. **Patching Dynamique Guidé par l'Entropie de Shannon (`DynamicBytePatcher`)**
   - Inspiré des architectures **Byte Latent Transformer (BLT)** et **MegaByte**.
   - Encode les octets via des convolutions causales multi-échelles (`kernel = 3, 5, 9` pour capturer les mots `uint16`, `uint32`, `uint64`), regroupe les octets en patchs latents pondérés par la densité d'information locale, et les décompresse vers la résolution octet avec un **décalage causal strict** garantissant zéro fuite du futur.

3. **Raisonnement Abstrait Renforcé (`AbstractReasoningCore` — Système 2)**
   - **Boucle de Réflexion Latente Récurrente (Adaptive Computation Time / Pondering)** : itère sur les états latents avec une porte d'arrêt apprise (*Halting Gate*) qui alloue dynamiquement plus de cycles de calcul aux transitions algorithmiques complexes.
   - **Registres de Mémoire de Travail (`Working-Memory Registers`)** : banque de registres latents interrogeables par attention croisée permettant de maintenir des invariants abstraits (accumulateurs de checksum, profondeur de pile WASM, harmonie MIDI).
   - **Scratchpad Explicite (`<THINK_START>...<THINK_END>`)** : génération d'une trace de raisonnement intermédiaire en octets avant l'émission du binaire final.

4. **Ingestion & Génération de Fichiers Binaires Natifs Contraints par Grammaire (`BinaryLogitProcessor`)**
   - **Code Machine & Assembleur (`ASMBuilder` / `AsmGrammarConstraint`)** : fonctions natives `x86_64` / `ARM64` et exécutables `ELF64` (`\x7fELF`).
   - **WebAssembly (`WASMBuilder` / `WasmGrammarConstraint`)** : modules `.wasm` v1 (`\x00asm\x01\x00\x00\x00`), encodage/décodage `ULEB128`, sections Type/Function/Export/Code.
   - **Paquets Réseau (`PCAPBuilder` / `NetworkPacketGrammarConstraint`)** : paquets `IPv4/UDP` et captures `.pcap` avec calcul et réparation automatique des sommes de contrôle **RFC 1071** (complément à un sur 16 bits).
   - **Fichiers Compressés (`CompressedBuilder`)** : flux `GZIP` (`\x1f\x8b`, vérification `CRC32` + `ISIZE`) et `ZLIB` (`\x78\x9c`, vérification `Adler32`).
   - **Musique MIDI (`MIDIBuilder` / `MidiGrammarConstraint`)** : fichiers Standard MIDI (`MThd` + `MTrk`), quantités à longueur variable (`VLQ`), événements de tempo et polyphonie.
   - **Audio Brut (`WAVBuilder` / `WavAudioGrammarConstraint`)** : synthèse harmonique polyphonique avec enveloppe ADSR en format natif `RIFF/WAVE` PCM 16-bit.

## Utilisation Rapide

### Exécuter la suite de tests complète
```bash
python tests/test_bytellm.py
```

### Lancer la démonstration de bout en bout (Entraînement + Génération Binaire)
```bash
python -m bytellm.cli demo --out-dir artifacts_out --steps 30
```

### Inspecter et valider n'importe quel fichier binaire
```bash
python -m bytellm.cli inspect artifacts_out/symphony_sketch.mid
python -m bytellm.cli inspect artifacts_out/module.wasm
python -m bytellm.cli inspect artifacts_out/capture.pcap
```

## Texte, instructions et corpus réel

Le texte (UTF-8, toutes langues, code source) passe par le même flux d'octets (domaine `TEXT`).
Les paires « consigne → fichier » utilisent `ByteCodec.encode_instruction`; la consigne est masquée de la perte.

```bash
python -m bytellm.cli train --corpus mes_donnees --instructions instr.jsonl \
  --weights "text=0.4,wasm=0.1,midi=0.1" --steps 5000 --d-model 512 --layers 8
python -m bytellm.cli generate --ckpt run_out/bytellm.pt --domain midi \
  --instruction "Une mélodie triste en la mineur" --out melodie.mid
```

## Benchmarks (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cheikhoudieng/bytellm/blob/main/notebooks/ByteLLM_benchmarks.ipynb)

```bash
python -m bytellm.benchmark --preset small --steps 3000 --max-mb 20 --amp
python -m bytellm.benchmark --preset small --steps 3000 --max-mb 20 --amp --reasoning-steps 1 --domains 0   # ablation
```

La suite mesure : BPB sur **enwik8** (comparé à zlib/bz2/lzma et à l'entropie d'ordre 0), ablation du raisonnement
récurrent, taux de fichiers valides **avec et sans** contraintes de grammaire, et débit.

## État honnête du projet

Le code (codec, validateurs, contraintes, pipeline, benchmarks) est testé. Le **modèle**, lui, est un prototype :
aucune performance de référence n'est revendiquée tant que les benchmarks ci-dessus n'ont pas été exécutés à l'échelle.
Les fichiers valides produits par `demo` viennent des builders déterministes et des contraintes, pas de ce que le modèle a appris ;
seul le taux « sans contrainte » mesure l'apprentissage réel.
