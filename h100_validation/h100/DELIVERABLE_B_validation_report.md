# FoundationScale H100 Validation Report

**Validation state:** **NOT COMPLETE.**  
**Hardware scope:** one measured node with eight H100 GPUs.  
**Distributed scope:** single-node execution only.  
**Current Phase 3 state:** zero of eight required legs executed through the framework.

## 1. Verdict

**No GPU has run under FoundationScale’s shipped launch path, and the genuine eight-GPU Phase 3 run has not occurred.** GPU execution is blocked by three open technical blockers — the launcher’s host-side GPU probe, the missing build-produced launcher input, and the unbounded adjudicator interface — together with an authorized cluster credential that is not available to the report author [M3; #142; #146; BLOCKERS]. One manually launched, eight-rank communication probe did execute in a container on eight H100s, but it bypassed the framework launch path and measured NCCL transport only; it is not model loading, training, checkpointing, resume, evaluation, or Phase 3 evidence [M12; E.3]. What is established is narrower: a measured one-node estate, singularity as the only working container runtime there, an HF/FSDP-capable image with no usable Megatron engine, controlled remediation of several environment and launch defects, a reproducible 20-stage source build with all 31 generated tests passing under the accessible credential, and synthetic execution of the checkpoint adjudicator [M1; M2; M14; CUR-BUILD; ADJ]. All real training-path claims remain UNMEASURED or BLOCKED.

## 2. Validation chain

| Link | Status | Evidence cited | What would move it to MEASURED |
|---|---|---|---|
| Environment | MEASURED | One visible partition, one node, and eight H100s were enumerated; partition limit was seven days; singularity was present, apptainer and enroot absent; scheduler commands were present; image contents and module visibility were inspected in-container. Scope is one accessible credential and one node only [M1; M2; M11; M14]. | No additional measurement is needed for the declared one-node scope. Before being presented as final production evidence, rerun the account-dependent checks under the authorized credential. |
| Model Loading | BLOCKED | Configuration recognition was executed for five checkpoints under transformers 5.5.0, and Deliverable E records four of seven architecture/meta instantiations under transformers 4.57.1. No full BF16 model was materialized by the framework; Phase 3 leg L2 did not run. The launch, adjudicator, and credential blockers prevent the intended execution [M9; M17; E.1; #142; #146; BLOCKERS]. | Run L2 through the final launcher: load the selected model in BF16, record the loader class, produce a parameter count greater than zero, and report the dtype histogram with denominators. |
| Dataset | UNMEASURED | No dataset inventory, tokenizer run, stream, batch, or token count has executed. Phase 3 deliberately uses synthetic input, so even a completed Phase 3 does not measure this link [PHASE3-SPEC]. | Execute a real dataset gate that resolves the dataset root, binds it, reads more than zero examples, counts batches and tokens, prints per-rank denominators, and carries a planted unreadable-input MUST_FIRE. |
| Distributed Training | BLOCKED | A manually launched eight-rank NCCL baseline measured `28.0` against a reference of `28.0` on eight ranks, at 420.8 GB/s for 256 MiB. That probe did not use FoundationScale. The framework rank path has not run [M12; E.3]. | Run Phase 3 through the generated launcher. L3 must prove all eight ranks contribute; L4 must prove per-rank shard counts sum to the unsharded denominator; L5 must prove optimizer execution and loss movement. |
| Checkpointing | BLOCKED | The adjudicator executed against four synthetic checkpoint shapes, including one complete synthetic checkpoint; it has never adjudicated a checkpoint written by a real training run. The writer and adjudicator also disagree over both framework checkpoint directory formats. Phase 3 L6 did not run [ADJ; #150]. | Land the format and launcher fixes, then run L6 against model plus optimizer state and report files, bytes, shards, manifest entries, and corruption controls, each with denominators greater than zero. |
| Resume Training | BLOCKED | Executed gates now forward both resume variable names through the shared runtime path, and exact-name negative controls were observed. Whether the trainer reads them and proves restored state equals saved state has never been exercised [M20]. | Run L7: observe the comparator failing on a deliberately perturbed tensor first, then restore into a fresh model and optimizer and report bitwise equality as `k of N`, with `N > 0`. |
| Evaluation | BLOCKED | A comparator was demonstrated on synthetic logits, but no saved-model/full-model evaluation equivalence has executed. Phase 3 L8 did not run [ADJ]. | Run L8 or its successor under `eval()` and `no_grad` with a fixed seed and fixed input; compare a nonzero number of output elements before save and after resume and report the maximum absolute delta. |

## 3. What was measured, and how

### 3.1 Executed measurements

| Measurement | Status | Result and denominator | Method and scope |
|---|---:|---|---|
| Estate inventory | MEASURED | Exactly one visible partition; one node; eight H100 devices; partition walltime seven days; queue empty at observation time [M1; M11]. | Scheduler inspection on the measured estate. The result is time- and credential-scoped. |
| Container runtimes | MEASURED | Singularity present: 1/1 required runtime on this estate. Apptainer absent: 1/2 opcional-runtime probes. Enroot absent: 2/2 optional-runtime probes. Scheduler launch tools present: 2/2 [M2]. | Host command discovery plus scheduler inspection. The source-level enroot arm cannot be executed on this estate. |
| Container image contents | MEASURED | Five Megatron-family import probes reported unavailable for module-absence or permission reasons: 5/5. The HF training stack components were present and importable: transformer engine, PEFT, accelerate, datasets, FlashAttention, transformers, and torch — 7/7. Distributed checkpoint import was additionally successful [M14; PHASE3-SPEC]. | In-container import probes. The image is an HF/FSDP image, not a usable Megatron image. |
| Mount boundary | MEASURED | In a two-arm in-container test, the asset filesystem was absent without a bind and visible with a declared bind: 2/2 arms differentiated. The bind test observed three visible entries in the asset root [M16]. | Controlled singularity execution differing only in the declared bind. Raw paths are replaced by `<asset-root>` here. |
| Torch-leak control | MEASURED | With a synthetic host torch stub and the containment control unset, the container imported the stub: 1/1 MUST_FIRE. With containment enabled and the same stub still planted, the container imported container torch: 1/1 MUST_PASS [M10]. | Two-arm synthesized leak experiment. Natural leakage cannot be produced by the currently accessible credential, so synthesis is explicitly part of the measurement. |
| Torch provenance detector | MEASURED | Detector reported four of four checks passing; sabotaging it to accept everything produced a nonzero exit: 1/1 MUST_FIRE [M7]. | Executed detector self-test plus sabotage. This validates discrimination only within the detector’s measured prefix scope; see M13. |
| Environment boundary | MEASURED | After remediation, all ten required names crossed into the container: 10/10. All four host-only names were held: 4/4. A suffix-similar name did not leak: 1/1 [M6]. | Executed against the real spliced runtime files. |
| Bind-path population | MEASURED | An empty derivation was refused: 1/1 MUST_FIRE. Four declared inputs reduced to three unique bind paths: 1/1 MUST_PASS. The eight bind gates B1–B8 were run and green: 8/8 [M18]. | Executed patch gates, including the gate requiring both runtime arms to consume the same declared array. |
| Resume-name forwarding | MEASURED | Both required resume names crossed: 2/2 MUST_PASS. Three forbidden or malformed names were refused: 3/3 MUST_FIRE. The harness extracted 23 names, so the controls had a nonempty denominator [M20]. | Executed allowlist controls. This establishes forwarding only, not trainer-side restore correctness. |
| Bidirectional environment drift | MEASURED | The gate initially went red on two real directions, D1 and D3; all three detectors then went red under planted violations: 3/3 MUST_FIRE. Remediation gates were 6/6 green [M21]. | Executed against the real launcher/backend pair before and after remediation. |
| Estate-path policy | MEASURED | Sixteen estate-literal matches over four lines were replaced by configured roots; the negative, boundary, and second-root controls all discriminated. The emitted result was 0 matches from a stated pre-patch denominator of 16 [M22]. | SOURCE-LEVEL scan plus execution of the generated matcher. |
| Launch-topology composition | MEASURED | The topology patch was 9/9 green. Eleven composer rows were executed: four MUST_PASS and seven MUST_FIRE. The live misconfiguration — eight GPUs with one rank — was refused: 1/1 [#124; M23]. | Executed composer controls in a clean environment. This does not remove the separate current launcher blockers M3 and #142. |
| Communication baseline | MEASURED | In the one manually launched eight-rank H100 run, world size and device count were both 8. The collective produced 28.0/reference 28.0. The 256 MiB reduction completed in 1.12 ms at 420.8 GB/s [M12]. | In-container `torchrun`, bypassing FoundationScale. It is a transport baseline only. |
| NCCL plugin controls | MEASURED | The image-default plugin configuration faulted once: 1/1. Disabling the network plugin gave the collective PASS once: 1/1. Poisoned data and a crashing rank were both detected: 2/2 MUST_FIRE. Socket and no-IB variants were also executed but do not generalize the fix [E.3]. | Eight-rank communication probe. Log inspection counted 24 collective-channel lines, 17 NVLS lines, and 24 P2P/CUMEM lines; these are line counts, not channel-count claims beyond the logged categories. |
| Root resolver on estate | MEASURED | Eight of eight independently measured estate rows reproduced the resolver prediction: 8/8. Two real ambiguous roots were refused with candidate counts intact; two shallowest-depth controls resolved [E.2; #133]. | The resolver executed inside the declared container because the host interpreter is Python 3.6.8. |
| Synthetic checkpoint adjudication | MEASURED | Four synthetic inputs produced four distinct observed outcomes: nonexistent directory→refuse 96; empty directory→UNMEASURED 95; junk-only directory→UNMEASURED 95; complete synthetic checkpoint→all A1 through A7a green. Dropping one shard moved the report from 8/8 green to 6/8 green [ADJ]. | Executed against synthetic artifacts only. No result covers checkpoints produced by training. |
| Current build | MEASURED | All 20 build stages completed green: 20/20. Generated suites passed 31/31 tests in 0.08 s. Seven generated artifacts parsed cleanly: 7/7. The latest record reports zero blocklist hits but does not state its file denominator, so that zero must not be given a wider scope [CUR-BUILD]. | Full source rebuild under the accessible credential. Stages stop on the first red gate. |
| Current-build determinism | UNMEASURED | An earlier 17-stage build produced byte-identical hashes for five artifacts across two rebuilds: 5/5 at that historical state. No double-build hash comparison is recorded for the current 20-stage, seven-artifact build [E.5; CUR-BUILD]. | Requires two consecutive complete current builds plus hashes for all seven artifacts. |

### 3.2 SOURCE-LEVEL findings

A SOURCE-LEVEL result is a read, search, parse, or generated-file inspection. It is not runtime evidence.

| Source-level result | Status | Evidence and denominator |
|---|---:|---|
| Original runtime concentration | MEASURED | The original framework source contained 58 enroot references and zero singularity references. After splicing, functions increased from ten to thirteen, no function was lost, all twelve distinct `fs_*` calls resolved among thirteen definitions, and shell parsing was clean [M2; M8]. |
| Original generated-file integrity | MEASURED | Post-splice reference inventory contained 46 singularity references and the higher enroot count did not remove that runtime’s source-level arm. This establishes source construction, not execution on this estate [M8]. |
| Time-limit guard | FAILED | Two checked values were admitted, including values below and above the seven-day partition limit: 2/2 admissions. The final pattern matches any value containing a colon, so the guard cannot reject [M4]. |
| Iteration-budget variables | FAILED | Two exported budget names had zero readers across three generated artifacts: 0/3 artifacts consumed them. The stated 20-iteration probe and save-at-five behavior were therefore decorative at the time of measurement [M5]. |
| Torch-provenance prefix scope | FAILED | The container has eight legitimate site directories: 8/8 enumerated. The detector pins one of them and excludes the venv default package directory. A rebuild installing torch into the venv would be rejected by a detector whose premise was measured only against the older layout [M13]. |
| Mount-declaration scope | FAILED | The R4 implementation verified readability of each declared path, not recursive materialization beneath it. The narrower measured claim is path readability for the declared set [E.4; #132]. |
| Checkpoint writer/adjudicator contract | FAILED | The writer emits one timestamp-style directory format while the adjudicator accepts neither of the two produced formats: cross-validation abstained on 2/2 framework checkpoint formats [ADJ; #150]. |
| Parser coverage | MEASURED | The original parser did not recognize single-line function definitions. After repair, the backend had thirteen parsed definitions and the launcher six, with zero unparsed definitions; both suites reran green [M19]. |
| Missing build-produced launcher input | BLOCKED | The shipped launcher sources a filename that the build has never produced. A fix is generated but not landed, so execution remains blocked [#142]. |
| Adjudicator interface containment | BLOCKED | The required adjudicator knob is neither containment-checked nor bound into the container. A fix is generated but not landed [#146]. |
| Engine model-loading seam | SOURCE-LEVEL | SOURCE-LEVEL inspection establishes that the framework is written against a Megatron-Bridge-style seam, while measured imports establish that the estate image exposes HF transformers/accelerate/FSDP instead. No HF-engine equivalence run has established the replacement seam [M14]. |
| Toolchain-version overlap | UNMEASURED | The image probes and Phase 3 specification report transformers 5.5.0, while Deliverable E’s architecture/meta matrix reports 4.57.1. The two matrices must remain separate toolchain-scoped observations until the same image/toolchain result is re-recorded [M14; E.1; PHASE3-SPEC]. |

### 3.3 Model evidence boundary

| Claim | Status | Measured denominator |
|---|---:|---|
| Configuration acceptance, transformers 5.5.0 | MEASURED | Five of five checkpoints resolved through `AutoConfig`: 5/5 [M17]. |
| Architecture/meta instantiation matrix, transformers 4.57.1 | MEASURED | Four of seven candidates built successfully, two were refused as unrecognized, and one remained UNMEASURED: 4/7, 2/7, and 1/7 respectively [E.1]. |
| Full-weight model loading | UNMEASURED | Zero full-weight models have been materialized through FoundationScale: 0/7 against the Deliverable E inventory. |
| Second-architecture full loading | UNMEASURED | Zero second architectures have had weights loaded. AutoConfig and meta-device instantiation do not count as loading weights [E.1; M17]. |
| Mixture-of-experts coverage | UNMEASURED | The originally selected two-model pair contained two dense models: 0/2 exercised the MoE expert-FQN logic [M9]. |
| Out-of-tree Gemma-4 registration | BLOCKED | Two out-of-tree Gemma-4 candidates were refused by the measured Deliverable E toolchain until a declared architecture-registration seam exists: 2/2 blocked for that seam [E.1; #119]. |
| Dense multimodal interpretation | MEASURED | Two green rows over the original dense-multimodal pair would certify only dense multimodal loading. They carry no MoE and no non-multimodal evidence [M9]. |

## 4. Findings-ledger summary

The normalized report below contains **29 finding records**. The denominator is these 29 records, not every test case, gate row, or historical note. `MEASURED` in this section means that the finding and its closure or retraction were themselves observed; it does not mean the corresponding training capability is operational.

| Status | Count |
|---|---:|
| MEASURED | 15/29 |
| FAILED | 9/29 |
| BLOCKED | 5/29 |
| UNMEASURED | 0/29 |
| **Total** | **29/29** |

### 4.1 Open Phase 3 blockers

| Blocker | Status | Full statement | Required closure evidence |
|---|---:|---|---|
| M3 — host-side GPU probe | BLOCKED | Executing the generated launcher’s GPU probe used the host interpreter, which is Python 3.6.8 and has no torch import. The probe received an empty visible-device count and correctly refused with exit 96 before container initialization. The source location of the defect is SOURCE-LEVEL, but the refusal was executed [M3]. | Move the measurement into the declared runtime or declare a different load-bearing rank source. Execute the launcher across both container-runtime adapters. MUST_PASS with eight devices; MUST_FIRE with zero, nonnumeric, and mismatched device reports. |
| #142 — missing launcher input | BLOCKED | The shipped launcher sources a filename that the build never produces. A fix has been generated but has not landed, so no producer/consumer proof exists [#142]. | Land the correction, rebuild from scratch, and require the consumed filename to be one of the build’s declared outputs. MUST_FIRE when the filename is absent. |
| #146 — adjudicator knob not contained | BLOCKED | The required checkpoint-adjudication knob is neither containment-checked nor bound into the container. Its in-container visibility therefore has no produced evidence. A fix has been generated but has not landed [#146]. | Declare the knob in the build/control vocabulary, bind or forward it through both runtime adapters, execute host-side and in-container probes, and drill absent and overwritten values. |
| Requester-held credential | BLOCKED | All current measurements are under the accessible credential. Production push and execution are blocked until the authorized credential is supplied through the secure channel. Account-scoped facts — permissions, user-site contents, quotas, and module visibility — are not transferable [CONFOUNDS; BLOCKERS]. | Supply the credential, push the immutable artifacts, rerun credential-scoped environment and torch-provenance controls, then execute Phase 3. |

### 4.2 Remaining finding records

| Finding | Status | Measured disposition |
|---|---:|---|
| #107 host-torch leak | MEASURED | Reproduced by synthesis in one arm and suppressed by containment in the second: 2/2 controlled arms discriminated [M10]. |
| #115 environment forwarding | MEASURED | Ten required variables crossed: 10/10; four host-only variables were held: 4/4 [M6]. |
| Provenance detector control | MEASURED | Four of four checks passed; the always-accept sabotage went red: 1/1 [M7]. Its narrowed prefix scope remains a separate FAILED finding, M13. |
| #117 mount plane | MEASURED | Bind gates were 8/8 green. Empty derivation refused once: 1/1. Four inputs produced three unique binds once: 1/1 [M18]. |
| #122 resume environment | MEASURED | Both resume names cross: 2/2; three negative-name controls refused: 3/3; extraction denominator was 23 names [M20]. |
| #116 drift gate | MEASURED | Both forward and reverse defects went red before remediation; all three detectors fired under planted violations: 3/3 [M21]. |
| #123 estate literals | MEASURED | Sixteen blocklisted literals over four lines were removed and replaced by configured roots: 16→0 [M22]. |
| #124 rank topology | MEASURED | Nine topology gates passed: 9/9; eleven composer controls discriminated, including refusal of the eight-GPU/one-rank configuration: 11/11 [M23]. |
| #133 model-root resolver | MEASURED | Twelve generated tests passed: 12/12; eight estate predictions reproduced: 8/8 [E.2]. |
| #129 NCCL plugin | MEASURED | The default configuration faulted once and the selected remediation passed once. Both negative communication controls were observed: 2/2 [E.3]. |
| #136 generated intermediates | MEASURED | The missing intermediate stage was incorporated and the ungenerated intermediate is removed/rebuilt each run [E.5]. |
| #137 build-input partition | MEASURED | Three non-artifact inputs were relocated and documented with hashes. Moving one away made the build red and restoring it made the build green: 1/1 control [E.5; #137]. |
| #138 host-interpreter assumption | MEASURED | The host has Python 3.6.8. The estate gate executed in the declared container and reproduced 8/8 rows [E.2; #138]. |
| #86 orphan suite | MEASURED | The build now executes the generated suite; current execution is 31/31. A missing suite is a declared build failure unless the explicit waiver is used [E.5; CUR-BUILD]. |
| M19 parser blind spot | MEASURED | Previously unparsed single-line functions are now counted. Backend and launcher parsing ended with zero unparsed definitions [M19]. |
| M4 time-limit guard | FAILED | Two tested limits, below and above the measured seven-day maximum, were both admitted: 2/2 [M4]. |
| M5 iteration budget | FAILED | Two budget variables had zero consumers across three artifacts: 0/3 [M5]. |
| M13 provenance prefix | FAILED | The detector pins one of eight legitimate container site directories: 1/8 covered by its assumed prefix [M13]. |
| #132 mount-claim narrowing | FAILED | Verification established readability of declared roots only, not content beneath them. Claimed recursive materialization was not measured [E.4; #132]. |
| #119 architecture-registration seam | BLOCKED | Two out-of-tree checkpoints were refused by the measured matrix and require a declared extension seam: 2/2 blocked [E.1]. |
| #150 checkpoint-format disagreement | FAILED | The cross-validation leg abstains on both framework-generated checkpoint formats: 2/2 [ADJ; #150]. |
| #127 launch-document generator | FAILED | The generator had never produced the launch document and was observed red on its L2 gate while correctly refusing to write a false artifact [E.4; #127]. |
| Retracted matrix row | FAILED | A row claiming three configuration candidates could not be reproduced because its measured path was not recorded and the directory was not findable. The row was removed rather than corrected [E.1]. |
| M9 dtype prediction | FAILED | The SOURCE-LEVEL prediction that a missing dtype field necessarily produced a crash or silent FP32 load was refuted by execution under transformers 5.5.0, which normalizes the observed Qwen dtype [M15]. The Gemma branch remained untested because configuration resolution failed first. |
| M15 Gemma-family generalization | FAILED | A failure from one non-stock Gemma checkpoint was overgeneralized to the family/toolchain. Execution of five checkpoints found stock Gemma configurations loadable, and the claim was retracted [M15; M17]. |

## 5. Controls: how the detectors are known to work

A detector without an observed MUST_FIRE is itself UNMEASURED. A green gate without a failure control is not evidence.

### 5.1 Detectors with an observed MUST_FIRE

| Detector or gate | Status | Observed MUST_FIRE | Observed MUST_PASS | Scope limit |
|---|---:|---|---|---|
| Torch provenance | MEASURED | Always-accept sabotage exited nonzero: 1/1; synthetic host torch was imported by the leaked arm: 1/1 [M7; M10]. | Container torch was imported with the same leak still planted: 1/1 [M10]. | Prefix premise is only 1/8 site directories; M13 remains FAILED. |
| Environment allowlist | MEASURED | A suffix-similar resume name was refused: 1/1; three forbidden resume names were refused: 3/3 [M6; M20]. | Ten required variables crossed: 10/10; both required resume names crossed: 2/2 [M6; M20]. | Tests forwarding, not trainer behavior. |
| Bidirectional drift gate | MEASURED | Real forward and reverse defects went red before remediation; all three planted-detector drills fired: 3/3 [M21]. | Green after remediation and retained as a standing build gate [M21; CUR-BUILD]. | Both files must be regenerated and rechecked under the final build. |
| Bind-path population | MEASURED | Empty derivation was refused: 1/1; B7 requires both runtime arms to consume the shared array [M18]. | Four declared inputs collapsed to three unique bind paths: 1/1 [M18]. | Declared-path readability is narrower than recursive materialization [#132]. |
| Estate-root matcher | MEASURED | Refused paths outside every root and a sibling-prefix boundary case; both controls fired [M22]. | A path under the second allowed root resolved: 1/1 [M22]. | The root list must be supplied at launch; behavior with an omitted list is fail-closed by design but current runtime execution remains blocked. |
| Launch topology composer | MEASURED | Seven of eleven clean-environment rows were negative controls and all seven refused. The eight-GPU/one-rank composition was refused: 1/1 [M23]. | Four valid composition rows passed: 4/4 [M23]. | The composer’s callsite gate is source-level; no training ranks have been launched by the framework. |
| Public blocklist | MEASURED | A planted token-shaped string matched once from one planted insertion: 1/1 [E.5]. | The current build reports zero hits, but its latest file denominator is unstated [CUR-BUILD]. | The live pattern control is historical; rerun the planted control against all seven current artifacts. |
| Build-input partition | MEASURED | A planted undeclared file is rejected on every build [E.5]. | Four acknowledged input/output categories are accepted: 4/4 [E.5]. | It partitions declared files; it does not infer hidden reads. |
| Model-root resolver | MEASURED | Two real ambiguous estate roots were refused: 2/2, with candidate counts reported [E.2]. | Two shallowest-depth controls resolved: 2/2. Eight estate predictions reproduced: 8/8 [E.2]. | Configuration search only; no weight loading. |
| Communication detector | MEASURED | A poisoned rank contribution and a crashing rank both went red: 2/2 [E.3]. | The unpoisoned collective passed once: 1/1 [E.3]. | Manual transport probe, not framework training. |
| Checkpoint adjudicator, synthetic set | MEASURED | Nonexistent, empty, junk-only, and drop-one-shard cases refused or removed green legs. The mutation moved 8/8 green to 6/8 green [ADJ]. | One complete synthetic checkpoint made every A1–A7a leg green [ADJ]. | All four inputs were synthetic. The writer/adjudicator contract remains failed under #150. |
| Comparator probe, synthetic logits | MEASURED | A deliberately perturbed input changed the label: 1/1 [ADJ]. | Unmodified synthetic tensors compared bytewise over 4,874,240 of 4,874,240 elements [ADJ]. | This does not validate a real model or evaluation corpus. |

### 5.2 Required controls without an observed MUST_FIRE

All thirteen rows below are **BLOCKED or UNMEASURED**, not certified.

| Control object | Status | Why there is no observed MUST_FIRE |
|---|---:|---|
| Phase 3 L1 environment/provenance | BLOCKED | The script has not started. No training-run torch path, user-site setting, device count, or world-size control has fired. |
| Phase 3 L2 full model load | BLOCKED | No full weights have materialized. No zero-parameter, wrong-dtype, or unreadable-weight negative control has fired. |
| Dataset ingestion | UNMEASURED | No dataset reader or detector has executed. |
| Framework rank composition | BLOCKED | The final launcher has not launched training ranks. Raw NCCL controls do not measure this hookup. |
| Optimizer and loss detector | BLOCKED | No nonfinite-loss, no-parameter-update, or nondecreasing-synthetic-batch control has run on the real training loop. |
| Real checkpoint writer | BLOCKED | No real training checkpoint exists. The synthetic adjudicator controls do not measure writer integration. |
| Real resume comparator | BLOCKED | No fresh-model load of a real checkpoint has attempted restored `k of N`. Its deliberately corrupted-tensor control has not fired on real state. |
| Real evaluation comparator | BLOCKED | The synthetic comparator control exists, but no saved/reloaded full model has undergone a labeled evaluation-equivalence case. |
| #142 producer/consumer gate | BLOCKED | The filename-producing fix has not landed; no missing-file red has been observed on the final gate. |
| #146 knob-containment gate | BLOCKED | The containment fix has not landed; no absent-knob or overwritten-knob drill exists. |
| Architecture-registration seam | BLOCKED | The seam does not exist. No valid registration MUST_PASS or invalid registration MUST_FIRE has executed. |
| Final-credential portability | BLOCKED | Credential delivery has not occurred, so no leak, permission, or module-scope control has run under it. |
| End-to-end chain detector | BLOCKED | No combined Environment→Evaluation execution summary exists, and therefore no skipped-leg or denominator-zero MUST_FIRE has been observed. |

Additional detector-coverage gaps:

| Coverage object | Status | Boundary |
|---|---:|---|
| MoE expert-FQN handling | UNMEASURED | The originally usable model pair is dense: 0/2 MoE candidates. |
| Text-only model loading | UNMEASURED | Text-only CausalLM configurations can resolve, but no full text-only weights have loaded. |
| Multi-node NCCL | BLOCKED | The estate exposes one node. Inter-node collectives and node-failure recovery cannot be exercised there. |
| Node-failure recovery | BLOCKED | No second node or induced multi-node failure surface exists. |

## 6. What is UNMEASURED and why

### 6.1 Phase 3

The genuine Phase 3 script has executed **zero of eight legs: 0/8**. Manual lower-level probes are not counted as framework legs.

| Phase 3 leg | Status | Missing measurement |
|---|---:|---|
| L1 — environment | BLOCKED | In-run torch version/path, process counts, device inventory, and user-site state under the launched training process. |
| L2 — model load | BLOCKED | Full BF16 load, loader class, parameter count, dtype histogram, and nonzero denominator. |
| L3 — distributed collective | BLOCKED | Eight-rank collective launched by FoundationScale and proven to include all eight ranks. |
| L4 — FSDP sharding | BLOCKED | FSDP1/FSDP2 selection, per-rank shard counts, and sum-over-ranks equality with the L2 parameter denominator. |
| L5 — optimizer steps | BLOCKED | Twenty bounded optimizer steps, per-step loss, finite-loss checks, strictly decreasing first/last synthetic loss, tokens/s, step time, and peak GPU memory. |
| L6 — checkpoint save | BLOCKED | Model-plus-optimizer state, file count, byte count, shard inventory, and residual-buffer behavior. |
| L7 — restore | BLOCKED | MUST_FIRE on a perturbed tensor followed by bitwise equality as `k of N`, where `N > 0`, in a fresh model and optimizer. |
| L8 — evaluation equivalence | BLOCKED | Fixed-seed, fixed-input, eval-mode output comparison with element count and maximum absolute delta. |

### 6.2 Phase 4 and broader generalization

| Claim | Status | Evidence boundary |
|---|---:|---|
| Full Phase 4 execution | UNMEASURED | Zero full model-load, distributed, checkpoint, resume, and evaluation runs have occurred for a second architecture. |
| Second-architecture full loading | UNMEASURED | Configuration resolution and meta-device construction do not materialize weights. |
| Original Gemma/Qwen pair | UNMEASURED | The original two candidates cover zero of two MoE cases and zero of two text-only cases [M9]. |
| Out-of-tree Gemma-4 architecture | BLOCKED | Both candidates require an architecture-registration seam; the current matrix records both as unable to build under its measured toolchain: 2/2 [E.1; #119]. |
| MoE checkpoint machinery | UNMEASURED | No MoE checkpoint exists, and the expert-FQN logic has not been exercised by a real model. |
| Multi-node distributed training | BLOCKED | Exactly one node is available. No inter-node launch is possible on the measured estate [M11]. |
| Inter-node NCCL | BLOCKED | The measured NVLink result is intra-node only. No inter-node transport was tested. |
| Node-failure recovery | BLOCKED | There is no second node on which to induce or survive a node failure. |
| Throughput and convergence claims | UNMEASURED | No model-training step has run. The 420.8 GB/s communication number applies only to a 256 MiB standalone collective. |
| Checkpoint correctness under training | UNMEASURED | Synthetic metadata and tensor files establish adjudicator discrimination, not writer correctness. |
| Dataset correctness | UNMEASURED | No dataset batch or token denominator exists. |
| Framework evaluation accuracy | UNMEASURED | No labeled corpus, reference answer set, or score denominator exists. |
| Final-credential equivalence | BLOCKED | Permissions, quotas, host user-site state, and module availability are credential-scoped and must be rerun. |
| Current 20-stage build determinism | UNMEASURED | The recorded byte-identical result belongs to the earlier 17-stage/five-artifact build. No current seven-artifact double-build comparison exists [E.5; CUR-BUILD]. |
| Framework-level image reproducibility | UNMEASURED | The raw communication run entered a container directly. The framework launch path has never entered it. |

## 7. What must happen next, in order

| Order | Required action | Current status | Named blocker | Acceptance evidence |
|---:|---|---:|---|---|
| 1 | Land the generated #142 correction. | BLOCKED | Fix is generated but not landed. | The launcher-consumed filename is a declared build output. A missing-file MUST_FIRE and present-file MUST_PASS both execute. |
| 2 | Land the generated #146 correction. | BLOCKED | Fix is generated but not landed. | The adjudicator knob is containment-checked, bound through both runtime adapters, and visible in-container. Absent and overwritten-value MUST_FIRE rows execute. |
| 3 | Repair M3’s launch measurement seam. | BLOCKED | The probe measures host Python before entering the runtime. | The measured GPU count is produced by the declared runtime or rank authority. Zero, nonnumeric, and mismatched-count negative controls all refuse. |
| 4 | Repair M4 and M5 before relying on bounded probes. | BLOCKED | M4 cannot reject invalid limits; M5’s two budget names have no consumers. | An over-limit value is refused and an in-limit value admitted. Both budget names are consumed by the probe and control its real step/save counts. |
| 5 | Reconcile #150 and tighten #132. | BLOCKED | Writer and adjudicator disagree over both generated formats; mount checking covers declared roots only where declared. | Both governed checkpoint formats are accepted or explicitly refused with a declared reason. Recursive or contract-scoped mount claims state exactly how many entries were checked. |
| 6 | Rebuild and publish the artifact manifest. | BLOCKED | Items 1–5 are not landed. Must list every artifact hash, its producing stage, its parser, its blocklist denominator, and all executed suite totals. |
| 7 | Receive the authorized credential. | BLOCKED | Credential must come from the requester’s secure channel. | An authorized non-interactive push succeeds and the receipt records the exact pushed hash set. |
| 8 | Push immutable artifacts and repeat credential-scoped measurements. | BLOCKED | Items 6–7. | Runtime inventory, host user-site, image import, mount, and torch-provenance controls rerun under the authorized credential with unchanged artifact hashes. |
| 9 | Run genuine Phase 3 on eight H100s. | BLOCKED | Items 1–8. | All eight FSLEG lines execute through the framework. Exit is 0 only when all eight are measured. The summary carries every `k of N` and parameter count is greater than zero. |
| 10 | Run the real-dataset gate. | UNMEASURED | Phase 3 is synthetic-only and cluster execution remains blocked. | The gate resolves and binds the dataset, reads more than zero examples, tokenizes a nonzero batch count, reports token denominators, and fires on a planted unreadable dataset. |
| 11 | Execute Phase 4 across genuinely distinct architectures. | BLOCKED | Phase 3 is incomplete; the architecture-registration seam required by the two out-of-tree candidates is absent. | At least two architecture families load full weights and pass loading, distributed, checkpoint, resume, and evaluation legs. Dense, multimodal, text-only, and MoE scope are reported separately. |
| 12 | Extend distributed claims only when the hardware denominator changes. | BLOCKED | Only one node exists on the measured estate. | Inter-node NCCL and node-failure claims require at least two nodes under the final credential and MUST_FIRE for a lost or wrong-rank contribution. |

**Bottom line:** the measured evidence establishes the estate boundary, the only usable runtime, the available HF engine, numerous controlled source-level remediators, one manual NCCL transport baseline, and synthetic checkpoint-adjudicator discrimination. It does not establish FoundationScale training on H100s. Phase 3 remains **0/8 legs executed**, and certification of FoundationScale as a generalizable foundation-model training framework remains **unachieved**.