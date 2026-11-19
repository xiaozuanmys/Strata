# gfx90a (Radeon Instinct MI210): pre-built engine + HIP compat fix

**English** · [中文](#中文说明)

This branch carries three things:

1. a **two-line fix** to the HIP compatibility header — without it the engine does not compile on ROCm 7.x;
2. a **pre-built `strata` engine binary for gfx90a**, so you do not have to build it yourself;
3. a worked **IQ3\_XXS example** with measured numbers for a single 64 GB MI210 at 256K context.

Verified against: Strata engine **0.1.48.1** / Terminal **0.2.3.1**, **ROCm 7.15**, hipBLAS 3.6,
Linux x86-64, one **Radeon Instinct MI210 64 GB** (gfx90a), passed through the
`gfx906 wave64-compat` build path because gfx90a has no official pre-built engine package.

---

## 1. The patch

`engine/include/strata/platform/hip_compat/strata_hip.h` aliases the `cuda*` entry points the code base
uses onto their `hip*` equivalents. Two aliases are missing, and the HIP build stops at
`hipcc` on ROCm 7.15 with:

```
error: use of undeclared identifier 'cudaDeviceGetStreamPriorityRange'
error: use of undeclared identifier 'cudaStreamCreateWithPriority'
```

Both exist in HIP with the same signature, so the fix is a straight alias — two lines, no
behaviour change for any other backend:

```diff
 #define cudaGetDeviceProperties hipGetDeviceProperties
+#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
@@
 #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
+#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
```

## 2. Pre-built binary — no rebuild needed

Grab `strata-0.1.48.1-gfx90a.tar.gz` from the Release attached to this branch.

| | |
|---|---|
| Engine version | 0.1.48.1 |
| Asset | `strata-0.1.48.1-gfx90a.tar.gz`, 8,841,667 bytes (8.4 MiB) |
| Asset sha256 | `f9ed0449149ad34b66e24da577746309733e531f8a2c08415528f6392c7c4f14` |
| Unpacked | `strata` (32,800,136 bytes, sha256 `b74571046cb2ef23c0e4afb09ef99ee05c877c3ba64c71da3dff85102119ca13`) and `strata-device` (54,192 bytes, sha256 `6b3971c78b3270e22c9140b59a6817e1153846dac6ac7a1ad61ee50985587c9f`) |
| File modes | both `0755`, archive stores uid/gid 0 — extract and run, no `chmod` needed |
| Requires | Linux x86-64, ROCm 7.x installed in `/opt/rocm` |
| Bundles | nothing — `ldd` resolves only ROCm libraries (`libhipblas`, `libhipblaslt`, `libamdhip64`, `librocblas`, `librocsolver`); no private `.so` next to the binary |

```bash
tar xzf strata-0.1.48.1-gfx90a.tar.gz
sha256sum strata strata-device    # must match the values above
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
./strata-device --list-devices    # → MI210, CC 9.0, 64 GiB
```

### 2.1 The binary alone is not enough — what else you need

Running a model needs three inputs besides the engine executable:

| input | where it comes from |
|---|---|
| the GGUF model shards | yours, not distributed here |
| `--pack` | rebuild locally in seconds: `.venv/bin/python terminal/tools/iq_pack.py --gguf <shard-1> --out <pack-dir>` (`--out` is required; the script needs numpy, so use the venv's interpreter). Without `--experts-bin` the engine reads the experts straight from the GGUF — that is what these numbers were produced with, and the pack then holds `dense.bin`, `index.txt`, `native_experts.txt` and `tokenizer/`. Add `--experts-bin` only if your GGUF carries no experts of its own. |
| `--expert-profile` | `terminal/data/expert-profile.bin`, ships with the repo |
| `--mtp <dir>/rt` | rebuilt locally from the BF16 checkpoint — see §2.2. The GGUF ships no MTP head, and nothing has to be downloaded from us. |

### 2.2 The MTP runtime (`rt/`): rebuild it, nobody has to ship it

`--mtp` wants a directory holding `experts.bin`, `dense.bin`, `dense.txt` and `draft_vocab.bin`. The GSQ-RCO GGUF
ships no MTP head, so these are built from the BF16 checkpoint (`Qwen/Qwen3.8-Flash-Next`, 360 GB in 131 shards)
— but **only the 31 `mtp.*` tensors are fetched**, via safetensors header + HTTP range requests, never the whole
checkpoint:

```bash
python3 terminal/tools/mtp_fetch.py inventory --out mtp-bf16   # shard headers only, a few KB each
python3 terminal/tools/mtp_fetch.py fetch     --out mtp-bf16   # the 31 mtp.* tensors, resumable
python3 terminal/tools/mtp_fetch.py verify    --out mtp-bf16   # against SHA256 (exit 3 = some are bad)

python3 terminal/tools/mtp_pack.py --src mtp-bf16 --experts q2_0 --out mtp-q2_0.gguf
python3 terminal/tools/mtp_rt.py   --gguf mtp-q2_0.gguf --out rt
```

`mtp_rt.py` writes `experts.bin` (the 512 routed experts in the engine's blob layout, a lossless relayout of the
GGUF's Q2_0 blocks), `dense.bin` (everything else: projections as Q8_0, hyper-connections and router as BF16,
RMSNorm as F32 with the Gemma "+1" applied) and `dense.txt` (one line per tensor).

Both `mtp_pack.py` and `mtp_rt.py` import llama.cpp's `gguf-py`. `./setup.sh` / `setup.py` downloads llama.cpp into
`third_party/`, which is where they look; otherwise point `STRATA_GGUF_PY` at a `gguf-py` folder.

**Chinese and Japanese users: rebuild the draft vocabulary.** `rt/draft_vocab.bin` as shipped was built from English
and code and covers 27 of the vocabulary's 55,328 Han tokens, so an answer written in Chinese drafts almost nothing
(upstream #137). That silently costs you the speculative-decoding speed-up:

```bash
python3 terminal/tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf \
        --base terminal/data/draft_vocab.bin --add cjk
```

## 3. Building it yourself instead

```bash
git clone https://github.com/Niko1221/Strata.git && cd Strata
# apply the two #define lines from §1 (or check out this branch)

cmake -B build-hip -S engine \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

ninja -C build-hip strata
```

Two points that are easy to miss:

* `-DSTRATA_HIP_GFX906=ON` selects the **wave64 compatibility path**. That is the switch that
  makes gfx90a compile at all — there is no gfx90a preset.
* Build the `strata` target only. The default target also produces ~40 `*_parity` test
  binaries and pulls a full `llama.cpp` checkout into `_deps/` (the build tree ends up
  close to 1 GB, and more than 400 MB of that is that nested `.git`).

## 4. Measured: IQ3\_XXS at 256K context on a single MI210 64 GB

`--max-context 262144` (256K) runs correctly on **one** MI210. All five prompt lengths passed
with zero errors.

Short answer, second call after warm-up: **85.0 tok/s** (124 tokens / 1458 ms; 71/106 drafts
accepted; 99.23 % of the KV working set resident in VRAM).

| prompt | tokens | prefill tok/s | TTFT | decode tok/s | total |
|---:|---:|---:|---:|---:|---:|
| 18k  | 18,290  | 1,093 | 16.7 s  | 81.5 | 36.4 s  |
| 50k  | 50,749  | 1,102 | 46.0 s  | 76.5 | 66.9 s  |
| 100k | 101,375 | 1,063 | 95.4 s  | 76.2 | 116.4 s |
| 150k | 152,138 | 1,014 | 150.1 s | 74.0 | 171.7 s |
| 200k | 202,905 | 975   | 208.2 s | 65.8 | 232.5 s |

Honest read of these numbers:

* **decode is the strong side** — 65.8 tok/s even at a 200k prompt.
* **prefill is the weak side**, ~1,000 tok/s, so time-to-first-token reaches 208 s at 200k.
  This configuration suits generation-heavy work, not long-document re-questioning.
* 0 swap throughout, host RAM peak ~51 GB, GPU junction peaked at **74 °C** (edge 64 °C),
  37 W idle.

## 5. Starting the service

The engine is driven through the Terminal's `serve.server` module. Install Terminal first:

```bash
git clone https://github.com/Niko1221/Strata.git && cd Strata
./setup.sh          # Linux; creates .venv and installs the pinned deps
```

`./setup.sh` is interactive and also wants to pick and download a model. If you only want the Python environment,
make it by hand — that is all `serve.server` needs (it imports `regex`, `numpy`, `jinja2`, `pyyaml`, `psutil`,
all pinned in `terminal/requirements.txt`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r terminal/requirements.txt
```

Drop the two binaries from the Release into the repo, then start the service:

```bash
ulimit -l unlimited    # required: pins the ~40 GB resident expert cache
rm -f <ready-file>     # a stale one makes serve.server exit immediately
cd terminal                 # serve.server lives under terminal/
python -m serve.server --engine strata \
    --config <your-iq3-config.json> \
    --host 127.0.0.1 --port 8092 \
    --ready-file <ready-file>
```

Run it with the interpreter from that `.venv`. Bind to `127.0.0.1`
unless you really need the LAN to reach it, and set a strong `api_key` before you widen it.

The engine flags that produced the numbers above (they go in the config's `args`):

| flag | value | note |
|---|---|---|
| `--max-context` | `262144` | 256K — single GPU only; a 2-GPU layer-split layout collapses to ~0.4 tok/s |
| `--kv` | `int8` | |
| `--kv-resident` | `20480` | |
| `--ple-io` | `mmap` | |
| `--vram-reserve-mib` | `1024` | headroom left on the card (default 700; accepted by the engine but not listed in `--help`) |
| `--spec` / `--spec-min-p` | `4` / `0.5` | speculative decoding — the 71/106 draft acceptance above |
| `--expert-cache` / `--prefill` | `auto` | |
| `--native` / `--ple-gguf` | shard 1 / shard 2 | the two IQ3\_XXS GGUF shards |
| `--backend` | `hip` | |
| `--gpu` | `[0]` | |

## 6. Upstream

The §1 patch is intended for upstream (`Niko1221/Strata`) — it is backend-neutral and unblocks
every ROCm build. Track it in the pull request opened from this branch.

---

# 中文说明

这个分支里有三样东西：

1. HIP 兼容头文件的**两行修复** —— 缺了它，引擎在 ROCm 7.x 上根本编译不过；
2. **gfx90a 的预编译 `strata` 引擎**，不用自己从头编译；
3. 一份完整跑通的 **IQ3\_XXS 实例**，附单张 64 GB MI210 在 256K 上下文下的实测数据。

实测环境：Strata 引擎 **0.1.48.1** / Terminal **0.2.3.1**、**ROCm 7.15**、hipBLAS 3.6、
Linux x86-64、单张 **Radeon Instinct MI210 64 GB**（gfx90a）。gfx90a 没有官方预编译引擎包，
所以走 `gfx906 wave64 兼容`路线编译。

## 1. 补丁

`engine/include/strata/platform/hip_compat/strata_hip.h` 负责把代码里用到的 `cuda*` 接口名转接到
`hip*`。这里少转接了两个，于是 ROCm 7.15 下 `hipcc` 编译直接报：

```
error: use of undeclared identifier 'cudaDeviceGetStreamPriorityRange'
error: use of undeclared identifier 'cudaStreamCreateWithPriority'
```

这两个函数 HIP 里都有、签名完全一致，所以修复就是补两行转接，对其它后端零影响：

```diff
 #define cudaGetDeviceProperties hipGetDeviceProperties
+#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
@@
 #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
+#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
```

## 2. 预编译二进制 —— 不用自己编

在本分支的 Release 里下载 `strata-0.1.48.1-gfx90a.tar.gz`。

| | |
|---|---|
| 引擎版本 | 0.1.48.1 |
| 附件 | `strata-0.1.48.1-gfx90a.tar.gz`，8,841,667 字节（8.4 MiB） |
| 附件 sha256 | `f9ed0449149ad34b66e24da577746309733e531f8a2c08415528f6392c7c4f14` |
| 解压后 | `strata`（32,800,136 字节，sha256 `b74571046cb2ef23c0e4afb09ef99ee05c877c3ba64c71da3dff85102119ca13`）与 `strata-device`（54,192 字节，sha256 `6b3971c78b3270e22c9140b59a6817e1153846dac6ac7a1ad61ee50985587c9f`） |
| 文件权限 | 两个都是 `0755`，归档内 uid/gid 全为 0 —— 解开就能跑，不用再 chmod |
| 依赖 | Linux x86-64，`/opt/rocm` 下装好 ROCm 7.x |
| 附带内容 | 无 —— `ldd` 只解析到 ROCm 自带库（`libhipblas`、`libhipblaslt`、`libamdhip64`、`librocblas`、`librocsolver`），二进制旁边没有任何私有 `.so` |

```bash
tar xzf strata-0.1.48.1-gfx90a.tar.gz
sha256sum strata strata-device    # 必须和上面的值一致
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
./strata-device --list-devices    # → MI210, CC 9.0, 64 GiB
```

### 2.1 光有二进制还不够 —— 其余几样输入

跑起一个模型，除引擎本体还需要以下输入：

| 输入 | 从哪来 |
|---|---|
| GGUF 模型分片 | 你自己的，本仓库不分发 |
| `--pack` | 本地几秒重建：`.venv/bin/python terminal/tools/iq_pack.py --gguf <分片1> --out <pack目录>`（`--out` 必填；脚本要用 numpy，所以用 venv 里的 python）。不加 `--experts-bin` 时引擎直接从 GGUF 读专家 —— 上面那组数据就是这么跑出来的，产出的 pack 里是 `dense.bin`、`index.txt`、`native_experts.txt` 和 `tokenizer/`。只有你的 GGUF 本身不带专家时才加 `--experts-bin`。 |
| `--expert-profile` | `terminal/data/expert-profile.bin`，仓库自带 |
| `--mtp <目录>/rt` | 从 BF16 权重本地重建 —— 见 §2.2。GGUF 本身不带 MTP 头，也不需要从我们这里下载任何东西。 |

### 2.2 MTP 运行时（`rt/`）：自己重建，不需要谁分发

`--mtp` 要的是一个装着 `experts.bin`、`dense.bin`、`dense.txt`、`draft_vocab.bin` 的目录。GSQ-RCO 的 GGUF 不带 MTP 头，
所以这些要从 BF16 原始权重（`Qwen/Qwen3.8-Flash-Next`，360 GB / 131 shard）重建 —— 但**只抓 31 个 `mtp.*` 张量**，
走 safetensors 头部 + HTTP range 请求，绝不下载整个 360 GB：

```bash
python3 terminal/tools/mtp_fetch.py inventory --out mtp-bf16   # 只读各 shard 的头部，每个几 KB
python3 terminal/tools/mtp_fetch.py fetch     --out mtp-bf16   # 取那 31 个 mtp.* 张量，可断点续传
python3 terminal/tools/mtp_fetch.py verify    --out mtp-bf16   # 按 SHA256 校验（退出码 3 = 有坏块）

python3 terminal/tools/mtp_pack.py --src mtp-bf16 --experts q2_0 --out mtp-q2_0.gguf
python3 terminal/tools/mtp_rt.py   --gguf mtp-q2_0.gguf --out rt
```

`mtp_rt.py` 会写出 `experts.bin`（512 个路由专家按引擎的 blob 布局重排，是 GGUF 里 Q2_0 block 的无损重排）、
`dense.bin`（其余张量：投影层 Q8_0、hyper-connection 与 router 保持 BF16、RMSNorm 用 F32 并套上 Gemma 的 "+1"）
和 `dense.txt`（每个张量一行）。

`mtp_pack.py` 和 `mtp_rt.py` 都要 import llama.cpp 的 `gguf-py`。`./setup.sh` / `setup.py` 会把 llama.cpp 下到
`third_party/`，正是它们找的位置；不然自己把 `STRATA_GGUF_PY` 指过去。

**中文 / 日文用户请重建 draft vocabulary。** 出厂的 `rt/draft_vocab.bin` 是按英文和代码建的子集，55,328 个汉字里
只覆盖 27 个，所以中文回答几乎 draft 不中（上游 #137）—— 这会在你察觉不到的情况下吃掉投机解码的加速：

```bash
python3 terminal/tools/draft_vocab.py --gguf <模型>-00001-of-0000N.gguf \
        --base terminal/data/draft_vocab.bin --add cjk
```

## 3. 想自己编译

```bash
git clone https://github.com/Niko1221/Strata.git && cd Strata
# 补上 §1 那两行 #define（或直接签出本分支）

cmake -B build-hip -S engine \
  -DSTRATA_HIP_GFX906=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a

ninja -C build-hip strata
```

两个最容易踩的点：

* `-DSTRATA_HIP_GFX906=ON` 打开的是 **wave64 兼容路线**。gfx90a 没有预设，就是这个开关让它能编过。
* 只要引擎就**只编 `strata` 这一个 target**。默认 target 会连带产出约 40 个 `*_parity` 测试程序，
  还会把整个 `llama.cpp` 拉进 `_deps/`（构建树最后接近 1 GB，其中 400 MB 以上是那个嵌套的 `.git`）。

## 4. 实测：单张 MI210 64 GB 跑 IQ3\_XXS，256K 上下文

`--max-context 262144`（256K）在**单张** MI210 上正常工作，五档长度全部通过、零报错。

短回答（预热后第二次调用）：**85.0 tok/s**（124 token / 1458 ms；草稿接受 71/106；
99.23 % 的 KV 工作集常驻显存）。

| 档位 | token 数 | prefill tok/s | 首字 TTFT | decode tok/s | 总耗时 |
|---:|---:|---:|---:|---:|---:|
| 18k  | 18,290  | 1,093 | 16.7 s  | 81.5 | 36.4 s  |
| 50k  | 50,749  | 1,102 | 46.0 s  | 76.5 | 66.9 s  |
| 100k | 101,375 | 1,063 | 95.4 s  | 76.2 | 116.4 s |
| 150k | 152,138 | 1,014 | 150.1 s | 74.0 | 171.7 s |
| 200k | 202,905 | 975   | 208.2 s | 65.8 | 232.5 s |

对这组数据的诚实解读：

* **强项是 decode** —— 200k 长上下文下仍有 65.8 tok/s。
* **弱项是 prefill**，约 1,000 tok/s，200k 时首字要等 208 秒。这套配置适合以生成为主的活，
  不适合反复追问长文档。
* 全程 0 swap，内存峰值约 51 GB，GPU 热点峰值 **74 °C**（边缘 64 °C），空闲 37 W。

## 5. 起服务

引擎由 Terminal 的 `serve.server` 模块带起来。先装 Terminal：

```bash
git clone https://github.com/Niko1221/Strata.git && cd Strata
./setup.sh          # Linux：建 .venv 并装好锁定版本的依赖
```

`./setup.sh` 是交互式的，还会让你选模型并下载。如果你只要那个 Python 环境，手动建一个就行 ——
`serve.server` 需要的也只是它（它要 `regex`、`numpy`、`jinja2`、`pyyaml`、`psutil`，版本都在 `terminal/requirements.txt` 里钉好）：

```bash
python3 -m venv .venv
.venv/bin/pip install -r terminal/requirements.txt
```

把 Release 里的两个二进制放进仓库目录，然后起服务：

```bash
ulimit -l unlimited    # 必须：锁住约 40 GB 常驻专家缓存
rm -f <ready-file>     # 残留的 ready-file 会让 serve.server 立刻退出
cd terminal                 # serve.server lives under terminal/
python -m serve.server --engine strata \
    --config <你的 iq3 配置.json> \
    --host 127.0.0.1 --port 8092 \
    --ready-file <ready-file>
```

用上面 `.venv` 里的解释器跑。除非确实要让局域网访问，否则只绑
`127.0.0.1`；放开之前先把 `api_key` 换成强口令。

产出上面那组数据的引擎参数（写在配置的 `args` 里）：

| 参数 | 值 | 说明 |
|---|---|---|
| `--max-context` | `262144` | 256K，仅限单卡；双卡 layer split 会因分层不均崩到约 0.4 tok/s |
| `--kv` | `int8` | |
| `--kv-resident` | `20480` | |
| `--ple-io` | `mmap` | |
| `--vram-reserve-mib` | `1024` | 给卡留的余量（默认 700；引擎认识它，但 `--help` 里不列） |
| `--spec` / `--spec-min-p` | `4` / `0.5` | 投机解码 —— 即上面 71/106 的草稿接受率 |
| `--expert-cache` / `--prefill` | `auto` | |
| `--native` / `--ple-gguf` | 分片 1 / 分片 2 | IQ3\_XXS 的两个 GGUF 分片 |
| `--backend` | `hip` | |
| `--gpu` | `[0]` | |

## 6. 回馈上游

§1 那个补丁是准备提给上游（`Niko1221/Strata`）的 —— 它与后端无关，能让所有 ROCm 构建解开。
进度见从本分支发起的 pull request。
