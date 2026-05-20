# Learning to Intervene on Concept Bottlenecks (CB2M)

**著者**: David Steinmann, Wolfgang Stammer, Felix Friedrich, Kristian Kersting  
**所属**: Artificial Intelligence and Machine Learning Group, TU Darmstadt / Hessian Center for Artificial Intelligence (hessian.AI) / Centre for Cognitive Science, TU Darmstadt / German Center for Artificial Intelligence (DFKI)  
**出版**: Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235, 2024 (arXiv:2308.13453)

---

## 1. 論文の目的または目標は何ですか？

Concept Bottleneck Models (CBM, Koh+ 2020) は人間が理解可能な中間 concept を介して推論することで解釈性を提供し，テスト時に人が concept へ\*\*介入 (intervention)\*\*して最終予測を修正できる．しかし，この介入は\*\*1 サンプル限りの使い捨て\*\*であり，同じミスが再発しても都度修正が必要で，運用コストが高い．

本論文の目的は，CBM の bottleneck・predictor 本体には手を入れず，\*\*外付けの 2-fold memory\*\*によって介入を「覚えて」似たサンプルへ自動で再利用させること．具体的には次の 2 つを実現する：

1. **ミス検出 (mistake detection)**: 事前知識なしに，過去のミスに似た入力を検出し，ユーザに的を絞った介入要求を出す．
2. **介入の汎化 (intervention generalization)**: 一度行った介入を，類似サンプルへ自動で再適用し，**追加の再学習なし**に CBM を修正する．

これにより，「少数の人手介入から汎化でき，柔軟に拡張可能な CBM」を作ることを目指す．

---

## 2. トピックを探求するために使用された方法またはアプローチは何ですか？

### 2-fold Memory Module（外付け記憶）

CB2M は CBM の bottleneck $g$・predictor $f$ をそのまま保ったうえで，**2 種類のメモリ**を外付けする．類似判定は最終的な concept ベクトルではなく，**bottleneck 最終層への入力 $x_e$**（concept に変換される直前の内部 encoding）の上で行う．距離はユークリッド距離 $d(x_e, x'_e) = \lVert x_e - x'_e \rVert$．

- **Mistake memory $M_M$**: 最終予測を誤り，かつ concept 予測精度 $\mathrm{Acc}_g(x)$ が閾値 $t_a \in [0,1]$ 未満のサンプルの encoding を貯める：
  $$M_M = \{\,x_e : f(g(x)) \neq y^{*} \,\wedge\, \mathrm{Acc}_g(x) < t_a\,\}$$
  これは bottleneck 由来の誤りを集めたメモリ．
- **Intervention memory $M_I$**: 実際に行われた介入 $i$ を記録．どの encoding にどの介入が対応するかを $\alpha(x_e, i)$ で紐付ける．

### ミス検出（Eq.1）と介入の汎化

- **ミス検出**: 新規入力 $\hat{x}$ について，$M_M$ 内に距離 $t_d$ 以内の既知ミスが $k$ 個見つかれば「既知のミスを犯す」と判定する：
  $$\forall j \in \{1,\dots,k\}:\ \exists\, x_{e,j} \in M_M:\ d(\hat{x}_e, x_{e,j}) \leq t_d$$
- **介入の汎化**: 介入のたびに $x_e \to M_M$，$i \to M_I$ を保存．新規入力は $k=1$（最類似ミス 1 つ）で照合し，$t_d$ 以内なら対応する $i$ をそのまま再適用 → 人手なしで自動修正．$M_M$ は運用中もユーザ feedback で継続更新される．
- 設計上，メモリは bottleneck/predictor から独立．従来の最近傍法でも，微分可能な neural nearest neighbor でも構成可能．

### 評価設定

- **データセット**: CUB (Aug., 鳥分類), Parity MNIST (unbalanced, 数字の偶奇), Parity C-MNIST（背景色との交絡を含む数字＝分布シフト・交絡条件）．
- **指標**: 「一度誤判定された画像」での正解率，ミス検出は AUROC，未知データへの介入汎化の波及効果は NRI で評価．concept 選択手法 ECTP との併用も可能．

---

## 3. 主要な結果または発見は何でしたか？

- **「一度間違えた画像」の正解率**（原典 Table 1, スライド本文に忠実）：

  | データ | 素の CBM | CB2M |
  |---|---|---|
  | CUB (鳥) | 5.0% | **88.7%** |
  | Parity MNIST (数字) | 22.5% | **93.7%** |
  | Parity C-MNIST (交絡数字) | 20.1% | **85.9%** |

  いずれも **モデルを再学習せず**，メモリの追加だけで達成．

- ごく少数の人手介入から大幅な精度改善：人が直すのは少数の画像で十分で，残りは記憶された介入が自動で再利用される．
- **未知データへの汎化** と **分布シフト・交絡データを含む challenging 条件** での頑健な動作を確認．
- ミス検出は事前知識なしで動作し，交絡データを含むベースラインを AUROC で超える．
- 介入の影響範囲は NRI で評価され，類似サンプルへ波及することを定量的に確認．

---

## 4. 結論は何であり，なぜそれが重要なのですか？

CB2M は，「**一度した修正をメモリに覚え，似た入力へ自動再利用する**」というシンプルなアイデアにより，CBM の最大の運用課題＝介入の使い捨て問題を解消した．bottleneck・predictor 本体には手を入れないため，既存の CBM にそのまま外付けでき，**追加の再学習も不要**で，少数の介入から大きな改善が得られる．

論文の 4 つの貢献は以下に整理される：

1. **介入の汎化** による CBM の自動修正．
2. **対話的介入**の柔軟な拡張（モデルを再学習せず継続的に改善）．
3. **未知データへの汎化** と「介入からの学習」．
4. **事前知識なしのミス検出**と的を絞った介入要求．

**重要性**：解釈可能 AI（CBM）を「使えるもの」にするための実用的な一歩である．特に医療・専門領域のように介入の質と人的コストが釣り合わない応用において，少数の人手介入で持続的に改善する CBM は，インタラクティブなフィードバックループを現実的にする．CBM 系の後続研究の流れの中では，「介入の単発性」という運用課題に正面から取り組み，bottleneck の不変性を保ったまま改善する数少ない手法として位置付けられる．
