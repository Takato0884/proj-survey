# 論文サーベイ

機械学習・コンピュータビジョン・HCI/HRI などを横断する論文サマリー集です。**手法（方法論）** の観点でクラスタリングし、各クラスタ内は **年が新しい順** に並べています。

論文サマリー本体は [summary/](summary/) 配下にあります。テンプレートは [summary/template.md](summary/template.md) を参照してください。

---

## 目次

1. [教師なしドメイン適応：分類タスク（敵対的・差異最小化）](#1-教師なしドメイン適応分類タスク)
2. [教師なしドメイン適応：回帰タスク](#2-教師なしドメイン適応回帰タスク)
3. [Negative Transfer と転移学習の理論](#3-negative-transfer-と転移学習の理論)
4. [概念ベース解釈可能AI（Concept Bottleneck 系）](#4-概念ベース解釈可能aiconcept-bottleneck-系)
5. [LLM 内部表現の幾何学とモデル編集](#5-llm-内部表現の幾何学とモデル編集)
6. [AI 評価ベンチマーク・人間嗜好学習（ペアワイズ/ランキング）](#6-ai-評価ベンチマーク人間嗜好学習)
7. [Vision 基盤モデル・セグメンテーション・3D 再構成](#7-vision-基盤モデルセグメンテーション3d-再構成)
8. [HRI/HCI ユーザースタディ](#8-hrihci-ユーザースタディ)
9. [行動・神経科学に基づく美的選好研究](#9-行動神経科学に基づく美的選好研究)

---

## 1. 教師なしドメイン適応：分類タスク

ラベル分布を持つ source → target で、敵対的整合・分類器不一致最大化・特徴統計整合・最適輸送など、**特徴/分布レベルでドメインギャップを縮める**手法群。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [Learning Invariant Representations and Risks for Semi-supervised Domain Adaptation (LIRR)](summary/Li_2021_LIRR.md) | UC Berkeley (BAIR) / UCSD / MSRA / UIUC | arXiv preprint | 2021 |
| [Adversarial-Learned Loss for Domain Adaptation (ALDA)](summary/Chen_2020_ALDA.md) | Zhejiang University / Fabu Inc. | AAAI | 2020 |
| [Learning to Transfer Examples for Partial Domain Adaptation (ETN)](summary/Cao_2019_ETN.md) | Tsinghua University / HKUST | CVPR | 2019 |
| [Partial Adversarial Domain Adaptation (PADA)](summary/Cao_2018_PDAN.md) | Tsinghua University | ECCV | 2018 |
| [DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation](summary/Damodaran_2018_DeepJDOT.md) | Univ. Bretagne Sud / Wageningen Univ. / Univ. Côte d'Azur | ECCV | 2018 |
| [Conditional Adversarial Domain Adaptation (CDAN)](summary/Long_2018_CDAN.md) | Tsinghua University / UC Berkeley | NeurIPS | 2018 |
| [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (MCD)](summary/Saito_2018_MCD.md) | University of Tokyo / RIKEN | CVPR | 2018 |
| [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](summary/Sun_2016_DCORAL.md) | UMass Lowell / Boston University | ECCV Workshop | 2016 |

---

## 2. 教師なしドメイン適応：回帰タスク

連続値を出力する回帰モデルへのドメイン適応。Gram 行列整合・不確実性駆動の整合・順序回帰・敵対的重み付けなど **回帰特有の整合手法**。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [Uncertainty-Guided Alignment for Unsupervised Domain Adaptation in Regression](summary/Nejjar_2024_UncertainUDAR.md) | EPFL | Reliability Engineering and System Safety | 2024 |
| [DARE-GRAM: Unsupervised Domain Adaptation Regression by Aligning Inverse Gram Matrices](summary/Nejjar_2023_DAREGRAM.md) | EPFL / ETH Zurich | CVPR | 2023 |
| [Representation Subspace Distance for Domain Adaptation Regression (RSD)](summary/Chen_2021_RSD.md) | Tsinghua University (BNRist) | ICML | 2021 |
| [Adversarial Weighting for Domain Adaptation in Regression (WANN)](summary/Mathelin_2021_WANN.md) | Michelin / EDF R&D / Univ. Paris-Saclay | arXiv preprint | 2021 |
| [Deep Domain Adaptation with Ordinal Regression for Pain Assessment Using Weakly-Labeled Videos](summary/Rajasekhar_2021_WDA-pain.md) | École de technologie supérieure, Univ. du Québec | Image and Vision Computing | 2021 |

---

## 3. Negative Transfer と転移学習の理論

転移学習がいつ・なぜ悪化するか（Negative Transfer）の特徴づけ・回避策・体系的整理。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [A Survey on Negative Transfer](summary/Zhang_2021_NegativeTrans.md) | 華中科技大学 / 重慶大学 | IEEE TNNLS | 2021 |
| [Characterizing and Avoiding Negative Transfer](summary/Zhang_2019_NT.md) | Carnegie Mellon University | arXiv preprint | 2019 |

---

## 4. 概念ベース解釈可能AI（Concept Bottleneck 系）

人間が理解可能な **中間概念** を介して予測を行う/概念に介入する解釈可能AIの系譜。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [From Concepts to Judgments: Interpretable Image Aesthetic Assessment](summary/Liu_2026_XAI_aest.md) | KU Leuven | arXiv preprint | 2026 |
| [WP-CLIP: Leveraging CLIP to Predict Wölfflin's Principles in Visual Art](summary/Abhijay_2025_WP-CLIP.md) | Portland State University | ICCV Workshop | 2025 |
| [Learning to Intervene on Concept Bottlenecks (CB2M)](summary/Steinmann_2024_CB2M.md) | TU Darmstadt / hessian.AI / DFKI | ICML | 2024 |
| [Concept Bottleneck Models (CBM)](summary/Koh_2020_CBM.md) | Stanford University / Google Research | ICML | 2020 |

---

## 5. LLM 内部表現の幾何学とモデル編集

LLM の表現空間の幾何構造の理論的解析、タスクベクトルによるモデル編集、語彙化（トークナイザ）が表現に与える影響。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [The Linear Representation Hypothesis and the Geometry of Large Language Models (LRH)](summary/Park_2024_LRH.md) | University of Chicago | ICML | 2024 |
| [Editing Models with Task Arithmetic](summary/Ilharco_2023_TaskArithmetic.md) | Univ. of Washington / Microsoft Research / AI2 | ICLR | 2023 |
| [How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models](summary/Rust_2021_Tokenizer.md) | TU Darmstadt / Cambridge / DeepMind | ACL | 2021 |

---

## 6. AI 評価ベンチマーク・人間嗜好学習

ペアワイズ比較・相対順位・構造化 VQA など、**人間の判断を AI 評価/学習に取り込む**方法論。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks](summary/Yang_2026_relative.md) | Xidian University | CVPR | 2026 |
| [VQArt-Bench: A semantically rich VQA Benchmark for Art and Cultural Heritage](summary/Alfarano_2025_VQArt-Bench.md) | University of Zurich / Max Planck Society | ICCV Workshop | 2025 |
| [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](summary/Chiang_2024_ChatbotArena.md) | UC Berkeley / Stanford / UCSD | ICML | 2024 |

---

## 7. Vision 基盤モデル・セグメンテーション・3D 再構成

汎用 Vision 基盤モデル、概念条件付きセグメンテーション、両レベル最適化によるモデル整列、暗黙関数表現による 3D 再構成など。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [SAM 3: Segment Anything with Concepts](summary/Carion_2026_SAM3.md) | Meta Superintelligence Labs | arXiv preprint | 2026 |
| [BLO-Inst: Bi-Level Optimization Based Alignment of YOLO and SAM for Robust Instance Segmentation](summary/Zhang_2026_BLOInst.md) | UC San Diego | arXiv preprint | 2026 |
| [PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization](summary/Saito_2019_PIFu.md) | USC / Waseda / UC Berkeley / Pinscreen | ICCV | 2019 |

> 補助ノート: [SAM3 学習ノート](summary/SAM3_Study.md)（SAM3 のエンコーダ構成の読解メモ）

---

## 8. HRI/HCI ユーザースタディ

LLM 連携ロボット・支援技術・行動観察を用いた **対人インタラクションの実証研究 / システマティックレビュー**。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [How Do We Research Human-Robot Interaction in the Age of Large Language Models? A Systematic Review](summary/Wang_2026_LLM-HRI-Survey.md) | HKUST(GZ) / Zhejiang Univ. / SCAD / WCH / HKUST | CHI '26 | 2026 |
| [AuslanSpell: An Interactive Technology for Improving Auslan Fingerspelling Comprehension](summary/Stefanov_2026_AuslanSpell.md) | Monash University | CHI '26 | 2026 |
| [Gaze Behavior During a Long-Term, In-Home, Social Robot Intervention for Children with ASD](summary/Ramnauth_2025_GazeASD.md) | Yale University / Seattle Children's Research Institute / Univ. of Washington | HRI '25 (Best Paper) | 2025 |
| [Understanding Large-Language Model (LLM)-powered Human-Robot Interaction](summary/Kim_2024_LLM-HRI.md) | University of Wisconsin–Madison | HRI '24 | 2024 |

---

## 9. 行動・神経科学に基づく美的選好研究

行動指標や神経科学的アプローチで、**ヒトの美的選好の構造**を解析する研究。

| タイトル | 所属 | 出版元 | 年 |
|---|---|---|---|
| [Cross Domain Consistency of Aesthetic Preference-driven Social Behavior](summary/Pham_2026_CrossAes.md) | 広島大学 脳・心・感性科学研究センター / 理研 iTHEMS / Araya Inc. / 高知工科大学 / 玉川大学脳科学研究所 | bioRxiv preprint | 2026 |

---

## 凡例・運用ルール

- **クラスタリング基準**: ドメイン（応用先タスク）ではなく、**手法・方法論** で分類。例: 「美的評価タスク」ではなく「概念ベース解釈可能AI」「人間嗜好学習」などのカテゴリに置く。
- **クラスタ内の並び**: 年が新しいものを上位に。同年内は重要度や引用関係で適宜並べる。
- **新規追加時**: 該当クラスタの先頭に行を追加し、必要なら新しいクラスタを増設する。
- **所属**: 第一著者の所属を主としつつ、共同研究色が強い場合は主要機関を併記。
- **出版元**: 査読会議・ジャーナル名を優先。プレプリント時は `arXiv preprint` 等と明示。
