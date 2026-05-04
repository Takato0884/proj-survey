# The Linear Representation Hypothesis and the Geometry of Large Language Models

**著者**: Kiho Park, Yo Joong Choe, Victor Veitch
**所属**: University of Chicago
**出版**: ICML (International Conference on Machine Learning) 2024

---

## 1. 論文の目的または目標は何ですか？

<p align="center"><img src="../image/template.jpg" width="60%"></p>

「線形表現仮説(Linear Representation Hypothesis)」とは、言語モデルにおいて高次元の意味的コンセプト(言語が英語かフランス語か、性別が男性か女性か、など)が表現空間における**線形な方向**として符号化されている、という考え方である。この仮説が正しければ、モデルの解釈・制御を線形代数的な操作で行えるため極めて有用だが、これまで「線形表現」という言葉自体が曖昧に使われてきた。

本論文は次の 2 つの未解決問題に正面から取り組むことを目的とする:

1. **「線形表現」とは具体的に何を意味するのか?** 既存研究では以下の 3 つの異なる解釈が混在していた:
   - **Subspace 解釈**: コンセプトは方向(部分空間)として表される(Mikolov et al., 2013c など)
   - **Measurement 解釈**: コンセプトは線形プローブで予測可能(Nanda et al., 2023 など)
   - **Intervention 解釈**: コンセプトはステアリングベクトルの加算で変更可能(Turner et al., 2023 など)

2. **表現空間における幾何(類似度・射影)をどう定義するか?** これらの操作には内積が必要だが、どの内積を使うべきかは自明ではない。

論文の目標は、**反事実(counterfactual)の言語**を用いて線形表現を厳密に定式化し、3 つの解釈が単一の幾何構造のもとで統一されることを示すことである。

---

## 2. トピックを探求するために使用された方法またはアプローチは何ですか？

### 理論的アプローチ

**1) Unembedding 表現と Embedding 表現の定式化**

言語モデルには 2 つの表現空間がある:出力単語の空間 $\Gamma$(unembedding)と入力コンテキストの空間 $\Lambda$(embedding)であり、$P(y \mid x) \propto \exp(\lambda(x)^\top \gamma(y))$ で関係づけられる。論文は**反事実ペア**を用いて両空間でのコンセプト表現を定義した:

- **Unembedding 表現** $\bar{\gamma}_W$:反事実単語ペア(例: king/queen, man/woman)の差ベクトルが共通して向く方向
- **Embedding 表現** $\bar{\lambda}_W$:W に関してだけ違うコンテキストペアの差ベクトルが共通して向く方向

そして次の 2 つの定理を証明した:
- **Theorem 2.2**:$\bar{\gamma}_W$ は線形プローブとして機能する(Subspace ⇔ Measurement)
- **Theorem 2.5**:$\bar{\lambda}_W$ は介入ベクトルとして機能する(Subspace ⇔ Intervention)

**2) Causal Inner Product の導入**

訓練からは表現が可逆な線形変換のぶんだけ不定になるため、Euclidean 内積で類似度を測る根拠がない。論文は「**因果分離可能なコンセプトは直交する**」という追加原理を要請する内積、**causal inner product**(Definition 3.1)を導入。

- **Theorem 3.2(統一)**:causal inner product のもとで、Riesz 同型を介して $\bar{\gamma}_W \leftrightarrow \bar{\lambda}_W$ が同一視される
- **Theorem 3.4(具体形)**:語彙からの一様サンプリングに関する独立性仮定(Assumption 3.3)のもとで、$M = \text{Cov}(\gamma)^{-1}$ という閉形式が得られる:
$$\langle \bar{\gamma}, \bar{\gamma}' \rangle_C = \bar{\gamma}^\top \text{Cov}(\gamma)^{-1} \bar{\gamma}'$$

### 実験的検証

- **モデル**:LLaMA-2-7B(主)、Gemma-2B(比較)
- **コンセプト**:全 27 種類(BATS 3.0 由来の文法・意味的コンセプト 22 種、言語ペア 4 種、頻度 1 種)。詳細は **Table 2** を参照
- **検証項目**:
  1. 反事実ペアの差が共通方向を向くか(Subspace 解釈)
  2. 推定した causal inner product のもとで因果分離可能なコンセプトが直交するか
  3. $\bar{\gamma}_W$ が Wikipedia 由来のテキストに対して線形プローブとして機能するか(Measurement 解釈)
  4. $\bar{\lambda}_W = \text{Cov}(\gamma)^{-1}\bar{\gamma}_W$ を加算することでターゲットコンセプトのみを変更できるか(Intervention 解釈)

---

## 3. 主要な結果または発見は何でしたか？

**(1) 反事実ペアは共通方向を向く(Subspace 解釈の経験的支持)**

27 コンセプト中、`thing⇒part` を除く 26 コンセプトで、反事実ペアの差ベクトルがランダムペアと比べて顕著に大きな射影値を示した。これは線形表現仮説が広範に成り立つことを示している(**Figure 2** および付録の **Figure 7** を参照)。

**(2) Causal inner product は意味構造を捉える**

推定した causal inner product のもとで、ほとんどの因果分離可能なコンセプトペアが近似的に直交する一方、意味的に関連するコンセプト群(動詞関連の 10 概念、言語ペア 4 概念など)はブロック対角構造を示した(**Figure 3** を参照)。例えば `lower⇒upper`(大文字化)は、英独仏西の言語ペアと非自明な内積を持ち、これは英独が独自の大文字化規則を持つ一方で仏西が類似していることと整合する。

**(3) Euclidean 内積は LLaMA-2 では偶然機能するが Gemma-2B では機能しない**

LLaMA-2 では Euclidean 内積も causal inner product と類似のパターンを示した(初期化や暗黙的正則化の効果と推測)。しかし Gemma-2B では Euclidean 内積は意味構造をほとんど捉えず、causal inner product のみが正しい構造を取り出した。これは causal inner product のモデル非依存性を示している(**Figure 8, Figure 9** を参照)。

**(4) Subspace 表現は線形プローブとして機能する**

Wikipedia の仏語・西語コンテキストに対し、$\bar{\gamma}_{\text{French⇒Spanish}}$ で射影すると分布が明確に分離した。一方、無関係な $\bar{\gamma}_{\text{male⇒female}}$ では分離しなかった(**Figure 4** を参照)。

**(5) Subspace 表現から介入ベクトルを構成可能**

$\bar{\lambda}_W = \text{Cov}(\gamma)^{-1}\bar{\gamma}_W$ をコンテキスト埋め込みに加算すると、ターゲットコンセプト W の確率のみが変化し、因果分離可能な他のコンセプトの確率は不変であった(**Figure 5** を参照)。例えば `male⇒female` 方向への介入は "queen" vs "king" の比を変えるが "King" vs "king"(大文字化)の比は変えない。

**(6) 介入により出力単語が劇的に変化する**

コンテキスト "Long live the " に対し、$\alpha\bar{\lambda}_{\text{male⇒female}}$ を加算する強さ $\alpha$ を 0 から 0.4 まで増やすと、最尤の次単語が "king" から "queen" に変化した(**Table 1** を参照)。

---

## 4. 結論は何であり、なぜそれが重要なのですか？

### 結論

本論文は線形表現仮説の 3 つの解釈(Subspace、Measurement、Intervention)が、**反事実ペアによる定式化**と **causal inner product による幾何構造**のもとで統一的に理解できることを示した。具体的には:

- 出力空間の Unembedding 表現 $\bar{\gamma}_W$ は線形プローブ(測定)として機能する(Theorem 2.2)
- 入力空間の Embedding 表現 $\bar{\lambda}_W$ は介入ベクトル(制御)として機能する(Theorem 2.5)
- Causal inner product を選ぶことで、両者は Riesz 同型を通じて同一視され、すべての解釈が一つの幾何構造に統合される(Theorem 3.2)
- 内積の具体形は $\text{Cov}(\gamma)^{-1}$ という閉形式で構成可能で、これは Mahalanobis 距離・白色化と密接に関連する(Theorem 3.4)

実験的にも LLaMA-2 でこれらの理論的予測が確認され、特に「Unembedding 表現から介入ベクトルを構成する」という応用が実用上有効であることが示された。

### 重要性

**学術的貢献**:

- 言語モデル解釈研究において曖昧に使われていた「線形表現」概念を厳密に定式化し、識別可能性(identifiability)の文脈に位置づけた
- モデルを再訓練するのではなく、**観測手段(内積)を工夫することで隠れた意味的幾何構造を取り出す**という新しいアプローチを提示
- Causal representation learning の分野に対し、LLM という具体的設定での識別可能性の事例を提供
- 「内積の選択が表現解釈の本質」という視点を明確にし、Euclidean 内積の暗黙の使用に理論的根拠がないことを指摘

**実用的貢献**:

- 反事実単語ペアという入手しやすいデータから、簡単に介入ベクトル(steering vector)を構成する手法を提供。コンテキストペアを直接探す必要がなく、実装が容易
- LLM の解釈・制御における新しい技術的基盤となり、安全性・アライメント研究にも応用可能

**残された課題**:

- 中間層の活性化や注意機構の解釈には未対応(本論文は最終層に限定)
- Causal inner product の自由度パラメータ $D$ を一意に決定する原理は未提案
- 早期停止戦略・教師なし設定での応用、より大規模モデルへの拡張、視覚言語モデル等への一般化が今後の課題