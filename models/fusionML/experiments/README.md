# fusionMLv2

## 1. Background

Anaerobic digester performance depends partly on the composition and activity of microbial communities. Microbial sequencing data offer a way to characterize these communities and investigate their associations with digester performance and operational warning indicators. In this project, the inputs are taxonomic abundance profiles and phylogenetic relationships derived from microbial marker-gene data. These profiles provide information about community composition; they do not directly measure metabolic activity or establish functional equivalence between organisms.

Two approaches provide the main predictive references:

- **Single-rank abundance models:** relative abundances aggregated at a fixed taxonomic rank, such as genus or family, are supplied to models such as GLMNET or XGBoost.
- **Phylogeny-aware models:** methods such as Phylo-Spec, MetaDR, and DeepPhylo incorporate phylogenetic structure alongside abundance information to learn predictive representations.

Microbial datasets often contain many candidate features but relatively few samples. Repeated observations from the same digester also mean that the number of independent sites is smaller than the number of sample rows. Correlated abundance patterns, sparsity, and the compositional nature of relative abundances complicate estimation and interpretation. However, high dimensionality or multicollinearity does not automatically imply poor prediction: fixed-rank aggregation already reduces dimensionality, and regularized models can stabilize estimates from correlated predictors. The [elastic-net formulation](https://academic.oup.com/jrsssb/article/67/2/301/7109482) explicitly addresses correlated predictors and settings with more predictors than observations.

Phylogeny provides a biological basis for proposing aggregations because related organisms can share functional traits. The relationship is trait-dependent: methanogenesis, for example, can be conserved across deeper clades than many substrate-utilization traits. Phylogenetic proximity therefore provides a prior for grouping rather than a guarantee of functional or predictive equivalence. [Martiny et al.](https://www.nature.com/articles/ismej2012160) describe this variation in microbial trait conservation.

Dimensionality reduction and interpretable tree-guided aggregation are established research directions. [DeepPhylo](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615782/) learns compact abundance and phylogenetic representations; [tree-guided variable fusion](https://doi.org/10.1080/01621459.2016.1270213) and [trac](https://www.nature.com/articles/s41598-021-93645-3) explicitly learn predictive microbial aggregations. The proposed contribution of fusionMLv2 is to investigate the added value of combining phylogenetic and empirical abundance relationships, with separately adjustable aggregation behavior for archaea and bacteria, for interpretable compression of digester microbial data.

## 2. Problem Statement

### 2.1 Domain & Data Bottlenecks (현장의 데이터적/기술적 병목)

혐기성 소화조(Anaerobic Digester)의 미생물 군집 시퀀싱 데이터(NGS)를 예측 모델에 활용할 때 다음과 같은 구조적 한계가 존재합니다.

1. **극심한 차원의 불균형 ($n \ll p$):** 독립된 소화조 지점(Site)의 수는 적은 반면, 측정되는 미생물 분류군(Taxa)의 수는 수백~수천 개에 달할 수 있어 모델 과적합과 불안정한 피처 선택의 위험이 큽니다. 동일 지점에서 여러 계절에 걸쳐 수집한 관측치는 서로 연관될 수 있으므로, 전체 샘플 수와 독립된 지점 수를 구분해야 합니다. 특히 새로운 지점으로의 일반화에서는 독립된 학습 지점의 부족이 중요한 병목입니다.

2. **상대 존재량(Compositional Data)의 종속성과 다중공선성:** 상대 존재량은 합이 1(100%)로 제한되므로 피처 간 종속성이 존재하며, 실제 생태적 공변동과 상대화 과정에서 발생한 연관성을 구분하기 어렵습니다. 함께 변동하는 분류군은 개별 회귀계수나 피처 기여도의 추정을 불안정하게 만들 수 있습니다. 따라서 일반 상관계수만으로 중복성을 판단하기보다 로그비 변동이나 존재량 비례성(Proportionality)을 고려해야 합니다. 다만 다중공선성이 반드시 예측 성능 저하를 의미하는 것은 아니며, 정규화 모델이 이미 이를 일부 완화할 수 있습니다. [Lovell et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075), [Zou and Hastie](https://academic.oup.com/jrsssb/article/67/2/301/7109482)

### 2.2 Limitations of Existing Approaches (기존 문헌의 한계 및 Research Gap)

기존 연구들은 고차원 미생물 데이터를 처리하기 위해 다양한 접근법을 제안했습니다. 본 연구는 차원 축소 자체를 새로운 기여로 주장하기보다, 다음 접근법들이 제공하는 표현의 해상도, 해석 가능성, 그리고 소화조 지점 간 재현성을 비교하고자 합니다.

1. **고정 분류 등급 기반 ML 모델 (Single-Rank Abundance Models)**

   Genus나 Family 등 특정 분류 등급(Rank)으로 존재량을 일괄 통합한 뒤 GLMNET, XGBoost 등의 모델에 입력합니다.

   **한계:** 기능적 특성이 보존되는 계통학적 깊이는 특성에 따라 다릅니다. 예를 들어 메탄생성은 비교적 깊은 Clade에서 보존되는 반면, 여러 탄소원 이용 특성은 더 얕은 Clade에 분포합니다. 따라서 모든 계통군에 단일 해상도를 적용하면 예측에 필요한 세부 차이를 제거하거나 불필요하게 많은 피처를 유지할 수 있습니다. 이는 고정 등급 모델이 항상 열등하다는 의미가 아니라, 계통군별 적응적 통합의 이점을 검증할 근거입니다. [Martiny et al.](https://www.nature.com/articles/ismej2012160)

2. **계통수 기반 딥러닝/임베딩 모델 (Phylogeny-Aware Neural Models)**

   DeepPhylo, Phylo-Spec, MetaDR 등은 계통수와 존재량 정보를 활용하여 예측 표현을 학습합니다. 특히 DeepPhylo는 저차원 계통학적 임베딩과 존재량 표현을 명시적으로 사용합니다. [DeepPhylo](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615782/)

   **한계:** 학습된 잠재 표현이 구성 분류군을 직접 열거할 수 있는 명시적 미생물 그룹과 일대일로 대응하는 것은 아닙니다. 피처 중요도나 계통학적 기여도를 설명하는 기능이 있더라도, 이를 통해 압축 그룹의 구성원과 그 그룹이 공유하는 생물학적 기능이 자동으로 확립되지는 않습니다. 따라서 연구 공백은 이러한 모델에 해석 가능성이 전혀 없다는 데 있지 않고, **구성원이 명확한 압축 그룹의 크기, 안정성, 예측 정보 보존을 함께 평가하는 것**에 있습니다. Phylo-Spec도 계통 구조를 통한 피처 중요도 해석을 제공하므로 이를 단순히 해석 불가능한 블랙박스로 규정하지 않습니다. [Phylo-Spec](https://doi.org/10.1128/msystems.01453-25)

3. **기존 계통수 기반 변수 통합 모델 (Tree-Guided Variable Fusion Models)**

   `trac`과 Tree-Guided Variable Fusion은 계통수 또는 분류학적 계층을 이용하여 피처를 통합하고 해석 가능한 예측 그룹을 생성합니다. 이들은 이미 예측 목표에 따른 적응적 통합을 수행하므로, 단순히 계통수를 이용해 차원을 줄이는 것만으로는 새로운 기여가 성립하지 않습니다. [Tree-Guided Variable Fusion](https://doi.org/10.1080/01621459.2016.1270213), [trac](https://www.nature.com/articles/s41598-021-93645-3)

   본 연구가 이들 접근법을 기준으로 검증할 확장 지점은 다음과 같습니다.

   - **계통학적 가이드와 경험적 가이드의 결합:** 위 방법들의 계층 구조에 기반한 통합을 비교 기준으로 삼아, 실제 학습 데이터의 존재량 비례성으로 구성한 관계와 계통학적 관계를 함께 사용하고 그 상대적 가중치를 조절하는 것이 추가적인 이점을 제공하는지 검증합니다. 계통학적 유사성, 존재량 중복성, 예측 효과의 유사성은 서로 구분합니다.
   - **도메인별 그룹화 규칙의 필요성:** 아키아(Archaea)와 박테리아(Bacteria)는 유용한 통합 해상도와 예측 관계가 다를 수 있습니다. 두 도메인에 공통 매개변수를 적용하는 경우와 별도 매개변수를 학습하는 경우를 비교하여 도메인 적응의 필요성을 검증합니다. 본 데이터의 두 도메인은 별도 분석으로 얻은 상대 존재량이므로, Read 수를 도메인 간 생물량 비율로 해석하지 않습니다.
   - **새로운 소화조 지점에서의 일반화와 그룹 안정성:** 기존 방법을 소화조 데이터에 적용할 때, 지점 단위로 분리한 검증에서 예측력이 유지되는지와 학습 지점의 구성이 달라져도 유사한 그룹이 재현되는지를 함께 평가해야 합니다. 고정된 그룹을 새 지점에 적용하는 것과, 다른 학습 지점들로부터 그룹을 다시 학습했을 때의 안정성은 별개의 검증 대상입니다. 이는 모든 선행 연구에 안정성 검증이 없다는 주장이 아니라, 본 적용 분야에서 추가로 검증할 문제입니다.

### 2.3 Core Research Question (핵심 연구 질문)

따라서 본 연구가 해결하고자 하는 핵심 문제는 다음과 같습니다.

> **"아키아와 박테리아의 그룹화 매개변수를 분리하고, 계통수 정보와 학습 데이터의 실제 존재량 비례성을 적응적으로 결합함으로써, 새로운 소화조 지점(Unseen Sites)에 대한 예측력을 사전에 정의한 허용 손실 범위 내에서 유지하면서도, 기존 압축 기법보다 더 안정적이고 구성 분류군을 해석할 수 있는 최소 규모의 미생물 표현(Representation)을 구축할 수 있는가?"**

여기서 최소 규모는 예측 정보 보존 조건을 만족하는 후보 중 가장 작은 표현을 의미하며, 모든 미생물을 하나의 피처로 통합한다는 의미는 아닙니다. 압축 크기, 그룹 안정성, 예측 성능을 함께 비교해야 하며, 학습된 그룹의 생물학적 역할은 독립적인 근거로 검증해야 합니다. 목표 변수의 정보를 이용해 그룹을 학습하는 경우에는 우선 목표별 표현으로 평가하고, 여러 목표에 공통으로 사용할 수 있는 표현인지는 별도로 검증합니다.

## 3. Research Objective

**The primary objective is to develop and evaluate a compact, stable, and interpretable microbial representation that preserves useful predictive information about anaerobic digester performance across independent sites.**

Predictive superiority is a secondary hypothesis. Prediction remains a required downstream validation of the representation: compression is useful only when the retained information and interpretability justify any loss in predictive performance.

The specific objectives are to:

- **Learn adaptive microbial aggregations.** Investigate phylogenetic relationships, empirical abundance proportionality, and their combination as guides to grouping, including separately adjustable behavior for archaea and bacteria. Establish the contribution of each component through ablation comparisons, including shuffled-phylogeny controls.
- **Quantify the prediction–compression tradeoff.** Measure held-out predictive performance across representation sizes. Report the number of input taxa before and after filtering, active group features, and any residual or interaction terms. Compare performance at similar feature budgets and the feature counts needed to achieve similar predictive performance. Define an acceptable predictive-loss margin before confirmatory evaluation; a nonsignificant difference alone does not establish equivalence.
- **Assess reproducibility and interpretability.** Measure group-membership stability under resampling of whole sites and variation in training size. Report group constituents and their fitted associations. Treat functional interpretations as hypotheses requiring independent biological evidence; phylogenetic coherence imposed during grouping is insufficient validation by itself.
- **Validate the intended downstream use.** Distinguish a target-specific supervised aggregation from a reusable representation. If groups are supplied to downstream models such as GLMNET or XGBoost, compare those learners on the original and compressed inputs. Learn outcome-informed groups only within training partitions.
- **Benchmark against relevant alternatives.** Include fixed-rank aggregation, sparse feature selection, an appropriate log-ratio PCA baseline, and existing tree-guided aggregation such as trac. Retain single-rank ML and phylogeny-aware models as predictive references. Use a common cohort, target definitions, and site-grouped nested cross-validation, with all data-adaptive preprocessing and selection confined to the relevant training folds.
- **Characterize dependence on independent training data.** Use learning curves to assess predictive retention and group stability as the number of training sites increases. Treat comparisons between collecting additional sites and additional seasons as a separate sampling-design question requiring direct evaluation.

A successful outcome would demonstrate substantially fewer interpretable features with an acceptable predictive loss, improved stability, or better prediction at a comparable feature budget relative to relevant baselines. Fewer groups with substantial information loss, unstable membership, or no advantage over simpler aggregation would not establish successful compression. Reduced representation size also does not, by itself, imply that fewer taxa need to be measured experimentally.
