
# 機器學習輔助的逆合成規劃 (Machine Learning-Assisted Retrosynthesis Planning)

## 任務簡介

本專案旨在概述**逆合成規劃 (Retrosynthesis Planning)** 的核心概念及其在機器學習領域的當前研究現狀。

逆合成規劃是一種解決有機小分子合成路線設計的關鍵技術。其核心思想是**從目標分子出發，以逆向化學反應的方式，逐步將其分解為更簡單、更容易獲得的起始原料**。此過程旨在為化學合成繪製一張清晰的「地圖」，指導化學家如何從現有的簡單化合物，通過一系列化學反應，最終合成出指定的目標分子。

由於可能的化學轉化路徑數量巨大，且對反應機理的理解尚不完整，逆合成規劃即使對於經驗豐富的化學家來說也極具挑戰性。近年來，人工智慧（AI）與機器學習（ML）技術的發展催生了**機器學習輔助的逆合成規劃**，利用演算法從海量的已知化學反應數據中學習規律，以期實現合成路線設計的自動化。

---

## 研究現狀

根據相關文獻，機器學習輔助的逆合成規劃領域正處於快速發展階段。主流研究方法可分為三大類：**基於模板的方法**、**無模板的方法**和**半模板的方法**。

### 研究方法分類

#### 1. 基於模板的方法 (Template-based Methods)

* **核心思想**:
    將逆合成預測視為一個**反應模板檢索與應用**的問題。此類方法首先從大量已知化學反應中提取出代表分子結構變化核心規律的「反應模板」，然後透過匹配演算法為目標分子選擇最合適的模板來預測反應物。

* **代表模型**:
    * `NeuralSym` & `RetroSim`: 利用神經網路和分子相似性進行模板排序的早期模型。
    * `LocalRetro`: 受化學反應局部性啟發，評估局部反應模板在所有可能反應中心的適用性，並結合全域注意力機制來考量非局部效應。
    * `RetroKNN`: 透過建構「原子-模板」和「鍵-模板」知識庫，在推理時使用k近鄰（KNN）搜索來檢索模板，以提升對罕見反應的預測能力。

* **優缺點**:
    * **優點**: 預測結果具有較好的化學可解釋性和可靠性。
    * **缺點**: 性能受限於預先提取的模板庫，無法預測新反應，且模板庫過大時計算成本高。

#### 2. 無模板的方法 (Template-free Methods)

* **核心思想**:
    不依賴預定義的反應模板，而是將逆合成視為一個「翻譯」任務，直接將產物分子（以SMILES字串或圖結構表示）「翻譯」成反應物分子。

* **代表模型**:
    * **基於序列 (Seq2Seq)**:
        * `EditRetro`: 將逆合成重構為一個分子字串**編輯**任務，透過迭代地修改目標分子字串來生成前體化合物，而非從頭生成。
    * **基於圖 (Graph-to-Graph)**:
        * `NAG2G`: 一種基於Transformer的圖到圖模型，它結合分子的2D圖和3D構象，並透過節點對齊（Node-Alignment）整合產物-反應物的原子對映資訊。

* **優缺點**:
    * **優點**: 具有更強的泛化能力，能夠發現新的反應路徑。
    * **缺點**: 生成的分子有效性難以保證，且化學可解釋性較差。

#### 3. 半模板的方法 (Semi-template-based Methods)

* **核心思想**:
    試圖結合前兩種方法的優點。通常採用「兩步走」策略：首先，識別目標分子中可能發生反應的「反應中心」或「斷開位置」，將分子分解為中間體（synthon）；然後，再利用生成模型將中間體轉化為完整的反應物。

* **代表模型**:
    * `Graph2Edits`: 該模型受化學反應的「箭頭推進」形式啟發，將逆合成預測看作在產物圖上進行自回歸式編輯的過程（如刪除鍵、改變原子手性等）。

* **優缺點**:
    * **優點**: 在一定程度上平衡了模板方法的可靠性和無模板方法的靈活性。
    * **缺點**: 兩步過程通常獨立訓練，可能導致誤差在步驟間傳遞。

---

### 新興研究趨勢

除了上述主流方法，該領域還出現了以下幾個重要的研究方向：

* **融合更多的化學知識**:
    將核磁共振化學位移、鍵解離能等化學描述符整合到分子圖表示中，以提供更豐富的資訊供模型學習（如 `CIMG` 模型）。

* **改進分子表示方法**:
    更精細地對分子結構（例如 `O-GNN` 模型將**環結構**視為獨立元素）進行建模，以增強圖神經網路的表達能力。

* **多步逆合成路徑規劃**:
    將單步預測模型與**蒙特卡洛樹搜索 (MCTS)** 等搜索演算法相結合，以規劃出完整的、由多個反應步驟組成的合成路線（如 `Segler et al.` 的工作）。

* **大型語言模型（LLM）的應用**:
    利用大型語言模型生成合成路線的**文本描述**，並將其與分子的多模態資訊（2D圖、3D幾何結構）融合，透過上下文學習來提升預測準確性（如 `RetroInText` 框架）。

---

## 參考文獻

本文件內容基於以下學術文獻的綜合分析：

* `Zhang et al. - 2022 - Chemistry-informed molecular graph as reaction descriptor for machine-learned retrosynthesis plannin.pdf`
* `EditRetro-Retrosynthesis prediction with an iterative string editing model.pdf`
* `LocalRetro-deep-retrosynthetic-reaction-prediction-using-local-reactivity-and-global-attention.pdf`
* `Machine learning-assisted retrosynthesis planning- Current status and future prospects.pdf`
* `NAG2G-Node-Aligned Graph-to-Graph- Elevating Template-free Deep Learning Approaches in Single-Step Retrosynthesis.pdf`
* `O-GNN- INCORPORATING RING PRIORS INTO MOLECULAR MODELING.pdf`
* `RetroInText- A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning.pdf`
* `RetroKNN-Retrosynthesis Prediction with Local Template Retrieval.pdf`
* `Retrosynthesis prediction using an end-to-end graph generative architecture for molecular graph editing.pdf`
* `Segler et al. - 2018 - Planning chemical syntheses with deep neural networks and symbolic AI.pdf`
