"""
Thử nghiệm mở rộng các mô hình và thuật toán Recommender System SOTA:
1. EASE (Embarrassingly Shallow Autoencoders - Steck, WWW 2019)
2. BM25 Item-Item Collaborative Filtering
3. iALS (Implicit Alternating Least Squares - Hu, Koren, Volinsky)
4. BPR-MF (Bayesian Personalized Ranking Matrix Factorization) với PyTorch
5. Hybrid EASE + Content-Based BOM (Super Hybrid)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.split import train_test_split_interactions
from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.evaluator import ModelEvaluator


# =====================================================================
# 1. EASE (Embarrassingly Shallow Autoencoders for Sparse Data - WWW 2019)
# =====================================================================
class EASERecommender(BaseRecommender):
    """
    EASE^R: Linear model with L2 regularization and zero diagonal constraint.
    Closed-form solution:
      G = X^T X + lambda * I
      P = G^(-1)
      B = - P / diag(P), diag(B) = 0
    """
    def __init__(self, reg_lambda: float = 200.0, name: str = "EASE (Autoencoder)"):
        super().__init__(name=name)
        self.reg_lambda = reg_lambda
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.B = None  # Trọng số Item-Item (I x I)
        self.X = None  # Ma trận User-Item train

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "EASERecommender":
        self._record_user_history(train_interactions)
        
        users = sorted(train_interactions["customer_id"].unique())
        items = sorted(item_features[item_features["status"] == "ACTIVE"]["product_id"].unique())
        
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}
        
        num_users = len(users)
        num_items = len(items)
        
        # Tạo ma trận dense/sparse X
        X = np.zeros((num_users, num_items), dtype=np.float32)
        for _, row in train_interactions.iterrows():
            u = row["customer_id"]
            it = row["product_id"]
            w = row.get("confidence_score", 1.0)
            if u in self.user2idx and it in self.item2idx:
                X[self.user2idx[u], self.item2idx[it]] = float(w)
                
        self.X = X
        
        # Tính Gram matrix G = X^T X
        G = X.T @ X  # (I x I)
        diag_indices = np.diag_indices(num_items)
        G[diag_indices] += self.reg_lambda
        
        # Nghịch đảo P = G^(-1)
        P = np.linalg.inv(G)
        
        # B = - P / diag(P), diag(B) = 0
        B = - P / np.diag(P)
        B[diag_indices] = 0.0
        self.B = B
        
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 5, exclude_purchased: bool = False):
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa fit!")
            
        purchased = self.get_user_purchased_items(user_id)
        
        if user_id not in self.user2idx:
            # Cold user
            return []
            
        u_idx = self.user2idx[user_id]
        u_vec = self.X[u_idx]  # (I,)
        scores = u_vec @ self.B  # (I,)
        
        rec_list = []
        for i_idx, score in enumerate(scores):
            pid = self.idx2item[i_idx]
            if exclude_purchased and pid in purchased:
                continue
            rec_list.append((pid, float(score)))
            
        rec_list.sort(key=lambda x: x[1], reverse=True)
        return rec_list[:top_k]


# =====================================================================
# 2. BM25 Item-Item Collaborative Filtering
# =====================================================================
class BM25ItemItemCF(BaseRecommender):
    """
    Item-Item CF với độ tương đồng BM25 thay cho Cosine thuần.
    Phạt các item quá phổ biến (hub) và thưởng các item niche cùng xuất hiện.
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75, name: str = "BM25 Item-Item CF"):
        super().__init__(name=name)
        self.k1 = k1
        self.b = b
        self.item2idx = {}
        self.idx2item = {}
        self.sim_matrix = None
        self.user_history_weights = {}

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "BM25ItemItemCF":
        self._record_user_history(train_interactions)
        
        items = sorted(item_features[item_features["status"] == "ACTIVE"]["product_id"].unique())
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}
        num_items = len(items)
        
        # User-item dict
        for u, grp in train_interactions.groupby("customer_id"):
            self.user_history_weights[u] = dict(zip(grp["product_id"], grp["confidence_score"]))
            
        # Ma trận User x Item
        users = sorted(train_interactions["customer_id"].unique())
        user2idx = {u: i for i, u in enumerate(users)}
        num_users = len(users)
        
        X = np.zeros((num_users, num_items), dtype=np.float32)
        for _, row in train_interactions.iterrows():
            u = row["customer_id"]
            it = row["product_id"]
            w = row.get("confidence_score", 1.0)
            if u in user2idx and it in self.item2idx:
                X[user2idx[u], self.item2idx[it]] = float(w)
                
        # Tính BM25 weights cho từng item trên user space:
        # doc = item, terms = users
        # X^T is (Item x User)
        item_user = X.T
        doc_lengths = item_user.sum(axis=1)  # (num_items,)
        avg_len = doc_lengths.mean() + 1e-9
        
        # IDF của user
        user_doc_freq = (item_user > 0).sum(axis=0)  # bao nhiêu item user đã tương tác
        idf = np.log(1.0 + (num_items - user_doc_freq + 0.5) / (user_doc_freq + 0.5))
        
        # BM25 tf
        len_norm = 1.0 - self.b + self.b * (doc_lengths[:, None] / avg_len)
        tf = (item_user * (self.k1 + 1.0)) / (item_user + self.k1 * len_norm + 1e-9)
        
        bm25_mat = tf * idf[None, :]  # (num_items x num_users)
        
        # Tương đồng Item-Item BM25
        sim = bm25_mat @ bm25_mat.T
        # Chuẩn hóa đường chéo
        diag = np.sqrt(np.diag(sim)) + 1e-9
        self.sim_matrix = sim / (diag[:, None] * diag[None, :])
        np.fill_diagonal(self.sim_matrix, 0.0)
        
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 5, exclude_purchased: bool = False):
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa fit!")
            
        purchased = self.get_user_purchased_items(user_id)
        if not purchased:
            return []
            
        u_weights = self.user_history_weights.get(user_id, {})
        scores = np.zeros(len(self.item2idx), dtype=np.float32)
        
        for pid, w in u_weights.items():
            if pid in self.item2idx:
                p_idx = self.item2idx[pid]
                scores += float(w) * self.sim_matrix[p_idx]
                
        rec_list = []
        for i_idx, score in enumerate(scores):
            pid = self.idx2item[i_idx]
            if exclude_purchased and pid in purchased:
                continue
            rec_list.append((pid, float(score)))
            
        rec_list.sort(key=lambda x: x[1], reverse=True)
        return rec_list[:top_k]


# =====================================================================
# 3. iALS (Implicit Alternating Least Squares)
# =====================================================================
class iALSRecommender(BaseRecommender):
    """
    Alternating Least Squares cho implicit feedback:
    Confidence c_ui = 1 + alpha * r_ui
    Preference p_ui = 1 nếu r_ui > 0, ngược lại 0
    """
    def __init__(self, factors: int = 16, alpha: float = 15.0, reg_lambda: float = 0.1, iterations: int = 15, name: str = "iALS (Implicit ALS)"):
        super().__init__(name=name)
        self.factors = factors
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.iterations = iterations
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.user_factors = None
        self.item_factors = None

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "iALSRecommender":
        self._record_user_history(train_interactions)
        
        users = sorted(train_interactions["customer_id"].unique())
        items = sorted(item_features[item_features["status"] == "ACTIVE"]["product_id"].unique())
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}
        
        num_users = len(users)
        num_items = len(items)
        f = self.factors
        
        # Ma trận R (implicit interaction weights)
        R = np.zeros((num_users, num_items), dtype=np.float32)
        for _, row in train_interactions.iterrows():
            u = row["customer_id"]
            it = row["product_id"]
            w = row.get("confidence_score", 1.0)
            if u in self.user2idx and it in self.item2idx:
                R[self.user2idx[u], self.item2idx[it]] = float(w)
                
        C = 1.0 + self.alpha * R
        P = (R > 0).astype(np.float32)
        
        np.random.seed(42)
        X = np.random.normal(0, 0.05, (num_users, f)).astype(np.float32)
        Y = np.random.normal(0, 0.05, (num_items, f)).astype(np.float32)
        
        I_f = np.eye(f, dtype=np.float32)
        
        for it in range(self.iterations):
            # Cập nhật User Factors: X
            YtY = Y.T @ Y
            for u in range(num_users):
                Cu = C[u]  # (num_items,)
                Pu = P[u]  # (num_items,)
                Cu_minus_1 = Cu - 1.0
                
                # Y^T (Cu - I) Y + Y^T Y + lambda * I
                # Chỉ lấy các item mà Cu > 1 (có tương tác)
                idx = np.where(Cu_minus_1 > 0)[0]
                if len(idx) > 0:
                    Y_idx = Y[idx]
                    A = YtY + (Y_idx.T * Cu_minus_1[idx]) @ Y_idx + self.reg_lambda * I_f
                    b = (Y_idx.T * Cu[idx]) @ Pu[idx]
                else:
                    A = YtY + self.reg_lambda * I_f
                    b = np.zeros(f, dtype=np.float32)
                X[u] = np.linalg.solve(A, b)
                
            # Cập nhật Item Factors: Y
            XtX = X.T @ X
            for i in range(num_items):
                Ci = C[:, i]
                Pi = P[:, i]
                Ci_minus_1 = Ci - 1.0
                idx = np.where(Ci_minus_1 > 0)[0]
                if len(idx) > 0:
                    X_idx = X[idx]
                    A = XtX + (X_idx.T * Ci_minus_1[idx]) @ X_idx + self.reg_lambda * I_f
                    b = (X_idx.T * Ci[idx]) @ Pi[idx]
                else:
                    A = XtX + self.reg_lambda * I_f
                    b = np.zeros(f, dtype=np.float32)
                Y[i] = np.linalg.solve(A, b)
                
        self.user_factors = X
        self.item_factors = Y
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 5, exclude_purchased: bool = False):
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa fit!")
        purchased = self.get_user_purchased_items(user_id)
        if user_id not in self.user2idx:
            return []
            
        u_idx = self.user2idx[user_id]
        u_vec = self.user_factors[u_idx]  # (f,)
        scores = self.item_factors @ u_vec  # (num_items,)
        
        rec_list = []
        for i_idx, score in enumerate(scores):
            pid = self.idx2item[i_idx]
            if exclude_purchased and pid in purchased:
                continue
            rec_list.append((pid, float(score)))
            
        rec_list.sort(key=lambda x: x[1], reverse=True)
        return rec_list[:top_k]


# =====================================================================
# 4. BPR-MF (Bayesian Personalized Ranking - PyTorch)
# =====================================================================
class PyTorchBPR(nn.Module):
    def __init__(self, num_users: int, num_items: int, factors: int = 16):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, factors)
        self.item_emb = nn.Embedding(num_items, factors)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i, j):
        u_vec = self.user_emb(u)
        i_vec = self.item_emb(i)
        j_vec = self.item_emb(j)
        x_ui = (u_vec * i_vec).sum(dim=-1)
        x_uj = (u_vec * j_vec).sum(dim=-1)
        return x_ui - x_uj


class BPRMFRecommender(BaseRecommender):
    def __init__(self, factors: int = 16, lr: float = 0.01, epochs: int = 30, reg: float = 1e-4, name: str = "BPR-MF (PyTorch)"):
        super().__init__(name=name)
        self.factors = factors
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.model = None

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "BPRMFRecommender":
        self._record_user_history(train_interactions)
        
        users = sorted(train_interactions["customer_id"].unique())
        items = sorted(item_features[item_features["status"] == "ACTIVE"]["product_id"].unique())
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: it for it, i in self.item2idx.items()}
        
        num_users = len(users)
        num_items = len(items)
        
        # Tập positive items cho mỗi user
        user_pos_items = {u_idx: set() for u_idx in range(num_users)}
        pos_triplets = []
        for _, row in train_interactions.iterrows():
            u = row["customer_id"]
            it = row["product_id"]
            if u in self.user2idx and it in self.item2idx:
                u_idx = self.user2idx[u]
                i_idx = self.item2idx[it]
                user_pos_items[u_idx].add(i_idx)
                pos_triplets.append((u_idx, i_idx))
                
        # Khởi tạo mô hình PyTorch
        self.model = PyTorchBPR(num_users, num_items, self.factors)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        
        self.model.train()
        n_samples = len(pos_triplets)
        all_item_indices = np.arange(num_items)
        
        for epoch in range(self.epochs):
            # Tạo negative samples
            np.random.shuffle(pos_triplets)
            users_batch = []
            pos_batch = []
            neg_batch = []
            
            for u_idx, i_idx in pos_triplets:
                # Sample negative item
                neg_item = np.random.randint(0, num_items)
                while neg_item in user_pos_items[u_idx]:
                    neg_item = np.random.randint(0, num_items)
                users_batch.append(u_idx)
                pos_batch.append(i_idx)
                neg_batch.append(neg_item)
                
            u_t = torch.tensor(users_batch, dtype=torch.long)
            i_t = torch.tensor(pos_batch, dtype=torch.long)
            j_t = torch.tensor(neg_batch, dtype=torch.long)
            
            optimizer.zero_grad()
            diff = self.model(u_t, i_t, j_t)
            loss = -torch.log(torch.sigmoid(diff) + 1e-9).mean()
            loss.backward()
            optimizer.step()
            
        self.model.eval()
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 5, exclude_purchased: bool = False):
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa fit!")
        purchased = self.get_user_purchased_items(user_id)
        if user_id not in self.user2idx:
            return []
            
        u_idx = self.user2idx[user_id]
        with torch.no_grad():
            u_t = torch.tensor([u_idx], dtype=torch.long)
            u_vec = self.model.user_emb(u_t)  # (1, f)
            all_i = self.model.item_emb.weight  # (num_items, f)
            scores = (u_vec * all_i).sum(dim=-1).cpu().numpy()
            
        rec_list = []
        for i_idx, score in enumerate(scores):
            pid = self.idx2item[i_idx]
            if exclude_purchased and pid in purchased:
                continue
            rec_list.append((pid, float(score)))
            
        rec_list.sort(key=lambda x: x[1], reverse=True)
        return rec_list[:top_k]


# =====================================================================
# 5. Super Hybrid: Kết hợp EASE + Content-Based BOM + Item-Item + Pop
# =====================================================================
class SuperHybridRecommender(BaseRecommender):
    def __init__(self, ease_model: EASERecommender, cb_model: ContentBasedRecommender, pop_model: PopularityRecommender, name: str = "Super Hybrid (EASE + BOM + Pop)"):
        super().__init__(name=name)
        self.ease = ease_model
        self.cb = cb_model
        self.pop = pop_model

    def fit(self, train_interactions: pd.DataFrame, item_features: pd.DataFrame) -> "SuperHybridRecommender":
        self._record_user_history(train_interactions)
        self.is_fitted = True
        return self

    def recommend(self, user_id: str, top_k: int = 5, exclude_purchased: bool = False):
        purchased = self.get_user_purchased_items(user_id)
        
        # Điểm EASE
        ease_recs = dict(self.ease.recommend(user_id, top_k=67, exclude_purchased=False))
        # Điểm Content-based
        cb_recs = dict(self.cb.recommend(user_id, top_k=67, exclude_purchased=False))
        # Điểm Popularity
        pop_recs = dict(self.pop.recommend(user_id, top_k=67, exclude_purchased=False))
        
        # Normalize
        def norm(d):
            if not d:
                return {}
            vals = list(d.values())
            mi, ma = min(vals), max(vals)
            if ma - mi < 1e-9:
                return {k: 0.5 for k in d}
            return {k: (v - mi) / (ma - mi) for k, v in d.items()}
            
        norm_ease = norm(ease_recs)
        norm_cb = norm(cb_recs)
        norm_pop = norm(pop_recs)
        
        all_pids = set(norm_ease.keys()) | set(norm_cb.keys()) | set(norm_pop.keys())
        
        combined = []
        for pid in all_pids:
            if exclude_purchased and pid in purchased:
                continue
            s_ease = norm_ease.get(pid, 0.0)
            s_cb = norm_cb.get(pid, 0.0)
            s_pop = norm_pop.get(pid, 0.0)
            
            # Trọng số: 0.50 EASE + 0.35 Content BOM + 0.15 Pop
            final_s = 0.50 * s_ease + 0.35 * s_cb + 0.15 * s_pop
            combined.append((pid, float(final_s)))
            
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


def main():
    print("=" * 70)
    print("  EXPERIMENT: KHẢO SÁT & BENCHMARK CÁC MÔ HÌNH THUẬT TOÁN SOTA  ")
    print("=" * 70)
    
    loader = DataLoader()
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    interaction_df, user_item_mat = prep.build_interaction_matrix()
    item_features = prep.build_item_features()
    active_items = item_features[item_features["status"] == "ACTIVE"]["product_id"].tolist()
    
    train_df, test_df = train_test_split_interactions(interaction_df, test_ratio=0.2, random_state=42)
    
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    print(f"Tổng số sản phẩm active: {len(active_items)}")
    
    print("\n--- Đang huấn luyện các mô hình ---")
    pop_m = PopularityRecommender().fit(train_df, item_features)
    cb_m = ContentBasedRecommender().fit(train_df, item_features)
    cf_ii = CollaborativeRecommender(algorithm="item_item").fit(train_df, item_features)
    cf_svd = CollaborativeRecommender(algorithm="svd", n_components=12).fit(train_df, item_features)
    hybrid_v1 = HybridRecommender(cf_weight=0.50, content_weight=0.35, pop_weight=0.15).fit(train_df, item_features)
    
    # New Models
    print("1. Huấn luyện EASE (Autoencoder)...")
    ease_m = EASERecommender(reg_lambda=200.0).fit(train_df, item_features)
    
    print("2. Huấn luyện BM25 Item-Item CF...")
    bm25_m = BM25ItemItemCF(k1=1.2, b=0.75).fit(train_df, item_features)
    
    print("3. Huấn luyện iALS (Implicit ALS)...")
    ials_m = iALSRecommender(factors=16, alpha=15.0, reg_lambda=0.1, iterations=15).fit(train_df, item_features)
    
    print("4. Huấn luyện BPR-MF (PyTorch Pairwise Ranking)...")
    bpr_m = BPRMFRecommender(factors=16, lr=0.02, epochs=35, reg=1e-4).fit(train_df, item_features)
    
    print("5. Xây dựng Super Hybrid (EASE + BOM + Pop)...")
    super_hybrid = SuperHybridRecommender(ease_m, cb_m, pop_m).fit(train_df, item_features)
    
    all_models = {
        "1. Popularity (Baseline)": pop_m,
        "2. Content-Based (BOM)": cb_m,
        "3. Item-Item Cosine CF (v1.1)": cf_ii,
        "4. SVD Matrix Factorization": cf_svd,
        "5. Hybrid v1.1 (CF+BOM+Pop)": hybrid_v1,
        "6. BM25 Item-Item CF": bm25_m,
        "7. EASE (Shallow Autoencoder)": ease_m,
        "8. iALS (Implicit ALS)": ials_m,
        "9. BPR-MF (PyTorch)": bpr_m,
        "10. Super Hybrid (EASE+BOM+Pop)": super_hybrid,
    }
    
    print("\n" + "=" * 70)
    print("  KẾT QUẢ ĐỐI ĐẦU BENCHMARK TOÀN DIỆN (Top-5)  ")
    print("=" * 70)
    evaluator = ModelEvaluator(k=5)
    results = evaluator.compare_models(all_models, test_df, active_items)
    
    # Định dạng hiển thị đẹp
    results["Recall@5"] = results["Recall@5"].apply(lambda x: f"{x*100:.2f}%")
    results["Precision@5"] = results["Precision@5"].apply(lambda x: f"{x*100:.2f}%")
    results["NDCG@5"] = results["NDCG@5"].apply(lambda x: f"{x:.4f}")
    results["MAP@5"] = results["MAP@5"].apply(lambda x: f"{x:.4f}")
    results["Coverage"] = results["Coverage"].apply(lambda x: f"{x*100:.2f}%")
    
    print(results.to_markdown(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
