# ===============================
# ⚾ 野球データ分析Webアプリ（PCA + クラスタ + 可視化 + 決定木）
# ===============================

# ===============================
# 🔧 ライブラリ読み込み
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ===============================
# 🌐 アプリタイトル
# ===============================
st.title("⚾ 野球データ分析アプリ")
st.markdown("""
PCAによる次元圧縮、クラスタリング、クラスタ平均可視化、
レーダーチャート、PCAバイプロット、決定木分析を一元管理。
""")

# ===============================
# 📤 ファイルアップロード
# ===============================
uploaded_file = st.file_uploader("ExcelまたはCSVをアップロード", type=["xlsx","csv"])

if uploaded_file:
    # ファイル読み込み
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, sheet_name="Sheet1", engine="openpyxl")
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("アップロードデータ")
    st.dataframe(df.head())

    # ===============================
    # 🔹 分析する特徴量選択
    # ===============================
    exclude_cols = ['No.', 'Name', 'Date']
    X_cols = [c for c in df.columns if c not in exclude_cols]
    selected_features = st.multiselect("分析する特徴量", X_cols, default=X_cols)

    if selected_features:
        # ===============================
        # 🔄 データ標準化
        # ===============================
        scaler = StandardScaler()
        X_std = scaler.fit_transform(df[selected_features])

        # ===============================
        # 📉 PCAによる2次元圧縮
        # ===============================
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X_std)
        df['PCA1'], df['PCA2'] = pca_result[:,0], pca_result[:,1]

        # ===============================
        # 🧠 KMeansクラスタリング
        # ===============================
        k = st.slider("クラスタ数を選択", 2, 6, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[['PCA1','PCA2']])

        # ===============================
        # 📊 PCA散布図（クラスタ色分け）
        # ===============================
        st.subheader("PCA散布図（クラスタ色分け）")
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster',
                        palette='viridis', s=100, edgecolor='black')
        for i in range(len(df)):
            plt.text(df.loc[i,'PCA1'], df.loc[i,'PCA2'], df.loc[i,'Name'],
                     fontsize=8, ha='center', va='center', color='black')
        st.pyplot(plt)

        # ===============================
        # 📊 クラスタ平均（棒グラフ＆ヒートマップ）
        # ===============================
        cluster_means = df.groupby('Cluster')[selected_features].mean()

        # 棒グラフ（クラスタごと）
        st.subheader("クラスタごとの特徴平均（棒グラフ）")
        plt.figure(figsize=(16,6))
        cluster_means.plot(kind='bar', rot=0, width=0.8)
        plt.ylabel('平均 Zスコア')
        plt.xlabel('クラスタ')
        plt.title('クラスタごとの特徴平均値')
        plt.legend(title='特徴量', bbox_to_anchor=(1.02,1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(plt)

        # ヒートマップ
        st.subheader("クラスタ × 特徴量 ヒートマップ")
        plt.figure(figsize=(12,6))
        sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap='RdYlBu_r', center=0)
        plt.title("クラスタ × 特徴量 ヒートマップ")
        plt.tight_layout()
        st.pyplot(plt)

        # ===============================
        # 🕸️ レーダーチャート
        # ===============================
        st.subheader("クラスタごとの特徴比較（レーダーチャート）")
        features = cluster_means.columns.tolist()
        num_vars = len(features)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
        plt.figure(figsize=(8,8))
        for i, label in enumerate(cluster_means.index):
            values = cluster_means.loc[label].tolist()
            values += values[:1]
            plt.polar(angles, values, label=f"クラスタ{label}", color=colors[i], linewidth=2)
            plt.fill(angles, values, alpha=0.15, color=colors[i])
        plt.xticks(angles[:-1], features, fontsize=10)
        plt.title("クラスタごとの特徴比較（レーダーチャート）", pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
        plt.tight_layout()
        st.pyplot(plt)

        # ===============================
        # 🌲 決定木によるクラスタ分類
        # ===============================
        st.subheader("決定木によるクラスタ分類")
        X = df[selected_features]
        y = df['Cluster']
        dt = DecisionTreeClassifier(max_depth=4, random_state=42)
        dt.fit(X, y)

        y_pred = dt.predict(X)
        acc = (y_pred==y).mean()
        st.write(f"決定木の学習精度: {acc:.3f}")

        plt.figure(figsize=(20,12))
        plot_tree(dt, feature_names=selected_features,
                  class_names=[f'クラスタ{i}' for i in range(k)],
                  filled=True, rounded=True)
        st.pyplot(plt)
