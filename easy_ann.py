import math
import random
import time

# Vectorクラスを活用してPCAを実装する
# Vectorクラスの定義
class Vector:
    def __init__(self, values):
        self.values = values

    def dot(self, other):
        """内積を計算"""
        return sum(v1 * v2 for v1, v2 in zip(self.values, other.values))

    def distance_squared(self, other):
        """2つのベクトル間の距離の二乗を計算"""
        return sum((v1 - v2) ** 2 for v1, v2 in zip(self.values, other.values))

    def distance(self, other):
        """2つのベクトル間のユークリッド距離を計算"""
        return math.sqrt(self.distance_squared(other))

    def norm(self):
        """ベクトルのノルム（長さ）を計算"""
        return math.sqrt(sum(v**2 for v in self.values))

    def __str__(self):
        return str(self.values)


# K-meansクラスタリングの実装（K-means++アルゴリズムを使用）
class KMeans:
    def __init__(self, k, embedding_vectors, max_iterations=100):
        self.k = k
        self.embedding_vectors = embedding_vectors  # 埋め込みベクトルを使用
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = []

    def _initialize_centroids(self, data_points):
        """初期重心をK-means++アルゴリズムに基づいて選定"""
        # 1つ目の重心をデータポイントからランダムに選択
        centroids = [random.choice(data_points)]  # 最初の重心をランダムに選ぶ

        for _ in range(1, self.k):
            # 各データポイントに対して最も近い既存の重心までの距離の二乗を計算
            distances = [
                min(point.distance_squared(centroid) for centroid in centroids)
                for point in data_points
            ]

            # 距離の二乗に基づいた確率で次の重心を選択
            total_distance = sum(distances)
            probabilities = [d / total_distance for d in distances]

            # 確率に基づいて次の重心を選ぶ
            selected_point = random.choices(data_points, probabilities)[0]
            centroids.append(selected_point)

        return centroids

    def _assign_clusters(self, data_points, centroids):
        """データポイントを最も近いクラスタに割り当てる"""
        clusters = [[] for _ in range(self.k)]
        for point in data_points:
            distances = [point.distance_squared(centroid) for centroid in centroids]
            closest_centroid_idx = distances.index(min(distances))
            clusters[closest_centroid_idx].append(point)
        return clusters

    def _calculate_new_centroids(self, clusters):
        """各クラスタの重心を再計算"""
        centroids = []
        for cluster in clusters:
            if cluster:
                mean_values = [
                    sum(point.values[i] for point in cluster) / len(cluster)
                    for i in range(len(cluster[0].values))
                ]
                centroids.append(Vector(mean_values))
            else:
                # クラスタが空の場合、埋め込みベクトルから新しい重心を選定
                centroids.append(
                    self.embedding_vectors[len(centroids) % len(self.embedding_vectors)]
                )
        return centroids

    def fit(self, data_points):
        """K-meansクラスタリングのメイン処理"""
        self.centroids = self._initialize_centroids(data_points)
        for _ in range(self.max_iterations):
            self.clusters = self._assign_clusters(data_points, self.centroids)
            new_centroids = self._calculate_new_centroids(self.clusters)
            if self.centroids == new_centroids:
                break
            self.centroids = new_centroids
        return self.clusters, self.centroids

    def add_data_point(self, new_point):
        """新しいデータポイントを追加し、クラスタを更新"""
        distances = [
            new_point.distance_squared(centroid) for centroid in self.centroids
        ]
        closest_centroid_idx = distances.index(min(distances))
        self.clusters[closest_centroid_idx].append(new_point)
        # 重心を再計算
        self.centroids = self._calculate_new_centroids(self.clusters)


# LSH (Locality-Sensitive Hashing) の実装
class LSH:
    def __init__(self, dimensions, num_planes, embedding_vectors):
        # 埋め込みベクトルからハイパープレーンを選択
        self.planes = [embedding_vectors[i] for i in range(num_planes)]

    def hash(self, vector):
        """各ハイパープレーンに対してベクトルがどちら側にあるかを基にハッシュを生成"""
        return "".join(
            ["1" if vector.dot(plane) >= 0 else "0" for plane in self.planes]
        )


# 近似的な最近傍検索
def find_nearest_by_lsh_in_cluster(query_vector, cluster, lsh):
    """指定したクラスタ内でLSHを使って近似最近傍を探索"""
    hashed_data_points = [(vec, lsh.hash(vec)) for vec in cluster]
    query_hash = lsh.hash(query_vector)
    closest_vector = None
    closest_distance = float("inf")

    for vec, hash_value in hashed_data_points:
        if hash_value == query_hash:  # 同じハッシュバケット内を探索
            distance = query_vector.distance_squared(vec)
            if distance < closest_distance:
                closest_vector = vec
                closest_distance = distance

    return closest_vector, math.sqrt(closest_distance)


# Vectorクラスを活用してPCAを実装する
class PCA:
    def __init__(self, num_components):
        self.num_components = num_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # データを中心化
        self.mean = self._compute_mean(X)
        X_centered = [x.subtract(self.mean) for x in X]

        # 共分散行列を計算
        cov_matrix = self._compute_covariance_matrix(X_centered)

        # パワーイテレーションで主成分の固有ベクトルを求める
        self.components = []
        for _ in range(self.num_components):
            eigenvalue, eigenvector = self._power_iteration(cov_matrix, 100)
            self.components.append(eigenvector)
            # 共分散行列を更新して次の主成分を取得
            cov_matrix = self._deflate(cov_matrix, eigenvalue, eigenvector)

    def transform(self, X):
        # データを中心化
        X_centered = [x.subtract(self.mean) for x in X]
        # 主成分に基づいてデータを変換
        return [self._project(x, self.components) for x in X_centered]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _compute_mean(self, X):
        # 各次元の平均を計算
        n = len(X)
        dim = len(X[0].values)
        mean_values = [sum(x.values[i] for x in X) / n for i in range(dim)]
        return Vector(mean_values)

    def _compute_covariance_matrix(self, X):
        # データの共分散行列を計算
        n = len(X)
        dim = len(X[0].values)
        cov_matrix = [[0] * dim for _ in range(dim)]

        for x in X:
            for i in range(dim):
                for j in range(dim):
                    cov_matrix[i][j] += x.values[i] * x.values[j]

        # 平均を取って共分散を計算
        return [[cov_matrix[i][j] / (n - 1) for j in range(dim)] for i in range(dim)]

    def _power_iteration(self, matrix, num_simulations):
        # パワーイテレーション法による最大固有値と固有ベクトルの計算
        b_k = Vector([1.0 for _ in range(len(matrix))])

        for _ in range(num_simulations):
            # 行列との積を計算
            b_k1_values = [
                sum(matrix[i][j] * b_k.values[j] for j in range(len(matrix)))
                for i in range(len(matrix))
            ]
            b_k1 = Vector(b_k1_values)

            # 正規化
            norm = math.sqrt(b_k1.norm_squared())
            b_k = Vector([x / norm for x in b_k1.values])

        # 固有値を計算
        eigenvalue = sum(
            b_k.values[i]
            * sum(matrix[i][j] * b_k.values[j] for j in range(len(matrix)))
            for i in range(len(matrix))
        )
        return eigenvalue, b_k

    def _deflate(self, matrix, eigenvalue, eigenvector):
        # 共分散行列からすでに得られた固有ベクトルの影響を削除
        dim = len(matrix)
        for i in range(dim):
            for j in range(dim):
                matrix[i][j] -= (
                    eigenvalue * eigenvector.values[i] * eigenvector.values[j]
                )
        return matrix

    def _project(self, vector, components):
        # 主成分にベクトルを投影
        return [vector.dot(component) for component in components]


# クラスタを基にしたLSH検索
def search_in_nearest_cluster(query_vector, clusters, centroids, lsh):
    """クエリベクトルに対して最も近いクラスタ内でLSH検索を行う"""
    closest_cluster_idx = min(
        range(len(centroids)), key=lambda i: query_vector.distance_squared(centroids[i])
    )
    closest_cluster = clusters[closest_cluster_idx]
    return find_nearest_by_lsh_in_cluster(query_vector, closest_cluster, lsh)


# 処理時間を計測する関数
def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


# サンプルデータを生成
def generate_data(num_points, dimensions=2, value_range=(0, 100)):
    return [
        Vector([random.randint(*value_range) for _ in range(dimensions)])
        for _ in range(num_points)
    ]


# メイン処理
if __name__ == "__main__":
    # データ生成
    num_data_points = 20000
    dimensions = 100
    data_points = generate_data(num_data_points, dimensions)

    # 埋め込みベクトルを生成（この埋め込みベクトルを利用）
    embedding_vectors = generate_data(100, dimensions)  # 100個の埋め込みベクトル

    # K-meansクラスタリングを実行してクラスタを作成
    kmeans = KMeans(k=5, embedding_vectors=embedding_vectors)
    clusters_and_centroids, kmeans_time = time_function(kmeans.fit, data_points)
    clusters, centroids = clusters_and_centroids

    # LSHを初期化
    num_planes = 5  # LSHで使用するハイパープレーンの数
    lsh = LSH(dimensions, num_planes, embedding_vectors)

    # クエリポイントの準備
    query_point = generate_data(1, dimensions)[0]

    # クラスタ内でのLSHによる検索と距離の算出
    (nearest_neighbor, distance), lsh_search_time = time_function(
        search_in_nearest_cluster, query_point, clusters, centroids, lsh
    )

    # 結果を表示
    print(f"Query point: {query_point.values}")
    print(f"Distance to Nearest Neighbor: {distance:.6f}")
    print(f"LSH Search Time: {lsh_search_time:.6f} seconds")
    print(f"K-means Clustering Time: {kmeans_time:.6f} seconds")
