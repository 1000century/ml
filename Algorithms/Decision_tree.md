python```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# =============================================================================
# 도움 함수: truncated normal 샘플 생성
def get_truncated_normal(mean, sd, low, upp, size):
  a, b = (low - mean) / sd, (upp - mean) / sd
  return truncnorm(a, b, loc=mean, scale=sd).rvs(size)

# =============================================================================
# 설정 값
np.random.seed(42)  # 재현성을 위한 시드 설정
n_samples = 1000  # 전체 데이터 샘플 수

# 전체 인구의 80%는 정상군, 20%는 환자군으로 가정
n_healthy = int(n_samples * 0.8)
n_patients = n_samples - n_healthy

# 나이 생성 (20~80세 사이 랜덤 생성; 전체 샘플에 대해)
ages = np.random.randint(20, 81, size=n_samples)
# 각 집단에 대해 나이를 나눠줍니다.
ages_healthy = ages[:n_healthy]
ages_patients = ages[n_healthy:]

# =============================================================================
# 1. PaCO2 생성
# 정상군: PaCO2 평균 40, sd 2, 범위 [35, 45]
PaCO2_healthy = get_truncated_normal(mean=40, sd=2, low=35, upp=45, size=n_healthy)
# 환자군: PaCO2 평균 50, sd 5, 범위 [45, 60]
PaCO2_patients = get_truncated_normal(mean=50, sd=5, low=45, upp=60, size=n_patients)

# =============================================================================
# 2. 폐포 산소압 (PAO2) 계산: PAO2 = 150 - 1.25 * PaCO2
PAO2_healthy = 150 - 1.25 * PaCO2_healthy
PAO2_patients = 150 - 1.25 * PaCO2_patients

# =============================================================================
# 3. AaDO2 (A-a gradient) 생성
# 정상군: AaDO2 평균 5 mmHg, sd 2, 범위 [2, 12]
AaDO2_healthy = get_truncated_normal(mean=5, sd=2, low=2, upp=12, size=n_healthy)
# 환자군: AaDO2 평균 20 mmHg, sd 5, 범위 [10, 40]
AaDO2_patients = get_truncated_normal(mean=20, sd=5, low=10, upp=40, size=n_patients)

# =============================================================================
# 4. PaO2 계산: PaO2 = PAO2 - AaDO2 
PaO2_healthy = PAO2_healthy - AaDO2_healthy
PaO2_patients = PAO2_patients - AaDO2_patients

# =============================================================================
# 5. 전체 데이터 결합
PaCO2_values = np.concatenate([PaCO2_healthy, PaCO2_patients])
PaO2_values = np.concatenate([PaO2_healthy, PaO2_patients])
AaDO2_values = np.concatenate([AaDO2_healthy, AaDO2_patients])
ages_all = np.concatenate([ages_healthy, ages_patients])

# 정상 A-a 기준 (나이별): (나이 / 4) + 4
normal_AaDO2_thresholds = (ages_all / 4) + 4

# 정상/비정상 라벨링 (여기서는 AaDO2 값이 나이별 정상 기준을 초과하면 '비정상'으로 라벨링)
labels = np.where(AaDO2_values > normal_AaDO2_thresholds, '1', '0')
```
```python
# 데이터프레임 생성
data = {
  'Age': ages,
  'PaCO2': PaCO2_values,
  'PaO2': PaO2_values,
    'Label': labels
}
df = pd.DataFrame(data)

df

from sklearn.model_selection import train_test_split

# 특성과 라벨 분리
X = df[['Age', 'PaCO2', 'PaO2']]
y = df['Label']

# 훈련/테스트 데이터셋 분리
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
```

python```
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

# 1. 결정 트리 학습
tree_model = DecisionTreeClassifier(random_state=0)
tree_model.fit(X_train, y_train)
print("Train score:", tree_model.score(X_train, y_train))
print("Test  score:", tree_model.score(X_test, y_test))
tree_struct = tree_model.tree_

# ---------------------------
# 2. 부분 트리 리프 노드 영역 추출 함수
# ---------------------------
def get_partial_tree_regions(node, bounds, tree, X, max_depth, depth=0):
    if node == -1:
        return []
    predicted_class = np.argmax(tree.value[node])
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    in_region = np.logical_and.reduce((
        X[:,0]>=xmin, X[:,0]<=xmax,
        X[:,1]>=ymin, X[:,1]<=ymax,
        X[:,2]>=zmin, X[:,2]<=zmax
    ))
    count = int(np.sum(in_region))
    is_leaf = (tree.children_left[node] == -1 and tree.children_right[node] == -1)
    if is_leaf or depth == max_depth:
        return [(bounds, predicted_class, depth, count)]
    
    f = tree.feature[node]
    thr = tree.threshold[node]
    if f == 0:
        left_bounds  = (xmin, thr, ymin, ymax, zmin, zmax)
        right_bounds = (thr, xmax, ymin, ymax, zmin, zmax)
    elif f == 1:
        left_bounds  = (xmin, xmax, ymin, thr, zmin, zmax)
        right_bounds = (xmin, xmax, thr, ymax, zmin, zmax)
    else:  # f == 2
        left_bounds  = (xmin, xmax, ymin, ymax, zmin, thr)
        right_bounds = (xmin, xmax, ymin, ymax, thr, zmax)
    
    left_list = get_partial_tree_regions(tree.children_left[node], left_bounds, tree, X, max_depth, depth+1)
    right_list= get_partial_tree_regions(tree.children_right[node], right_bounds,tree, X, max_depth, depth+1)
    return left_list + right_list

# ---------------------------
# 3. 3D 직육면체 / 영역 중앙 텍스트
# ---------------------------
def create_cuboid(bounds, color="lightblue", opacity=0.4, name=""):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    vertices = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ])
    I = [0,0,4,4,0,0,3,3,0,0,1,1]
    J = [1,2,5,6,1,5,2,6,3,7,2,6]
    K = [2,3,6,7,5,4,6,7,7,4,6,5]
    
    mesh = go.Mesh3d(
        x=vertices[:,0],
        y=vertices[:,1],
        z=vertices[:,2],
        i=I, j=J, k=K,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=True,   # Legend에 표시
        flatshading=True,
        showscale=False
    )
    return mesh

def create_annotation(bounds, count):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = 0.5*(xmin+xmax)
    cy = 0.5*(ymin+ymax)
    cz = 0.5*(zmin+zmax)
    return go.Scatter3d(
        x=[cx], y=[cy], z=[cz],
        mode='text',
        text=[f"Count: {count}"],
        showlegend=False
    )

# ---------------------------
# 4. (선택) 부분 트리 요약 문자열
# ---------------------------
def summarize_partial_tree(node, bounds, tree, X, max_depth, depth=0, indent=""):
    if node == -1:
        return ""
    predicted_class = np.argmax(tree.value[node])
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    in_region = np.logical_and.reduce((
        X[:, 0]>=xmin, X[:, 0]<=xmax,
        X[:, 1]>=ymin, X[:, 1]<=ymax,
        X[:, 2]>=zmin, X[:, 2]<=zmax
    ))
    count = int(np.sum(in_region))
    is_leaf = (tree.children_left[node] == -1 and tree.children_right[node] == -1)
    summary_line = f"{indent}Node {node}: depth={depth}, class={predicted_class}, count={count}\n"
    if is_leaf or depth == max_depth:
        return summary_line
    f = tree.feature[node]
    thr= tree.threshold[node]
    if f == 0:
        lb = (xmin, thr, ymin, ymax, zmin, zmax)
        rb = (thr, xmax, ymin, ymax, zmin, zmax)
    elif f == 1:
        lb = (xmin, xmax, ymin, thr, zmin, zmax)
        rb = (xmin, xmax, thr, ymax, zmin, zmax)
    else:
        lb = (xmin, xmax, ymin, ymax, zmin, thr)
        rb = (xmin, xmax, ymin, ymax, thr, zmax)
    if depth < max_depth:
        s_left = summarize_partial_tree(tree.children_left[node], lb, tree, X, max_depth, depth+1, indent+"    ")
        s_right= summarize_partial_tree(tree.children_right[node], rb, tree, X, max_depth, depth+1, indent+"    ")
        return summary_line + s_left + s_right
    else:
        return summary_line

# ---------------------------
# 5. Figure 생성
# ---------------------------
fig = go.Figure()

# (a) 훈련데이터 산점도 (항상 표시, Legend에도 표시)
scatter_trace = go.Scatter3d(
    x=X_train[:,0],
    y=X_train[:,1],
    z=X_train[:,2],
    mode='markers',
    marker=dict(
        size=5,
        color=y_train.astype(float),  # 숫자형으로 변환
        colorscale='Portland',
        opacity=0.7
    ),
    name="Training Data",
    showlegend=True
)
fig.add_trace(scatter_trace)

# 전체 데이터 범위
xmin, xmax = X_train[:,0].min(), X_train[:,0].max()
ymin, ymax = X_train[:,1].min(), X_train[:,1].max()
zmin, zmax = X_train[:,2].min(), X_train[:,2].max()
overall_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

# 트리 최대 depth 찾기
node_depths = []
def get_node_depths(node, depth=0):
    if node==-1:
        return
    node_depths.append(depth)
    l = tree_model.tree_.children_left[node]
    r = tree_model.tree_.children_right[node]
    if l!=-1: get_node_depths(l, depth+1)
    if r!=-1: get_node_depths(r, depth+1)

get_node_depths(0,0)
max_depth_val = max(node_depths)

# (e) 깊이별 부분 트리 리프들 -> (cuboid, annotation)
class_colors = {0:'lightblue', 1:'salmon'}
partial_tree_traces = []
for d in range(max_depth_val+1):
    regs = get_partial_tree_regions(0, overall_bounds, tree_struct, X_train, d)
    cuboids=[]
    annots=[]
    for idx,(bds, pred_cls, depth, ccount) in enumerate(regs):
        cmesh = create_cuboid(
            bds,
            color=class_colors.get(pred_cls,'gray'),
            opacity=0.4,
            name=f"Depth={depth}, Class={pred_cls}, Region#{idx}"
        )
        ann = create_annotation(bds, ccount)
        cuboids.append(cmesh)
        annots.append(ann)
    partial_tree_traces.append((cuboids,annots))

# (f) 각 depth 구간 Trace들 figure에 추가
for d in range(max_depth_val+1):
    cbs, ans = partial_tree_traces[d]
    for c in cbs:
        fig.add_trace(c)
    for a in ans:
        fig.add_trace(a)

# 몇 개 trace인지 계산
trace_count_scatter=1
trace_count_per_depth=[]
for d in range(max_depth_val+1):
    c_count=len(partial_tree_traces[d][0])
    a_count=len(partial_tree_traces[d][1])
    trace_count_per_depth.append(c_count+a_count)

# (g) 슬라이더 steps
steps=[]
offset=trace_count_scatter
total_tr = trace_count_scatter + sum(trace_count_per_depth)

for d in range(max_depth_val+1):
    n_this = trace_count_per_depth[d]
    vis = [False]*total_tr
    # 훈련데이터 보이기
    vis[0]=True
    # d에 해당하는 cuboid+annotation 보이기
    for i in range(n_this):
        vis[offset+i] = True
    
    summary_str = summarize_partial_tree(0, overall_bounds, tree_struct, X_train, max_depth=d)
    summary_str = summary_str.replace("\n","<br>")
    if not summary_str.strip():
        summary_str = f"No nodes at depth={d}"
    
    step=dict(
        method="update",
        args=[
            {"visible":vis},
            {
                "title":f"3D Decision Regions (Depth ≤ {d})",
                "annotations":[
                    dict(
                        x=1.3,  # 범례보다 더 오른쪽
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        text=summary_str,
                        showarrow=False,
                        align="left",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=5,
                        bgcolor="white",
                        font=dict(size=12, family="monospace")
                    )
                ]
            }
        ],
        label=f"Depth {d}"
    )
    steps.append(step)
    offset+=n_this

sliders=[dict(
    active=0,
    currentvalue={"prefix":"Max Depth: "},
    pad={"t":40},
    steps=steps
)]

# 레이아웃 수정: 범례 오른쪽에 배치, annotation도 오른쪽에 표시
fig.update_layout(
    title="3D Decision Tree Regions (Partial Tree by Depth)",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    sliders=sliders,
    legend=dict(
        x=1.0,   # 범례를 그래프 오른쪽 경계 근처에
        y=1.0,
        xanchor="left",
        yanchor="top"
    ),
    width=1200,
    height=800
)

# 초기 상태(Depth=0)에서 cuboid/annotation 비활성
fig.update_traces(visible=False, selector=dict(type='mesh3d'))
fig.update_traces(visible=False, selector=dict(type='scatter3d', mode='text'))

# depth=0 요소만 활성
initial_d=0
offset=trace_count_scatter
for dd in range(initial_d):
    offset+=trace_count_per_depth[dd]
n_this=trace_count_per_depth[initial_d]
for i in range(n_this):
    fig.data[offset+i].visible=True
# 훈련데이터 표시
fig.data[0].visible=True

fig.show()


print("네모박스 짜증나면text=summary_str 이거 찾아서 summary_str 부분을 ''로바꾸면됨")
```
