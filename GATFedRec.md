# Dataset
- https://www.kaggle.com/datasets/aravindaraman/ciao-data
# 1. Định nghĩa
- U = {u₁, u₂, …, u_N}: tập N người dùng.
- I = {i₁, i₂, …, i_M}: tập M mục (items).
- R ∈ ℝ^(M×N): ma trận user–item interaction, với R_{u,i}=1 nếu user u đã click/mua item i.
- S ∈ ℝ^(N×N): ma trận mạng xã hội, với S_{u,v}=1 nếu có kết nối (friend/trust) giữa u và v. (Trustnetwork)
- T ∈ ℝ^(M×M): ma trận quan hệ item–item, với T_{p,q}=1 nếu item p và q được xác định là “liên quan” (tương đồng) dựa trên clustering và cosine similarity. (dùng GNN clustering)

- Sau đó Local GAT sẽ điều chỉnh mức độ ảnh hưởng của các neighbor lên từng nút đỉnh.

- Self-attention sẽ tổng hợp từng đỉnh với các neighbor xung quanh.

# 2. Step by step
## Client
- Tham số của các mô hình: 
    + $e_u$: embedding user
    + $e_i$: embedding item
    + $W_1$: ma trận trọng số GAT cho item-item
    + $W_2$: ma trận trọng số GAT cho user-user
    + $W_3$: ma trận trọng số GAT cho user-item
    + $W_g$: ma trận trọng số tổng hợp neighbor của item (neighbor là items) để self-attention + update lại embedding item
    + $W_h$: ma trận trọng số tổng hợp neighbor của user (neighbor là users & items) để self attention + update lại embedding user
    + $a,b,c,d,e$: lần lượt là các tham số cho GAT item-item, GAT user-user, GAT user-item, self-attention item, self-attention user 
    + $u_i​,u_s​,v_u​,v_i​,v_s$: vectơ quan hệ (table II ghi), mục đích giống $a,b,c,d,e$ ở trên

- Sau bước self-attention để update user embedding & item embedding, ta sẽ có $e_u*$ & $e_i*$, sẽ có 1 hàm Loss $L$ để tính.

- Với bộ tham số $Θ$ = { $E_u​,E_i​,W_1​,W_2​,W_3​,W_g​,W_h​,a,b,c,d,f,u_i​,u_s​,v_u​,v_i​,v_s$ ​}, tính đạo hàm L trên từng tham số { $ \frac{L}{θ} | θ ∈ Θ $ }

- Mọi gradient sau khi đạo hàm sẽ được flatten sau đó concat lại thành 1 vectơ duy nhất $g^{(n)}$ để gửi lên server

## Server
- Aggegrate các vecto sau đó gửi về lại từng client tương ứng 
