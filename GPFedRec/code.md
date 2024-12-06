## HR@k (Hit Ratio) - Đánh giá sự xuất hiện
- Đo `tỉ lệ phần trăm số người dùng có ít nhất một test_item nằm trong top-k`
- Ví dụ HR@10 = 0.72, có nghĩa là `có 72% user có ít nhất 1 mục mà họ thích trong top 10 mục được gọi ý`.

## NDCG (Normalized Discounted Cumulative Gain) - Đánh giá cả xuất hiện và thứ tự
- Giả sử người dùng nhận danh sách 10 bài hát với điểm yêu thích là [3, 2, 3, 0, 1, 2, 3, 1, 0, 1]

- $$ DCG@10 = \sum_{i=1}^{10} \frac{2^{rel_i} - 1}{log_2(i + 1)} $$
    + Trong đó $`rel_i`$ là điểm yêu thích của bài hát tại vị trí i

- IDCG là giá trị DCG của danh sách lý tưởng, với danh sách trên sẽ là [3, 3, 3, 2, 2, 1, 1, 1, 0, 0] (`vì mục tiêu là giúp người dùng có được các đề xuất đúng ý người dùng hàng đầu`) 

- $$ NDCG@10 = \frac{DCG@10}{IDCG@10} $$

- NDCG gần 1 chứng tỏ hệ thống xếp hạng đã đặt được các đề xuất đúng sở thích người dùng ở các vị trí đầu tiên.


# data.py
- UserItemRatingDataset: gồm `getItem` & `getLength`

- `SampleGenerator`: 
    + `user_pool` & `item_pool` chứa tất cả `userId` và `itemId` duy nhất.

    + `_normalize(self, ratings)`: chuẩn hóa rating về khoảng [0, 1]
    + `_binarize(self, ratings)`: Biến đổi rating thành nhị phân (0 hoặc 1)
    + `_split_loo()`: chia dữ liệu thành 3 tập `train, val, test`
        + `test`: lấy ratings mới nhất - mục tiêu kiểm tra hiệu quả mô hình với các mục mới nhất mà user đã tương tác
        + `val`: lấy ratings gần thứ 2 - gần giống với tập kiểm tra, dùng điều chỉnh mô hình và tối ưu các tham số mô hình
        + `train`: các ratings còn lại
        + Đảm bảo `num_user` bằng nhau trong cả 3 tập

    + `_sample_negative()`: Tạo các mẫu âm cho người dùng. Với từng `item` sẽ lấy `1 tương tác` kèm `99 mẫu âm` (`198` cho cả 2 tập `val` và `test`)

    + `validate_data()` & `validate_test`: Tạo dữ liệu cho 2 tập val & test
    
# engine.py
- Sử dụng hàm mất mát `Binary Cross-Entropy Loss` với thư viện `PyTorch` (`torch.BCELoss()`)
- Nếu `server_model_param` là từ điển chứa tất cả trọng số mô hình thì `server_model_param['embedding_item.weight'][user]` sẽ trả về `user-specific item embedding` cho từng user

<br/>

- `instance_user_train_loader()`: Chuyển các vecto `user`, `item`, `ratings` qua `Tensors` để thêm khả năng tính toán trên `GPU`.
- `fed_train_single_batch()`:
    + 

- `aggregate_clients_params()` - Xây dựng đồ thị quan hệ ở `server`:


- `fed_train_a_round()` - 1 round `client-server`:
    + `CLIENT`:
        + `model`: sử dụng `MLP`
            + embedding_user
            + embedding_item
            + fc_layers: các Hidden Layers, sử dụng Linear
                + Lớp Linear thực hiện phép biến đổi tuyến tính
                + $$ y = xW^T + b $$
                + x-input vector, W-ma trận trọng số, b-bias, y-output vector
            + affine_out: lớp đầu ra 
            + logistic: hàm kích hoạt, sử dụng Sigmoid (hàm biến đổi 1 giá trị thực z thành 1 giá trị trong khoảng (0, 1))
            
        + `self.model.state_dict()`: chứa các tham số & buffer của model 
            + 
        + `round_id == 0`:
            + Các client được tải các tham số mô mình khởi tạo default (model.state_dict())
        + Đoạn `if round_id != 0:`: tải các tham số mô hình từ lần train trước

# metrics.py
- `MetronAtK`: dùng để đánh giá mô hình dựa trên 2 độ đo là `Hit Ratio` và `NDCG`
- Tập `test` đưa vào chứa các dữ liệu thực và tập `full` chứa các rating dự đoán kèm rating kiểm tra.
- `cal_hit_ratio()`: tính toán xem các `test_item` có nằm trong `top-k` mục gợi ý (dựa trên `rank rating`) cho người dùng không. Nó đo `tỉ lệ phần trăm số người dùng có ít nhất một test_item nằm trong top-k`.    
- `cal_ndcg()`: tính bằng công thức
    + $$ ndcg = \frac{log(2)}{log(1 + rank)} = \frac{1}{log_2(rank + 1)} $$
    + `log(2)` giữ cho các giá trị nằm trong khoảng `0 đến 1`
    + `log(1 + rank)`: rank 1 có ndcg cao nhất và hạng sau đó thì ndcg giảm dần, thể hiện mức độ ưu tiên trong `top-k`

# mlp.py
- `Embedding là gì`:
    + Nhúng là 1 chuỗi số đóng vai trò là 1 mã định danh duy nhất
- `Tensor`: hỗ trợ phép toán ma trận & tận dụng GPU
- `torch.nn.Embedding(num_embeddings, embedding_dim)`:
    + `num_emmbeddings`: số lượng `*từ` trong từ điển
    + `embedding_dim`: số chiều của mỗi `vectơ nhúng`

# utils.py
- Khởi tạo số lượng user random
- 