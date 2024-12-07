## fed_train_single_batch
- Huấn luyện một batch dữ liệu

    ### Khởi tạo và xử lý dữ liệu
    - ```
        users, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
    - Tách dữ liệu người dùng (users), mục (items), và đánh giá thực tế (ratings)

    - ```
        reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data)
    - reg_item_embedding: `USER-SPECIFIC ITEM EMBEDDING`
    

    ### Lấy optimizer
    - ```
        optimizer, optimizer_u, optimizer_i = optimizers

    - optimizer: Cập nhật các tham số MLP (Multi-Layer Perceptron).
    - optimizer_u: Cập nhật embedding của người dùng.
    - optimizer_i: Cập nhật embedding của item.

    ### Reset optimizers
    - ```
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()
    - Xóa gradient cũ của tất cả các optimizer trước khi train.

    ### Dự đoán và tính toán lỗi
    - Sử dụng model dự đoán ratings:
    - ```
        ratings_pred = model_client(items)

    - Tính toán lỗi giữa giá trị dự đoán và giá trị thực tế:
    - ```
        loss = self.crit(ratings_pred.view(-1), ratings)

    ### Tính regularization
    - Hàm `compute_regularization` sử dụng MSE (Mean Squared Error) để so sánh chênh lệch giữa model_client (`global shared`) & reg_item_embedding (`user-specific`)

    - Cộng thêm regularization * hệ số chuẩn hóa λ `(self.config['reg'])` vào lỗi tổng (loss).
    - ```
        loss += self.config['reg'] * regularization_term

    ### Lan truyền ngược và cập nhật tham số
    - Tính toán gradient của hàm mất mát với respect tới các tham số mô hình.
    - ```
        loss.backward()
    
    - Cập nhật tham số dựa trên gradient đã tính:
    - ```
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()

## aggregate_clients_params



## fed_train_a_round
- Chịu trách nhiệm huấn luyện 1 round học liên kết.

    ### Lựa chọn người dùng tham gia
    - ```
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
    - `num_participants`: số lượng người dùng tham gia = tổng người dùng * tỉ lệ client mẫu
    - `participants`: danh sách ID ngẫu nhiên lấy mẫu từ num_users

    ### Khởi tạo tham số cho máy chủ vòng đầu tiên
    - ```
        if round_id == 0:
        self.server_model_param['embedding_item.weight'] = {}
        for user in participants:
            self.server_model_param['embedding_item.weight'][user] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())
        self.server_model_param['embedding_item.weight']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())
    - Nếu là vòng đầu tiên (round_id == 0)
        - `self.server_model_param['embedding_item.weight']`: Dictionary chứa tham số của item_embedding.
        - `self.model.state_dict()['embedding_item.weight'].data`: 
            + `state_dict()`: lấy các tham số của mô hình
            + `['embedding_item.weight']`: lấy attribute model Embedding `embedding_item` & `.weight` để lấy attribute của Embedding model.
        - `[user]`: Lặp qua tất cả `participants` để sao chép tham số của model toàn cục cho từng user.
        - `[global]`: tham số toàn cục 
    
    ### Huấn luyện từng người dùng tham gia
    - ```
        for user in participants:
        ```    
        #### Khởi tạo mô hình của người dùng ở vòng đầu
        - ```
            model_client = copy.deepcopy(self.model)
        - Sử dụng mô hình của máy chủ làm mô hình cục bộ cho từng người dùng.

        #### Tải tham số model ở các vòng sau
        - ```
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight']['global'].data).cuda()
                model_client.load_state_dict(user_param_dict)

        - Tạo 1 bản sao tham số mô hình tai client `user_param_dict` từ máy chủ.
        - Nếu id user có ở trong danh sách (client_model_params.keys()), tải các tham số đó
            + `user_param_dict[key]` ~ `['fc.weight', 'fc.bias', 'embedding_user.weight', ...]`
        - **Tham số `embedding_item.weight` được tải từ tham số global trên máy chủ. (GLOBAL SHARED ITEM EMBEDDING)**
        - Update mô hình cục bộ của người dùng (model_client) = `user_param_dict[key]` (vi chi code tren server nen update model o client moi can buoc nay)

        #### Khởi tạo bộ tối ưu 
        - optimizers = [optimizer, optimizer_u, optimizer_i] (MLP Scored Function - User Embedding - Item Embedding)

        #### Tải dữ liệu người dùng
        - ```
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
        
        - `user_dataloader`: Dữ liệu được chia thành các batch dựa trên `batch_size`

        #### Huấn luyện mô hình người dùng
        - ```
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss = self.fed_train_single_batch(model_client, batch, optimizers, user)
            
        - Lặp qua số lần huấn luyện cục bộ (local_epoch).
        - Trong mỗi epoch, huấn luyện trên từng batch dữ liệu bằng cách gọi hàm `fed_train_single_batch`.

        #### Lưu tham số mô hình của người dùng
        - ```
            client_param = model_client.state_dict()
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
        - `[key] = [key].data`
        - Lưu tham số mô hình của người dùng để sử dụng ở vòng tiếp theo. Các tham số này được chuyển về CPU.

        #### Thêm nhiễu để bảo vệ quyền riêng tư
        - ```
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(self.client_model_params[user]['embedding_item.weight'])
            round_participant_params[user]['embedding_item.weight'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_item.weight'].shape).sample()

        #### Tổng hợp tham số người dùng trên máy chủ
        - ```
            self.aggregate_clients_params(round_participant_params)