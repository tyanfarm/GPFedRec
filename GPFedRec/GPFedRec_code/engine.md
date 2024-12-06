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

        - Tạo 1 bản sao tham số mô hình `user_param_dict` từ máy chủ.
        - Nếu id user có ở trong danh sách (client_model_params.keys()), tải các tham số đó
            + `user_param_dict[key]` như ``