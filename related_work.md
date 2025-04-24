## Graph Learning-based Recommendation System:**

### 1. Kết hợp Đồ thị Tương tác Người dùng-Sản phẩm vào Lọc Cộng tác (Collaborative Filtering)
Ý tưởng: Sử dụng đồ thị tương tác người dùng-sản phẩm (user-item interaction graph) để cải thiện việc học biểu diễn trong khung lọc cộng tác.
Ví dụ:
- LightGCN [15]: Một mô hình áp dụng Mạng Tích chập Đồ thị (Graph Convolutional Network - GCN) lên đồ thị tương tác người dùng-sản phẩm.
Cách hoạt động: LightGCN lan truyền thông tin qua đồ thị để làm phong phú biểu diễn của người dùng và sản phẩm, từ đó dự đoán sở thích của người dùng chính xác hơn.
Ý nghĩa: Cách tiếp cận này đơn giản nhưng hiệu quả, vì nó tận dụng cấu trúc đồ thị để ghi nhận các mẫu tương tác phức tạp.
### 2. Gợi ý Theo Chuỗi (Sequential Recommendation)
Ý tưởng: Sắp xếp các chuỗi sản phẩm mà người dùng đã tương tác (như lịch sử mua hàng hoặc xem) thành một đồ thị, sau đó phân tích các mẫu chuyển đổi (transition patterns) trong đồ thị để gợi ý.
Cách thực hiện:
Các sản phẩm được liên kết dựa trên tính liền kề (adjacency) trong chuỗi tương tác.
Mô hình học cách dự đoán sản phẩm tiếp theo dựa trên cấu trúc đồ thị của chuỗi.
Tham chiếu: Các nghiên cứu [21, 28] [` Streaming Session-Based Recommendation: When Graph Neural Networks meet the Neighborhood. `, ` Star graph neural networks for session-based recommendation`] đã phát triển các phương pháp gợi ý theo chuỗi dựa trên đồ thị.
Ứng dụng: Phù hợp với các kịch bản như gợi ý video tiếp theo trên YouTube hoặc sản phẩm tiếp theo trong giỏ hàng.
### 3. Gợi ý Xã hội (Social Recommendation)
Ý tưởng: Kết hợp thông tin từ mạng xã hội để cải thiện mô hình hóa người dùng, dựa trên giả định rằng những người có quan hệ xã hội (như bạn bè) sẽ có sở thích tương tự.
Hai cách tiếp cận:
Tách biệt hai đồ thị: Một số nghiên cứu [41] `Aneural influence diffusion model for social recommendation` sử dụng đồ thị tương tác người dùng-sản phẩm và đồ thị mạng xã hội riêng lẻ để học biểu diễn người dùng, sau đó kết hợp chúng.
Hợp nhất đồ thị: Các nghiên cứu khác [40] ` A neural influence and interest diffusion network for social recommendation.` tích hợp cả hai loại đồ thị (tương tác và xã hội) thành một đồ thị thống nhất để học biểu diễn người dùng cải tiến.
Ý nghĩa: Cách tiếp cận này khai thác thêm thông tin xã hội, giúp mô hình gợi ý chính xác hơn trong các nền tảng như mạng xã hội.
### 4. Gợi ý Dựa trên Đồ thị Tri thức (Knowledge Graph-based Recommendation)
Ý tưởng: Kết hợp đồ thị tri thức (knowledge graph) vào hệ thống gợi ý để bổ sung thông tin phụ (side information), như đặc điểm sản phẩm (ví dụ: thể loại phim, thương hiệu sản phẩm).
Cách thực hiện: Các nghiên cứu [36, 44, 51] [ ` Learning intents behind interactions with knowledge graph for recommendation.`, `Knowledge graph contrastive learning for recommendation`, `Multi-level cross-view contrastive learning for knowledge aware recommender system. ` ] xây dựng mô hình gợi ý bằng cách tích hợp thông tin từ đồ thị tri thức vào quá trình học biểu diễn.
Ví dụ: Trong gợi ý phim, đồ thị tri thức có thể chứa thông tin về đạo diễn, diễn viên, hoặc thể loại, giúp mô hình hiểu sâu hơn về sản phẩm.
Ưu điểm: Tăng cường khả năng gợi ý bằng cách kết hợp thông tin ngữ nghĩa phong phú.
Hạn chế của Các Phương pháp Hiện tại
Vấn đề bảo mật: Các phương pháp hiện tại thường thu thập dữ liệu người dùng tập trung (centralized data collection), điều này vi phạm quyền riêng tư của người dùng vì dữ liệu nhạy cảm có thể bị lộ.

### 5. Kết hợp Graph Attention Network vào Gợi ý Xã hội
Ý tưởng: Kết hợp Học Liên Kết Phân Tán (Federated Learning) và Mạng Nơ-ron Đồ thị Chú ý (Graph Attention Networks - GATs) để cải thiện các khuyến nghị xã hội trong khi bảo vệ quyền riêng tư của người dùng. Phương pháp này tận dụng GATs để học biểu diễn từ các đồ thị xã hội và tương tác người dùng-sản phẩm, đồng thời sử dụng Federated Learning để huấn luyện mô hình mà không cần tập trung hóa dữ liệu nhạy cảm.
Ví dụ: *Enhancing Federated Learning-Based Social Recommendations with Graph Attention Networks* (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4978530)

Cách thực hiện: GATs học biểu diễn từ đồ thị xã hội tại client cục bộ, sau đó mới cập nhật lên server để tổng hợp, đảm bảo quyền riêng tư.
Ưu điểm: Bảo vệ quyền riêng tư bằng cách không chia sẻ dữ liệu thô. Nâng cao chất lượng gợi ý nhờ khai thác thông tin xã hội và tương tác qua GATs. 


## Federated Learning:
- Federated Learning (FL) là một phương pháp học máy phân tán, trong đó nhiều thiết bị hoặc client hợp tác huấn luyện một mô hình mà không chia sẻ dữ liệu thô. Thay vào đó, các client chia sẻ các cập nhật mô hình (như trọng số và độ lệch) với một máy chủ trung tâm, nơi các cập nhật này được tổng hợp để cải thiện mô hình toàn cầu. Phương pháp này đặc biệt quan trọng trong hệ thống FedSystem, nơi quyền riêng tư người dùng cần được bảo vệ, vì dữ liệu nhạy cảm như lịch sử tương tác hoặc sở thích vẫn được giữ trên thiết bị cục bộ.
- Lợi ích:
Bảo vệ quyền riêng tư: Bằng cách không chia sẻ dữ liệu thô, FL giảm nguy cơ vi phạm dữ liệu
Khả năng mở rộng: Cho phép huấn luyện trên các tập dữ liệu lớn và phân tán mà không cần lưu trữ tập trung.
Giảm chi phí giao tiếp: Chỉ truyền tải các cập nhật mô hình, giúp tiết kiệm băng thông, đặc biệt hữu ích cho các thiết bị có tài nguyên hạn chế.
Xử lý dữ liệu không đồng nhất: FL được thiết kế để xử lý dữ liệu không đồng nhất (non-IID), phù hợp với các kịch bản thực tế nơi dữ liệu của các client khác nhau.


## Federated Recommendation System:
- Các phương pháp và hệ thống hiện có:
    + Dựa trên ma trận tương tác: FCF, FedMF, MetaMF, FedRecon, và FedNCF tập trung vào ma trận tương tác để dự đoán sở thích người dùng.
    + Sử dụng thông tin phong phú: FedFast và Efficient-FedRec sử dụng nhiều nguồn dữ liệu như đặc trưng người dùng và thuộc tính tin tức, giảm chi phí tính toán.
    + Tiến bộ gần đây: Tích hợp FMs như ChatGPT với các hệ thống như Federated Adaptation và Fellas, cải thiện hiệu suất và cá nhân hóa.

#### Hệ thống Dựa trên Ma trận Tương tác
+ FCF [Ammad-Ud-Din, 2019]: Là phương pháp lọc hợp tác dựa trên FL đầu tiên, sử dụng gradient ngẫu nhiên để cập nhật mô hình cục bộ và FedAvg để cập nhật mô hình toàn cầu.
+ FedMF [Chai, 2020]: Thích nghi phân tích ma trận phân phối với FL, giới thiệu mã hóa đồng hình trên gradient để bảo vệ quyền riêng tư trước khi tải lên máy chủ.
+ MetaMF [Lin, 2020b]: Khung phân tích ma trận phân phối với mạng meta để tạo mô hình dự đoán đánh giá và nhúng mục tư nhân, nhưng máy chủ có thể rò rỉ thông tin tương tác người dùng.
FedRecon [Singhal, 2021]: Phương pháp dựa trên meta-learning, bảo tồn mô hình cục bộ cho mỗi client và huấn luyện mô hình toàn cầu với FedAvg.
+ FedNCF [Perifanis và Efraimidis, 2022]: Thích nghi Lọc hợp tác thần kinh (NCF) với FL, sử dụng mạng thần kinh để học hàm tương tác người dùng-mục, nâng cao khả năng học của mô hình.
+ PFedRec [Zhang, 2023]: Khung FedRec cá nhân hóa, loại bỏ nhúng người dùng và học hàm điểm số cá nhân hóa để nắm bắt sở thích người dùng, nhưng bỏ qua mối quan hệ giữa các người dùng.

#### Hệ thống Sử dụng Thông tin Phong phú
+ FedFast [Muhammad, 2020]: Mở rộng FedAvg với phương pháp tổng hợp chủ động để tăng tốc độ hội tụ, sử dụng nhiều nguồn dữ liệu như đặc trưng người dùng và thuộc tính tin tức. (https://www.researchgate.net/publication/343782627_FedFast_Going_Beyond_Average_for_Faster_Training_of_Federated_Recommender_Systems)
+ Efficient-FedRec [Yi, 2021]: Phân tách mô hình thành mô hình tin tức trên máy chủ và mô hình người dùng trên client, giảm chi phí tính toán và giao tiếp, dựa vào dữ liệu phong phú. (https://arxiv.org/abs/2109.05446)

#### Kết hợp Mô hình Nền tảng (Foundation Models - FMs)
+ Federated recommendation via hybrid retrieval augmented generation [Zeng, 2024]: Đề xuất cách tiếp cận lai kết hợp truy xuất và sinh thành, sử dụng FMs. (https://arxiv.org/abs/2403.04256)

+ Federated Adaptation for Foundation Model-based Recommendations [Zhang, 2024]: Giới thiệu cơ chế thích nghi liên bang, nơi mỗi client học adapter cá nhân hóa nhẹ, hợp tác với FMs để cung cấp khuyến nghị hiệu quả và chi tiết. (https://arxiv.org/abs/2405.04840)


