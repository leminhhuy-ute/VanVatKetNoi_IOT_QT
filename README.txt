SMART TRASH AI - QT DESIGNER DASHBOARD

Mở giao diện bằng Qt Designer:
- File > Open > smart_trash_ui.ui
- Hoặc chạy open_qt_designer.bat

Chạy chương trình:
- Chạy run.bat
- Hoặc chạy: python main.py

File chính:
- smart_trash_ui.ui: bố cục giao diện với thanh điều hướng ngang
- main.py: code chức năng camera, UART, dataset và thống kê
- assets/hcmute_logo.jpg: logo trường dùng ở thanh tiêu đề
- models/trash_classifier.onnx: model AI phân loại WasteWise YOLOv8n-cls
- models/model_classes.txt: 8 lớp gốc của model, mỗi dòng một lớp
- models/classes.txt: 3 nhóm phân loại dùng trong giao diện
- auto_captures/: ảnh được lưu tự động sau khi phân loại thành công
- config.json: cấu hình được tạo sau khi bấm Lưu cấu hình

Chức năng chính:
- Giám sát: trạng thái thiết bị, kết quả AI, camera có khung nhận diện và lịch sử phân loại
- Lịch sử: nhật ký phân loại và số rác trong ngày
- Quản Lý Hệ Thống: thu thập ảnh, tạo dataset, mở thư mục dataset và theo dõi sức chứa thùng rác
- Cài đặt: cấu hình UART, ngưỡng AI, epochs và nguồn camera

Ghi chú AI:
- Model gốc phân loại 8 lớp: battery, biological, cardboard, glass, metal, paper, plastic, trash.
- Ứng dụng tự ánh xạ về 3 nhóm: Rác hữu cơ, Rác vô cơ, Rác tái chế.
- biological -> Rác hữu cơ; cardboard/glass/metal/paper/plastic -> Rác tái chế; battery/trash -> Rác vô cơ.
- Nếu độ tin cậy thấp hơn ngưỡng trong Cài đặt, giao diện vẫn hiển thị nhóm dự đoán nhưng ghi chú "độ tin cậy thấp".
- Lịch sử phân loại chỉ ghi kết quả phân loại, không ghi trạng thái mở/tắt camera.
- Khi kết quả đủ tin cậy, ứng dụng tự chụp ảnh vùng vật thể và lưu vào auto_captures/.
- Ảnh tự động có khung đỏ và nhãn tiếng Việt, ví dụ "Chai nước | Rác tái chế".
- Nhãn vật thể được suy ra từ lớp gốc của model: plastic cao/dài -> Chai nước, paper -> Giấy, metal -> Lon kim loại...
- Mỗi lần tự chụp cách nhau tối thiểu 5 giây.
- Nếu vùng phát hiện giống người hoặc có khuôn mặt, ứng dụng bỏ qua và không tự chụp.
