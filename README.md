# Face Authentication Android

## Mô tả dự án

Face Authentication Android là một ứng dụng di động được phát triển để thực hiện điểm danh sinh viên tự động thông qua công nghệ nhận diện khuôn mặt. Dự án sử dụng các thuật toán học sâu và xử lý ảnh thời gian thực để xác định danh tính của sinh viên dựa trên hình ảnh hoặc video từ camera của thiết bị Android. Ứng dụng được xây dựng với mục tiêu thay thế phương pháp điểm danh truyền thống, mang lại sự tiện lợi, chính xác và hiệu quả trong môi trường giáo dục.

Dự án tích hợp các công nghệ như FaceNet cho nhận diện khuôn mặt, Google ML Kit cho phát hiện khuôn mặt, và TensorFlow Lite để tối ưu hóa mô hình trên thiết bị di động. Mã nguồn được viết bằng Kotlin và Java, hỗ trợ triển khai trên nhiều thiết bị Android khác nhau.

## Tính năng chính

- **Phát hiện khuôn mặt thời gian thực**: Sử dụng Google ML Kit để xác định vị trí khuôn mặt từ camera.
- **Nhận diện danh tính**: Áp dụng mô hình FaceNet đã được huấn luyện để so sánh và xác định danh tính sinh viên.
- **Ghi nhận điểm danh**: Lưu thông tin điểm danh vào cơ sở dữ liệu cục bộ hoặc đám mây.
- **Tối ưu hóa hiệu suất**: Sử dụng TensorFlow Lite để đảm bảo ứng dụng chạy mượt mà trên thiết bị có cấu hình thấp.

## Yêu cầu hệ thống

- **Phần cứng**: Thiết bị Android với camera, tối thiểu phiên bản Android 5.0 (API 21) trở lên.
- **Phần mềm**:
  - Android Studio (phiên bản mới nhất).
  - Java Development Kit (JDK) 11 hoặc cao hơn.
  - Gradle (được cài đặt tự động qua Android Studio).
- **Thư viện phụ thuộc**:
  - Google ML Kit SDK.
  - TensorFlow Lite.
  - OpenCV (tùy chọn, để xử lý ảnh nâng cao).

## Hướng dẫn cài đặt

### 1. Chuẩn bị môi trường
- Tải và cài đặt [Android Studio](https://developer.android.com/studio) trên máy tính.
- Cài đặt JDK 11 từ [trang chính thức của Oracle](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html) hoặc OpenJDK.
- Đảm bảo máy tính có kết nối internet để tải các gói phụ thuộc.

### 2. Clone repository
Mở terminal hoặc Git Bash và chạy lệnh sau để clone repository:
```bash
git clone https://github.com/[your-username]/face-authentication-android.git
cd face-authentication-android
