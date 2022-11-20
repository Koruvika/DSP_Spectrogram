## Problem 1
### Nội dung
- Input: Tín hiệu tiếng nói
- Output: Ảnh phổ, bộ 3 tần số formant (F1, F2, F3) của mỗi nguyên âm.
### Yêu cầu
- Wideband Spectrogram
- Formant Frequencies F1 F2 F3
### Lý thuyết
- Wideband Spectrogram: sử dụng thư viện scipy (cửa sổ ngắn hơn T0 (chu kỳ lấy mẫu))
- Formant Frequencies: Sử dụng WaveSurface để xác định Formant thủ công

## Problem 2
### Nội dung
### Yêu cầu
### Lý thuyết - Hướng giải quyết
- Phân đoạn nguyên âm - khoảng lặng
  - Phân đoạn tín hiệu tiếng nói - khoảng lặng
  - Chia 3 phần
  - Lấy phần ở giữa
- Trích xuất đặc trưng của vùng vừa tìm được
  - Tính FFT từng frame trong vùng vừa tìm được
  - Lấy trung bình các FFT để được kết quả của 1 tín hiệu tiếng nói
  - 

## Problem 3
### Nội dung
### Yêu cầu
### Lý thuyết - Hướng giải quyết