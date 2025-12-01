# Bài toán Text-To-Speech (TTS)
## Mục tiêu
Báo cáo này trình bày tổng quan học thuật về bài toán Text-To-Speech (TTS): lịch sử và ba “level” phát triển (rule-based, deep-learning, few-shot/zero-shot voice cloning); kiến trúc điển hình (text encoder => acoustic model => vocoder); các kỹ thuật chính để tối ưu hoá (distillation, quantization, multilingual pretraining, prosody modeling); thách thức nghiên cứu (tốc độ, tài nguyên, đa ngôn ngữ, cảm xúc, đạo đức); và các phương pháp bảo mật như watermark cho giọng tổng hợp để giảm nguy cơ lạm dụng (deepfake).
## 1. Giới thiệu
Text-To-Speech (TTS) là công nghệ chuyển đổi văn bản thành giọng nói. Ứng dụng của TTS trải dài từ trợ lý ảo, đọc báo tự động, hỗ trợ người khiếm thị, tạo audiobook, đến các hệ thống tương tác thoại trong công nghiệp và giáo dục. Mục tiêu nghiên cứu bao gồm: (1) chất lượng âm thanh (naturalness), (2) tương đồng giọng (speaker similarity), (3) độ trôi chảy, (4) hiệu suất thời gian thực, và (5) an toàn/đạo đức khi sử dụng voice cloning.

Trong nhiều năm phát triển, TTS đã đi qua nhiều thế hệ: từ các hệ thống dựa trên luật (rule-based), đến mô hình học sâu (deep learning), và gần đây nhất là công nghệ few-shot/zero-shot cho phép "clone" giọng nói chỉ với vài giây mẫu âm thanh.

## 2\. Lịch sử phát triển ngắn gọn (Background)
- **Rule-based / concatenative / formant** (giai đoạn lịch sử): các hệ thống ghép mẫu (concatenative) hoặc mô tả đặc trưng âm học bằng luật (formant synthesis) — có thể chạy nhẹ nhưng giọng thiếu tự nhiên.

- **Statistical parametric TTS** (VD: HMM-based): dùng mô hình thống kê để dự đoán tham số âm thanh; mượt hơn nhưng vẫn kém tự nhiên so với giọng người.

- **Deep Learning TTS** (từ khoảng 2016–): sự xuất hiện của mô hình seq2seq và neural vocoder (Ví dụ Tacotron2 + WaveNet, FastSpeech + HiFi-GAN) đã nâng chất lượng lên mức rất gần người thật. Tacotron2 báo cáo điểm MOS rất cao trong thí nghiệm của họ. ([arXiv](https://arxiv.org/abs/1712.05884 "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"))

- **Large generative / few-shot TTS (2022–2024)**: các mô hình như VALL-E, Voicebox tiếp cận zero-shot/few-shot cloning, cho phép tổng hợp giọng của người lạ chỉ với vài giây mẫu. Những tiến bộ này mạnh mẽ nhưng đặt ra thách thức đạo đức và bảo mật. ([Microsoft](https://www.microsoft.com/en-us/research/project/vall-e-x/ "VALL-E"))
## 3. Tình hình nghiên cứu và các hướng tiếp cận chính

### 3.1. Level 1 - Rule-based TTS
Đây là giai đoạn đầu của TTS. Các hệ thống sử dụng tập luật âm vị hoặc quy tắc chuyển đổi kí tự => âm tiết => âm thanh. 

**Đặc điểm:**
- Chạy nhanh, nhẹ
- Không yêu cầu dữ liệu huấn luyện
- Dễ hỗ trợ đa ngôn ngữ vì chỉ cần định nghĩa tập luật

**Nhược điểm:**
- Giọng nghe máy móc, thiếu cảm xúc
- Không tự nhiên, nhất là với các ngôn ngữ khó

**Phù hợp với:** thiết bị nhúng, thông báo hệ thống, ứng dụng có giới hạn tài nguyên.
Ví dụ: eSpeak, Festival

### 3.2. Level 2 - Deep Learning TTS
Với sự phát triển của Deep Learning, các mô hình TTS trở nên tự nhiên và mượt mà hơn. Pipeline phổ biến:
`Text => (text preprocessing => phoneme) => Text Encoder => Acoustic Model => Mel-Spectrogram => Vocoder => Waveform`.

Trong đó:
- *Phoneme* (âm vị): đơn vị âm nhỏ nhất.

- *Mel-Spectrogram*: biểu diễn năng lượng tần số theo thang Mel (mô phỏng cảm nhận tai người).

- *Vocoder*: mô hình chuyển spectrogram thành sóng âm (waveform).

Một số kiến trúc tiêu biểu:
- Tacotron, Tacotron 2.  Tacotron 2 (seq2seq acoustic + WaveNet vocoder) — đạt MOS rất cao trong thử nghiệm gốc; FastSpeech / FastSpeech 2 (non-autoregressive) cải thiện tốc độ suy luận; HiFi-GAN và biến thể là vocoder nhanh và chất lượng cao. ([arXiv](https://arxiv.org/abs/1712.05884 "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"))
- FastSpeech, FastSpeech 2
- WireNet, HiFi-GAN

**Đặc điểm:**
- Tạo giọng nói tự nhiên gần giống người thật
- Có thể fine-tune cho từng người dùng => tạo giọng cá nhân hóa
- Các pipeline nhiều nghiên cứu sử dụng kiến trúc encoder => decoder => vocoder

**Nhược điểm:**
- Cần dữ liệu lớn (hàng giờ thu âm)
- Cần GPU huấn luyện
- Khó mở rộng đa ngôn ngữ nếu thiếu dữ liệu huấn luyện

**Phù hợp:** sản phẩm thương mại, audiobook, trợ lý ảo.

### 3.3. Level 3 - Few-shot/Zero-shot TTS (Voice Cloning)
Đây là hướng nghiên cứu hiện đại nhất: chỉ cần 3-10 giây mẫu âm thanh để tái tạo giọng nói của một người.

Dùng mô hình lớn (thường là transformer/language model hoặc diffusion) để học biểu diễn giọng đa dạng; chỉ cần vài giây (ví dụ 3–10s) audio mẫu để tổng hợp giọng mới. Ví dụ: VALL-E (Microsoft Research) và Voicebox (Meta/Google?) — các hệ thống này thể hiện khả năng zero-shot/one-shot nâng cao. ([Microsoft](https://www.microsoft.com/en-us/research/project/vall-e-x "VALL-E"))

VD: VALL-E, Voicebox, GPT-SoVITS

**Đặc điểm:**
- Tự nhiên cao
- Giữ được đặc trưng giọng nói (ngữ điệu, tốc độ, độ cao âm)
- Không cần huấn luyện riêng cho từng người, phù hợp cho chuyển giọng nhanh

**Nhược điểm:**
- Mô hình nặng và phức tạp
- Cần nhiều tài nguyên tính toán
- Nguy cơ bị lạm dụng tạo deepfake

**Phù hợp:** dubbing, đạo diễn âm thanh, ứng dụng cần clone giọng nhanh (nhưng phải cân nhắc đạo đức).

### 3.4. So sánh ưu - nhược điểm

| Hướng tiếp cận | Ưu điểm | Nhược điểm |
|----------------|---------|------------|
| **Level 1** | Rất nhanh, nhẹ, hỗ trợ nhiều ngôn ngữ | Giọng robot, thiếu tự nhiên |
| **Level 2** | Giọng tự nhiên, ổn định, có thể cá nhân hóa | Cần dữ liệu lớn, tốn tài nguyên |
| **Level 3** | Clone giọng chỉ cần vài giây; chất lượng cao | Rất nặng, khó triển khai, rủi ro deepfake |
### 3.5. Trường hợp sử dụng phù hợp

#### Level 1 – Rule-based
- Thiết bị IoT, robot nhỏ, vi điều khiển
- Hệ thống cần tốc độ cao, tài nguyên thấp

#### Level 2 – Deep Learning
- Ứng dụng thương mại: trợ lý ảo, audiobook
- Tổng hợp giọng nói đa cảm xúc
- Cá nhân hóa giọng cho từng người dùng

#### Level 3 – Voice Cloning
- Dubbing phim
- Game, avatar ảo
- Sản phẩm giải trí
- Ứng dụng thay thế giọng người bệnh

## 4\. Kiến trúc chi tiết và các thành phần chính
### 4.1 Tiền xử lý text và phonemization
- **Grapheme-to-Phoneme (G2P)**: chuyển chữ cái (grapheme) => phoneme; giảm phụ thuộc vào hệ chữ viết; rất quan trọng cho đa ngôn ngữ.

- Xử lý chữ viết tắt, dấu câu, số (normalization).

### 4.2 Text encoder / linguistic frontend
- Chuyển văn bản/phoneme sang embedding (vector). Các encoder hiện đại dùng transformer hoặc convolution + self-attention.

### 4.3 Acoustic model (sinh spectrogram)
- **Autoregressive models** (ví dụ Tacotron): sinh từng frame theo chuỗi; chất lượng cao nhưng chậm. ([arXiv](https://arxiv.org/abs/1712.05884 "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"))

- **Non-autoregressive models** (FastSpeech, FastSpeech2): dự đoán toàn bộ sequence song song, cần duration predictor; rất nhanh và ổn định. FastSpeech2 bổ sung điều kiện pitch/energy để giải quyết one-to-many mapping. ([arXiv](https://arxiv.org/abs/2006.04558 "[2006.04558] https://arxiv.org/abs/2006.04558"))

### 4.4 Vocoder (spectrogram => waveform)
- WaveNet (chất lượng cao nhưng chậm), Parallel WaveNet, WaveGlow, rồi tới GAN-based vocoders như HiFi-GAN (cao tốc, chất lượng tốt). HiFi-GAN/biến thể được dùng rộng rãi vì cân bằng giữa chất lượng và tốc độ. ([gfx.cs.princeton.edu](https://gfx.cs.princeton.edu/gfx/pubs/Su_2021_HSS/Su-HiFi-GAN-2-WASPAA-2021.pdf "hifi-gan-2: studio-quality speech enhancement via generative"))


## 5. Các kỹ thuật tối ưu hoá và pipeline thực nghiệm
Các nghiên cứu gần đây tập trung vào:
### 5.1. Tối ưu kiến trúc
- Kết hợp mô hình text encoder => acoustic model => vocoder
- Dùng vocoder thế hệ mới (HiFi-GAN, WaveGlow) để tái tạo âm thanh chân thực

### 5.2 Giảm chi phí tính toán
- **Knowledge distillation**: huấn luyện model nhỏ học theo model lớn (teacher → student) để giảm tham số.

- **Quantization**: giảm độ chính xác số học (float32 → int8) để tăng tốc inference.

- **Pruning**: cắt tham số ít đóng góp.

### 5.3 Cải thiện đa ngôn ngữ / multi-speaker
- **Multilingual pretraining**: huấn luyện trên nhiều ngôn ngữ để model học tính phổ quát (shared representations).

- **Speaker embedding**: học vector biểu diễn speaker, cho phép model tổng hợp nhiều giọng.

- **Phoneme-based modeling**: dùng phoneme làm đơn vị chung để giảm phụ thuộc script.

### 5.4 Tăng tính tự nhiên và cảm xúc
- **Prosody modeling**:  Tạo prosody modeling (ngữ điệu), dự đoán duration, pitch, energy; explicit conditioning giúp model tạo ngữ điệu phong phú.

- **Style tokens / Global Style Tokens (GST)**: cho phép điều chỉnh phong cách và cảm xúc bằng latent vector.

- **Diffusion models**: gần đây có xu hướng dùng diffusion để điều khiển tốt hơn đặc tính âm thanh.

### 5.5 Tối ưu few-shot
- Speaker embedding + diffusion-based vocoder
- Tối ưu giữ đặc trưng giọng trong mọi câu nói
- Sử dụng discrete codec + conditional language modeling (VALL-E approach) hoặc autoregressive encoder-decoder với speaker encoder để generalize từ vài giây mẫu. Những mô hình này tận dụng biểu diễn ngôn ngữ và audio mã hoá để in-context hoặc conditional generation. ([arXiv](https://arxiv.org/html/2406.07855v1 "VALL-E R: Robust and Efficient Zero-Shot Text-to-Speech Synthesis via Monotonic Alignment"))

## 6\. So sánh chuyên sâu (kỹ thuật và đánh giá)
### 6.1 Chỉ tiêu đánh giá
- **MOS (Mean Opinion Score)**: điểm trung bình ý kiến người nghe về naturalness (thường 1–5). Tacotron2 báo cáo MOS ~4.5 so với giọng ghi âm. ([arXiv](https://arxiv.org/abs/1712.05884 "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"))

- **Speaker similarity**: độ tương đồng giọng với mẫu gốc (trong cloning).

- **WER (Word Error Rate)** khi dùng speech-to-text để đo intelligibility.

- **Latency / real-time factor (RTF)**: thời gian suy luận so với thời lượng audio.

### 6.2 Phân tích trade-off
- **Quality vs Speed:** Autoregressive (Tacotron + WaveNet) → chất lượng rất cao nhưng latency lớn; FastSpeech + HiFi-GAN → trade-off tốt (gần bằng chất lượng, rất nhanh). ([arXiv](https://arxiv.org/abs/2006.04558 "[2006.04558] FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"))

- **Few-shot convenience vs Safety:** Few-shot TTS hiển nhiên tiện lợi nhưng dễ bị misuse (deepfake), cần biện pháp kiểm soát.

## 7\. Thách thức nghiên cứu hiện tại
- **Hiệu suất real-time** trong tài nguyên giới hạn (cpu/mobile).

- **Đa ngôn ngữ & low-resource languages**: thiếu dữ liệu huấn luyện cho nhiều ngôn ngữ; cần kỹ thuật transfer learning / multilingual modeling.

- **Cảm xúc và controlability**: điều khiển prosody, cảm xúc theo kịch bản vẫn là bài toán khó.

- **Robustness to noisy prompts**: khi audio reference có tạp âm / ghi âm kém, chất lượng cloning giảm.

- **An toàn và đạo đức**: phát hiện giọng tổng hợp, watermarking, chính sách sử dụng.

## 7. Vấn đề đạo đức trong nghiên cứu TTS
Giọng nói là một dạng dữ liệu sinh học (biometric). Vì vậy, các vấn đề đạo đức rất quan trọng:

### 7.1 Nguy cơ deepfake
- Giả mạo giọng để lừa đảo
- Tạo nội dung sai sự thật
- Voice cloning có thể giả mạo cá nhân/đánh lừa. Báo cáo và tin tức cho thấy công nghệ như VALL-E có khả năng cao tạo ra giọng giống thật đến mức rủi ro. ([the-sun.com](https://www.the-sun.com/tech/11883067/microsoft-ai-voice-clone-tts-system-public-release "Microsoft AI can now clone voices to sound perfectly 'human' in seconds - but it's too dangerous to release to public"))
### 7.2 Giải pháp nghiên cứu
- **Watermarking và detection:** Gắn watermark vào mọi giọng TTS để nhận diện. Các nghiên cứu mới không chỉ phát triển bộ phát (synthesis) mà còn phát triển watermark để đánh dấu âm thanh gốc hoặc phát hiện deepfake. Ví dụ AudioMarkNet là một hướng nghiên cứu watermarking cho audio, nhằm ngăn việc sử dụng giọng thật để fine-tune mô hình TTS độc hại. ([USENIX](https://www.usenix.org/system/files/usenixsecurity25-zong.pdf "Audio Watermarking for Deepfake Speech Detection"))
- **Quyền riêng tư dữ liệu giọng nói**: Thu thập voice dataset cần xin phép, bảo mật, và tuân thủ chính sách (GDPR/PDPA). Cảnh báo người dùng khi sử dụng voice cloning.
- **Chính sách truy xuất có trách nhiệm**: Giới hạn truy cập model mạnh, kiểm soát API, và quy định minh bạch watermark/label. Quy định về sử dụng dữ liệu giọng nói cá nhân

## 8. Kết luận
TTS đang phát triển cực nhanh qua ba hướng chính: rule-based, deep learning và voice cloning few-shot. Mỗi hướng có ưu – nhược điểm riêng và phù hợp với từng bài toán cụ thể.

Xu hướng tương lai của TTS tập trung vào:
- Đa ngôn ngữ
- Giảm dữ liệu đầu vào
- Tự nhiên, giàu cảm xúc
- Voice cloning an toàn với watermark tích hợp

Công nghệ TTS hứa hẹn đóng vai trò quan trọng trong trợ lý ảo, giáo dục, giải trí và rất nhiều ứng dụng khác trong tương lai.

## 9. Tài liệu tham khảo
- Tacotron 2 — Shen et al., *Natural TTS Synthesis by Conditioning WaveNet on Mel-Spectrograms*. ([arXiv](https://arxiv.org/abs/1712.05884 "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"))
- FastSpeech 2 — Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech*. ([arXiv](https://arxiv.org/abs/2006.04558 "[2006.04558] FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"))
- HiFi-GAN and GAN vocoders (multi-resolution discriminator work). ([gfx.cs.princeton.edu](https://gfx.cs.princeton.edu/gfx/pubs/Su_2021_HSS/Su-HiFi-GAN-2-WASPAA-2021.pdf "hifi-gan-2: studio-quality speech enhancement via generative"))
- VALL-E / VALL-E X and Voicebox (few-shot / universal speech generative models). ([arXiv](https://arxiv.org/html/2406.07855v1 "VALL-E R: Robust and Efficient Zero-Shot Text-to-Speech Synthesis via Monotonic Alignment"))
- Audio watermarking for deepfake speech detection — AudioMarkNet (USENIX). ([USENIX](https://www.usenix.org/system/files/usenixsecurity25-zong.pdf "Audio Watermarking for Deepfake Speech Detection"))