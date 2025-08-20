Dự án: Chatbot hỗ trợ truy vấn dữ liệu phản ánh khách hàng của hệ thống Mobifone

## 📌 Giới thiệu
Dự án xây dựng **Chatbot hỗ trợ truy vấn dữ liệu phản ánh khách hàng của Mobifone**, giúp người dùng có thể đặt câu hỏi bằng ngôn ngữ tự nhiên để lấy thông tin từ hệ thống nội bộ mà không cần thao tác thủ công phức tạp.  

Chatbot sử dụng **mô hình ngôn ngữ lớn (LLM)** (OpenAI/Gemini) để phân tích câu hỏi tiếng Việt, sinh câu lệnh SQL, truy vấn dữ liệu trong **Oracle DB**, và trả về kết quả dưới dạng hội thoại dễ hiểu.

## 🎯 Mục tiêu
- Tạo giao diện trò chuyện thân thiện, trực quan cho người dùng.  
- Chuyển đổi câu hỏi tự nhiên sang **SQL** chính xác.  
- Truy vấn dữ liệu nhanh chóng, chính xác và hiển thị kết quả dễ hiểu.  
- Hỗ trợ lãnh đạo, quản trị viên và chuyên viên Mobifone trong việc khai thác dữ liệu.  

## 🏗️ Kiến trúc hệ thống
Hệ thống được thiết kế theo mô hình **Client – Server** gồm 4 thành phần chính:
1. **Frontend (ReactJS):** giao diện web để nhập câu hỏi và hiển thị câu trả lời.  
2. **Backend (FastAPI):** nhận request từ frontend, gọi LLM, truy vấn DB, xử lý kết quả.  
3. **Cơ sở dữ liệu (Oracle DB):** chứa dữ liệu phản ánh khách hàng.  
4. **Mô hình LLM (Gemini/OpenAI):** phân tích câu hỏi, sinh SQL và diễn giải kết quả.  

## 🔄 Luồng hoạt động
1. Người dùng nhập câu hỏi trên giao diện web (React).  
2. Backend (FastAPI) nhận request → tìm schema liên quan → gửi prompt đến LLM.  
3. LLM sinh câu SQL phù hợp, backend thực thi trên Oracle DB.  
4. Kết quả được LLM diễn giải và trả về dưới dạng hội thoại.  
5. Frontend hiển thị kết quả cho người dùng.  

## ⚙️ Công nghệ sử dụng
- **Frontend:** ReactJS  
- **Backend:** FastAPI (Python)  
- **Database:** Oracle DB  
- **AI Model:** OpenAI GPT / Gemini (Google)  
- **Triển khai:** Docker (tùy chọn)  

